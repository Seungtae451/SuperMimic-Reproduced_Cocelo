"""Bounded, resumable BONES-SEED G1 CSV -> V10 compact reference conversion.

This is the fast calibrated route. It does not use JAX/PyRoki or GPU memory.
Only geometrically admissible clips are appended to the existing V10 compact
stream; rejects are tracked separately so direct retargeting can retry them.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/retargeting"))
from pilot_fast_g1_to_v10 import G1_XML, V10_XML, joint_mapping, load_g1_csv, map_motion
from supermimic.data.motion_convert import forward_kinematics, resample_motion
from apps.humanoid_retargeter.quality import evaluate_reference
from apps.humanoid_retargeter.physical_constraints import project_physical_trajectory

_g1 = _v10 = _mapping = _calibration = None
_root_index = None


def _start_worker(calibration_path: str, g1_xml: str) -> None:
    global _g1, _v10, _mapping, _calibration, _root_index
    _g1 = mujoco.MjModel.from_xml_path(g1_xml)
    _v10 = mujoco.MjModel.from_xml_path(str(V10_XML))
    _mapping = joint_mapping(_g1, _v10)
    settings = json.loads(Path(calibration_path).read_text())
    _calibration = {row["v10"]: row for row in settings["joints"]}
    expected = {_v10.joint(i + 1).name for i, _, _ in _mapping}
    if set(_calibration) != expected:
        raise ValueError("Calibration joint names differ from the V10 model")
    body_names = [_v10.body(i).name for i in range(1, _v10.nbody)]
    _root_index = body_names.index("pelvis_link")


def _convert(payload: bytes) -> dict:
    try:
        raw = load_g1_csv(payload)
        qpos, motion, metrics = map_motion(raw, _g1, _v10, _mapping, _calibration)
        metrics["maximum_joint_speed_rad_s"] = float(
            np.max(np.abs(np.diff(qpos[:, 7:], axis=0)), initial=0.0) * 50.0)
        bounds = _v10.jnt_range[1:]
        source = np.zeros_like(motion.joint_pos)
        # Reconstruct the unclipped calibrated angles for the distortion gate.
        source_motion = resample_motion(raw, 120.0, 50.0)
        for target, original, sign in _mapping:
            row = _calibration[_v10.joint(target + 1).name]
            source[:, target] = (sign * source_motion.joint_pos[:, original]
                                 * row["scale"] + row["offset"])
        excess = np.maximum(bounds[:, 0] - source, source - bounds[:, 1]).clip(min=0)
        metrics["joint_limit_excess_rms_rad"] = float(np.sqrt(np.mean(excess**2)))
        if (metrics["maximum_joint_limit_excess_rad"] > 0.5
                or metrics["joint_limit_excess_rms_rad"] > 0.1):
            return {"status": "joint_distortion", "metrics": metrics}
        if metrics["maximum_root_lift_m"] > 0.15:
            return {"status": "ground_distortion", "metrics": metrics}
        if metrics["maximum_joint_speed_rad_s"] > 20.0:
            return {"status": "joint_velocity", "metrics": metrics}
        quality = evaluate_reference(_v10, qpos, 50.0,
                                     np.zeros(len(qpos)), np.zeros(len(qpos)))
        metrics.update({key: quality[key] for key in (
            "ground_penetration_frames", "self_collision_frames",
            "minimum_ground_distance_m", "minimum_self_distance_m")})
        if not quality["geometric_pass"]:
            metrics["pre_repair_self_collision_frames"] = quality["self_collision_frames"]
            # Recover small mesh penetrations while keeping pelvis orientation
            # and horizontal travel fixed. A bounded solve avoids pathological
            # motions monopolising a CPU worker.
            if (quality["ground_penetration_frames"]
                    or quality["self_collision_frames"] > len(qpos) * 0.5
                    or quality["minimum_self_distance_m"] < -0.04):
                return {"status": "geometry", "metrics": metrics}
            def timed_out(_signum, _frame):
                raise TimeoutError("Mesh projection exceeded 12 seconds")
            previous_handler = signal.signal(signal.SIGALRM, timed_out)
            signal.setitimer(signal.ITIMER_REAL, 12.0)
            try:
                repaired, repair = project_physical_trajectory(
                    _v10, qpos, np.arange(7, _v10.nq),
                    bounds[:, 0], bounds[:, 1],
                    np.zeros(len(qpos)), np.zeros(len(qpos)))
            except (RuntimeError, TimeoutError) as exc:
                metrics["projection_failure"] = str(exc)
                return {"status": "geometry", "metrics": metrics}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
            metrics.update(repair)
            if (repair["maximum_joint_correction_rad"] > 0.1
                    or repair["maximum_root_height_correction_m"] > 0.05):
                return {"status": "projection_distortion", "metrics": metrics}
            qpos = repaired
            motion.root_pos = qpos[:, :3]
            motion.joint_pos = qpos[:, 7:]
            motion.root_lin_vel_w = np.gradient(qpos[:, :3], 0.02, axis=0)
            motion.joint_vel = np.gradient(qpos[:, 7:], 0.02, axis=0)
            metrics["maximum_joint_speed_rad_s"] = float(
                np.max(np.abs(np.diff(qpos[:, 7:], axis=0)), initial=0.0) * 50.0)
            if metrics["maximum_joint_speed_rad_s"] > 20:
                return {"status": "joint_velocity", "metrics": metrics}
            quality = evaluate_reference(_v10, qpos, 50.0,
                                         np.zeros(len(qpos)), np.zeros(len(qpos)))
            if not quality["geometric_pass"]:
                return {"status": "geometry", "metrics": metrics}
            metrics["ground_penetration_frames"] = quality["ground_penetration_frames"]
            metrics["self_collision_frames"] = quality["self_collision_frames"]
            metrics["minimum_ground_distance_m"] = quality["minimum_ground_distance_m"]
            metrics["minimum_self_distance_m"] = quality["minimum_self_distance_m"]
        bodies = forward_kinematics(_v10, motion)
        root = _root_index
        block = np.concatenate((
            bodies.pos_w[:, root], bodies.quat_w[:, root], motion.joint_pos,
            bodies.lin_vel_w[:, root], bodies.ang_vel_w[:, root], motion.joint_vel,
        ), axis=1).astype(np.float16)
        if block.shape != (len(qpos), 65) or not np.isfinite(block).all():
            raise ValueError("Nonfinite or malformed compact reference")
        stored_speed = float(np.max(np.abs(np.diff(
            block[:, 7:33].astype(np.float32), axis=0)), initial=0.0) * 50.0)
        metrics["stored_joint_speed_rad_s"] = stored_speed
        if stored_speed > 20.0:
            return {"status": "joint_velocity", "metrics": metrics}
        return {"status": "accepted", "metrics": metrics, "block": block.tobytes(),
                "frames": len(qpos)}
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def _prepare_main(db: sqlite3.Connection, stream: Path, width: int) -> tuple[int, set[str]]:
    db.execute("""CREATE TABLE IF NOT EXISTS clips (
        source_path TEXT PRIMARY KEY, category TEXT NOT NULL, status TEXT NOT NULL,
        reason TEXT, frames INTEGER, position_rms_m REAL, max_joint_speed_rad_s REAL,
        elapsed_s REAL, output_path TEXT, processed_utc TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS compact_clips (
        source_path TEXT PRIMARY KEY, name TEXT NOT NULL, start INTEGER NOT NULL,
        length INTEGER NOT NULL, width INTEGER NOT NULL
    )""")
    end = 0
    for start, length, got_width in db.execute(
            "SELECT start,length,width FROM compact_clips ORDER BY start"):
        if start != end or got_width != width or length < 2:
            raise RuntimeError("Existing compact stream index is inconsistent")
        end += length
    stream.parent.mkdir(parents=True, exist_ok=True)
    stream.touch(exist_ok=True)
    expected = end * width * 2
    if stream.stat().st_size < expected:
        raise RuntimeError("Existing compact stream is truncated")
    if stream.stat().st_size > expected:
        with stream.open("r+b") as out:
            out.truncate(expected)
    done = {row[0] for row in db.execute(
        "SELECT source_path FROM clips WHERE status != 'error'")}
    db.commit()
    return end, done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=ROOT / "data/v10/downloads/g1.tar.gz")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/v10/manifests/bones_seed_prefilter.csv")
    ap.add_argument("--calibration", type=Path,
                    default=ROOT / "configs/robots/humanoid_v10/g1_to_v10_calibration.json")
    ap.add_argument("--g1-xml", type=Path, default=G1_XML)
    ap.add_argument("--main-db", type=Path, default=ROOT / "data/v10/manifests/v10_process.sqlite")
    ap.add_argument("--fast-db", type=Path, default=ROOT / "data/v10/fast_g1/process.sqlite")
    ap.add_argument("--stream", type=Path, default=ROOT / "data/v10/library/accepted_stream.f16")
    ap.add_argument("--accepted-dir", type=Path, default=ROOT / "data/v10/accepted")
    ap.add_argument("--snapshot-output", type=Path,
                    default=ROOT / "data/v10/library/accepted_compact.npy")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Maximum new candidates in this run")
    ap.add_argument("--snapshot-every", type=int, default=10000)
    ap.add_argument("--minimum-free-gb", type=float, default=15.0)
    args = ap.parse_args()
    if args.workers not in range(1, 9) or (args.limit is not None and args.limit < 1):
        ap.error("workers must be 1–8 and limit must be positive")
    if args.snapshot_every < 1 or args.minimum_free_gb < 1:
        ap.error("snapshot-every and minimum-free-gb must be positive")
    if not args.calibration.is_file() or not args.g1_xml.is_file():
        ap.error("Calibration and G1 XML must exist")
    with args.manifest.open(newline="") as stream:
        candidates = {row["soma_uniform_path"].replace("soma_uniform/bvh/", "g1/csv/")
                      .replace(".bvh", ".csv"): row
                      for row in csv.DictReader(stream) if row["status"] == "candidate"}
    args.fast_db.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.stream.parent / (args.stream.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another V10 compact-stream converter is running") from exc
        main_db = sqlite3.connect(args.main_db)
        fast_db = sqlite3.connect(args.fast_db)
        fast_db.execute("""CREATE TABLE IF NOT EXISTS clips (
            source_path TEXT PRIMARY KEY, status TEXT NOT NULL, reason TEXT,
            metrics_json TEXT, elapsed_s REAL,
            processed_utc TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        fast_db.execute("""CREATE TABLE IF NOT EXISTS run_config (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        )""")
        calibration_digest = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
        previous = fast_db.execute("SELECT value FROM run_config WHERE key='calibration_sha256'").fetchone()
        if previous and previous[0] != calibration_digest:
            raise RuntimeError("Fast conversion database was created with a different calibration")
        fast_db.execute("INSERT OR IGNORE INTO run_config VALUES (?,?)",
                        ("calibration_sha256", calibration_digest))
        fast_db.commit()
        model = mujoco.MjModel.from_xml_path(str(V10_XML))
        width = 13 + 2 * (model.njnt - 1)
        end, done_main = _prepare_main(main_db, args.stream, width)
        done_fast = {row[0] for row in fast_db.execute(
            "SELECT source_path FROM clips WHERE status != 'error'")}
        accepted_stems = {path.stem for path in args.accepted_dir.glob("*.npz")}
        print(f"Candidates {len(candidates)}; existing stream frames {end}; "
              f"main done {len(done_main)}; fast done {len(done_fast)}", flush=True)
        submitted = processed = accepted = rejected = errors = 0
        started = time.monotonic()
        pending: dict[futures.Future, tuple[str, dict, float]] = {}

        def snapshot() -> None:
            subprocess.run([sys.executable, str(ROOT / "scripts/v10/build_compact_library.py"),
                            "--accepted-dir", str(args.accepted_dir),
                            "--stream", str(args.stream), "--database", str(args.main_db),
                            "--manifest", str(args.manifest),
                            "--output", str(args.snapshot_output)], check=True)

        def receive_one() -> None:
            nonlocal end, processed, accepted, rejected, errors
            complete, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for future in complete:
                name, row, sent = pending.pop(future)
                result = future.result()
                status = result["status"]
                metrics = result.get("metrics", {})
                elapsed = time.monotonic() - sent
                reason = result.get("reason", "")
                soma = row["soma_uniform_path"]
                if status == "accepted":
                    block = result["block"]
                    frames = result["frames"]
                    if len(block) != frames * width * 2:
                        raise RuntimeError(f"Wrong block size for {name}")
                    if shutil.disk_usage(args.stream.parent).free < args.minimum_free_gb * 1e9:
                        raise RuntimeError("Free disk is below the requested safety floor")
                    with args.stream.open("ab") as out:
                        out.write(block)
                        out.flush()
                        os.fsync(out.fileno())
                    stem = f"{Path(name).parent.name}__{Path(name).stem}"
                    main_db.execute("INSERT INTO compact_clips VALUES (?,?,?,?,?)",
                                    (soma, stem, end, frames, width))
                    main_db.execute("""INSERT OR REPLACE INTO clips
                        (source_path,category,status,reason,frames,position_rms_m,
                         max_joint_speed_rad_s,elapsed_s,output_path)
                         VALUES (?,?,?,?,?,?,?,?,?)""",
                        (soma, row["category"], "accepted", "fast_g1_calibrated",
                         frames, None, metrics.get("maximum_joint_speed_rad_s"), elapsed,
                         str(args.stream.resolve())))
                    main_db.commit()
                    end += frames
                    accepted += 1
                elif status == "error":
                    errors += 1
                else:
                    rejected += 1
                fast_db.execute("""INSERT OR REPLACE INTO clips
                    (source_path,status,reason,metrics_json,elapsed_s)
                    VALUES (?,?,?,?,?)""",
                    (soma, status, reason, json.dumps(metrics, allow_nan=False), elapsed))
                fast_db.commit()
                processed += 1
                if processed <= 10 or processed % 100 == 0:
                    rate = processed / max(1, time.monotonic() - started)
                    print(f"{processed}/{submitted} {status}: {name}; accepted={accepted}, "
                          f"rejected={rejected}, errors={errors}, rate={rate:.1f}/s, "
                          f"frames={end}", flush=True)
                if status == "error" and errors >= 20:
                    raise RuntimeError("At least 20 fast conversion errors; stopping")
                if accepted and accepted % args.snapshot_every == 0:
                    snapshot()

        try:
            with futures.ProcessPoolExecutor(
                    max_workers=args.workers, initializer=_start_worker,
                    initargs=(str(args.calibration), str(args.g1_xml))) as pool:
                with tarfile.open(args.archive, mode="r|gz") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        name = member.name.removeprefix("./")
                        row = candidates.get(name)
                        if row is None:
                            continue
                        soma = row["soma_uniform_path"]
                        stem = f"{Path(name).parent.name}__{Path(name).stem}"
                        if soma in done_main or soma in done_fast or stem in accepted_stems:
                            continue
                        if args.limit is not None and submitted >= args.limit:
                            break
                        while len(pending) >= args.workers * 2:
                            receive_one()
                        input_file = tar.extractfile(member)
                        if input_file is None:
                            continue
                        sent = time.monotonic()
                        pending[pool.submit(_convert, input_file.read())] = (name, row, sent)
                        submitted += 1
                    while pending:
                        receive_one()
            snapshot()
            print(f"Complete: submitted={submitted}, accepted={accepted}, "
                  f"rejected={rejected}, errors={errors}, elapsed_s={time.monotonic()-started:.1f}",
                  flush=True)
        finally:
            main_db.close()
            fast_db.close()


if __name__ == "__main__":
    main()
