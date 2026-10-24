"""Stream the BONES-SEED archive into resumable V10 PyRoki retargeting.

The 45 GB gzip archive is scanned once per invocation. Only one source BVH and
its temporary solver output exist on disk at a time. Accepted named reference
NPZ files and an SQLite audit are retained. This does not train a policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/retargeting"))


def audit(npz: Path, metadata: Path, source_frames: int, model: mujoco.MjModel) -> tuple[str, float, float, int]:
    details = json.loads(metadata.read_text())
    quality = details.get("quality", {})
    rms = float(details.get("position_rms_m", math.nan))
    speed = float(quality.get("joint_velocity_max_rad_s", math.nan))
    with np.load(npz, allow_pickle=False) as data:
        names = [str(x) for x in data["joint_names"]]
        bodies = [str(x) for x in data["body_names"]]
        pos = data["joint_pos"]
        vel = data["joint_vel"]
        frames = len(pos)
        fps = float(data["fps"])
        expected_names = [model.joint(i).name for i in range(1, model.njnt)]
        expected_bodies = [model.body(i).name for i in range(1, model.nbody)]
        if names != expected_names or bodies != expected_bodies:
            return "schema_mismatch", rms, speed, frames
        if not np.isclose(fps, 50.0) or frames < 2:
            return "fps_or_length", rms, speed, frames
        source_fps = float(details["source_fps"])
        expected_frames = math.floor((source_frames - 1) * 50.0 / source_fps + 1e-9) + 1
        if abs(frames - expected_frames) > 1:
            return "partial_clip", rms, speed, frames
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            return "nonfinite", rms, speed, frames
        if np.any(pos < model.jnt_range[1:, 0] - 1e-6) or np.any(pos > model.jnt_range[1:, 1] + 1e-6):
            return "joint_limit", rms, speed, frames
    if not quality.get("geometric_pass", False):
        return "geometric_fail", rms, speed, frames
    if not math.isfinite(rms) or rms > 0.10:
        return "position_error", rms, speed, frames
    if not math.isfinite(speed) or speed > 20.0:
        return "joint_velocity", rms, speed, frames
    return "accepted", rms, speed, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/v10/downloads/soma_uniform.tar.gz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/v10/manifests/bones_seed_prefilter.csv")
    parser.add_argument("--database", type=Path, default=ROOT / "data/v10/manifests/v10_process.sqlite")
    parser.add_argument("--accepted-dir", type=Path, default=ROOT / "data/v10/accepted")
    parser.add_argument("--limit", type=int, default=20, help="New clips to process; use --all for all remaining")
    parser.add_argument("--all", action="store_true", help="Process all remaining candidates")
    parser.add_argument("--category", action="append", default=[], help="Restrict to a metadata category")
    parser.add_argument("--name-contains", default="", help="Restrict to a case-insensitive motion-name substring")
    parser.add_argument("--max-per-category", type=int, default=None)
    parser.add_argument("--minimum-free-gb", type=float, default=10.0)
    args = parser.parse_args()
    if args.limit < 1 or (args.max_per_category is not None and args.max_per_category < 1):
        parser.error("limit and max-per-category must be positive")

    from apps.external.formats import load_bvh
    from apps.humanoid_retargeter.pyroki_retargeter import retarget_motion_clip

    with args.manifest.open(newline="") as stream:
        candidates = {row["soma_uniform_path"]: row for row in csv.DictReader(stream)
                      if row["status"] == "candidate" and (not args.category or row["category"] in args.category)
                      and args.name_contains.lower() in row["move_name"].lower()}
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
    args.database.parent.mkdir(parents=True, exist_ok=True)
    args.accepted_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.database)
    db.execute("""CREATE TABLE IF NOT EXISTS clips (
        source_path TEXT PRIMARY KEY, category TEXT NOT NULL, status TEXT NOT NULL,
        reason TEXT, frames INTEGER, position_rms_m REAL, max_joint_speed_rad_s REAL,
        elapsed_s REAL, output_path TEXT, processed_utc TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    done = {row[0] for row in db.execute("SELECT source_path FROM clips WHERE status != 'error'")}
    selected = processed = accepted = 0
    per_category: Counter[str] = Counter()
    failures = 0
    limit = float("inf") if args.all else args.limit
    try:
        with tarfile.open(args.archive, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name.removeprefix("./")
                row = candidates.get(name)
                if row is None or name in done:
                    continue
                category = row["category"]
                if args.max_per_category is not None and per_category[category] >= args.max_per_category:
                    continue
                parts = Path(name).parts
                if not parts or parts[0] != "soma_uniform" or ".." in parts or Path(name).is_absolute():
                    raise ValueError(f"Unsafe archive member: {name}")
                selected += 1
                per_category[category] += 1
                output = args.accepted_dir / f"{Path(name).parent.name}__{Path(name).stem}.npz"
                if output.exists():
                    sidecar = output.resolve().with_suffix(".json")
                    if sidecar.is_file():
                        status, rms, speed, frames = audit(output, sidecar, int(row["frames"]), model)
                    else:
                        status, rms, speed, frames = "unverified_existing_output", math.nan, math.nan, 0
                    reason, elapsed = "existing_output", 0.0
                else:
                    if shutil.disk_usage(args.accepted_dir).free < args.minimum_free_gb * 1e9:
                        raise RuntimeError("Free disk space is below the requested safety floor")
                    start = time.monotonic()
                    try:
                        with tempfile.TemporaryDirectory(prefix="v10-motion-") as tmp:
                            source = Path(tmp) / name
                            source.parent.mkdir(parents=True, exist_ok=True)
                            payload = tar.extractfile(member)
                            if payload is None:
                                raise ValueError(f"Unable to read archive member {name}")
                            with source.open("wb") as dst:
                                shutil.copyfileobj(payload, dst)
                            if source.stat().st_size != member.size:
                                raise ValueError("Incomplete BVH extraction")
                            clip = load_bvh(source, length_scale=0.01)
                            if clip.frame_count != int(row["frames"]):
                                raise ValueError("BVH frame count differs from metadata")
                            work = Path(tmp) / "retargeted"
                            result = retarget_motion_clip(
                                clip, work, source_motion_path=source,
                                source_kind="BONES-SEED SOMA BVH", config_path=ROOT / "configs/robots/humanoid_v10/bones_seed_pyroki.json",
                                output_fps=50.0,
                            )
                            status, rms, speed, frames = audit(result.npz, result.npz.with_suffix(".json"), clip.frame_count, model)
                            reason = "" if status == "accepted" else status
                            if status == "accepted":
                                partial = output.with_suffix(".npz.part")
                                sidecar = output.with_suffix(".json")
                                sidecar_partial = sidecar.with_suffix(".json.part")
                                try:
                                    shutil.copy2(result.npz.with_suffix(".json"), sidecar_partial)
                                    os.replace(sidecar_partial, sidecar)
                                    shutil.copy2(result.npz, partial)
                                    os.replace(partial, output)
                                finally:
                                    partial.unlink(missing_ok=True)
                                    sidecar_partial.unlink(missing_ok=True)
                    except Exception as exc:
                        status, reason, frames, rms, speed = "error", f"{type(exc).__name__}: {exc}", 0, math.nan, math.nan
                    elapsed = time.monotonic() - start
                db.execute("""INSERT OR REPLACE INTO clips
                    (source_path, category, status, reason, frames, position_rms_m,
                     max_joint_speed_rad_s, elapsed_s, output_path)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (name, category, status, reason, frames, rms, speed, elapsed,
                     str(output.resolve()) if status == "accepted" else ""))
                db.commit()
                processed += 1
                accepted += status == "accepted"
                failures = failures + 1 if status == "error" else 0
                print(f"{processed}/{limit}: {status} {name} ({elapsed:.1f}s) {reason}", flush=True)
                if failures >= 5:
                    raise RuntimeError("Five consecutive retargeting errors; stopping instead of skipping the dataset")
                if processed >= limit:
                    break
    finally:
        db.close()
    print(f"Processed {processed} new clips; accepted {accepted}; database: {args.database}")


if __name__ == "__main__":
    main()
