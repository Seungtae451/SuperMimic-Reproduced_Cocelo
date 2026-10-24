"""Retarget eligible BONES-SEED motions with GPU-assigned PyRoki workers.

The archive is streamed once. Accepted references are appended directly to a
compact float16 row store, so tens of thousands of intermediate NPZ files do
not fill the disk. ``build_compact_library.py`` publishes a training snapshot.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import math
import multiprocessing as mp
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/retargeting"))
from process_bones_seed import audit


def _worker(slot: int, gpu: int, jobs: mp.Queue, results: mp.Queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        import jax
        if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
            raise RuntimeError(f"GPU {gpu} is unavailable to JAX: {jax.devices()}")
        from apps.external.formats import load_bvh
        from apps.humanoid_retargeter.pyroki_retargeter import retarget_motion_clip

        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
        expected_joints = [model.joint(i).name for i in range(1, model.njnt)]
        expected_bodies = [model.body(i).name for i in range(1, model.nbody)]
        root = expected_bodies.index("pelvis_link")
        width = 13 + 2 * len(expected_joints)
        results.put(("ready", slot, gpu, width))
    except Exception as exc:
        results.put(("startup_error", slot, gpu, f"{type(exc).__name__}: {exc}"))
        return

    while True:
        job = jobs.get()
        if job is None:
            return
        name, source_path, source_frames = job
        source = Path(source_path)
        start = time.monotonic()
        try:
            clip = load_bvh(source, length_scale=0.01)
            if clip.frame_count != source_frames:
                raise ValueError("BVH frame count differs from metadata")
            result = retarget_motion_clip(
                clip, source.parent / "retargeted", source_motion_path=source,
                source_kind="BONES-SEED SOMA BVH",
                config_path=ROOT / "configs/robots/humanoid_v10/bones_seed_pyroki.json",
                output_fps=50.0,
            )
            status, rms, speed, frames = audit(
                result.npz, result.npz.with_suffix(".json"), clip.frame_count, model
            )
            compact_path = ""
            if status == "accepted":
                with np.load(result.npz, allow_pickle=False) as data:
                    block = np.concatenate((
                        data["body_pos_w"][:, root], data["body_quat_w"][:, root],
                        data["joint_pos"], data["body_lin_vel_w"][:, root],
                        data["body_ang_vel_w"][:, root], data["joint_vel"],
                    ), axis=1)
                if block.shape != (frames, width) or not np.isfinite(block).all():
                    raise ValueError("Invalid compact reference block")
                compact_path = str(source.parent / "accepted.f16")
                compact_block = block.astype(np.float16)
                if not np.isfinite(compact_block).all():
                    raise ValueError("Nonfinite values after float16 conversion")
                compact_block.tofile(compact_path)
            reason = "" if status == "accepted" else status
        except Exception as exc:
            status, reason, frames, rms, speed, compact_path = (
                "error", f"{type(exc).__name__}: {exc}", 0, math.nan, math.nan, ""
            )
        results.put(("clip", slot, gpu, name, status, reason, frames, rms, speed,
                     time.monotonic() - start, compact_path))


def _prepare_db(db: sqlite3.Connection, stream_path: Path, width: int) -> int:
    db.execute("""CREATE TABLE IF NOT EXISTS clips (
        source_path TEXT PRIMARY KEY, category TEXT NOT NULL, status TEXT NOT NULL,
        reason TEXT, frames INTEGER, position_rms_m REAL, max_joint_speed_rad_s REAL,
        elapsed_s REAL, output_path TEXT, processed_utc TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS compact_clips (
        source_path TEXT PRIMARY KEY, name TEXT NOT NULL, start INTEGER NOT NULL,
        length INTEGER NOT NULL, width INTEGER NOT NULL
    )""")
    rows = db.execute("SELECT start, length, width FROM compact_clips ORDER BY start").fetchall()
    end = 0
    for start, length, row_width in rows:
        if start != end or row_width != width or length < 2:
            raise RuntimeError("Compact stream index is inconsistent")
        end += length
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    if not stream_path.exists():
        stream_path.touch()
    required = end * width * np.dtype(np.float16).itemsize
    actual = stream_path.stat().st_size
    if actual < required:
        raise RuntimeError("Compact stream is shorter than its committed index")
    if actual > required:
        with stream_path.open("r+b") as f:
            f.truncate(required)  # Remove an uncommitted tail after a crash.
    db.commit()
    return end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/v10/downloads/soma_uniform.tar.gz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/v10/manifests/bones_seed_prefilter.csv")
    parser.add_argument("--database", type=Path, default=ROOT / "data/v10/manifests/v10_process.sqlite")
    parser.add_argument("--stream", type=Path, default=ROOT / "data/v10/library/accepted_stream.f16")
    parser.add_argument("--accepted-dir", type=Path, default=ROOT / "data/v10/accepted")
    parser.add_argument("--snapshot-output", type=Path,
                        default=ROOT / "data/v10/library/accepted_compact.npy")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--limit", type=int, default=None, help="New clips to process; omitted means all remaining")
    parser.add_argument("--snapshot-every", type=int, default=1000,
                        help="Refresh the trainable .npy library after this many accepted clips")
    parser.add_argument("--minimum-free-gb", type=float, default=10.0)
    parser.add_argument("--jax-cache-dir", type=Path,
                        default=Path(os.environ.get("JAX_COMPILATION_CACHE_DIR",
                                                    ROOT / "data/v10/cache/jax_compilation")),
                        help="Persistent JAX compilation cache shared by GPU workers")
    args = parser.parse_args()
    if (not args.gpus or any(gpu < 0 for gpu in args.gpus)
            or (args.limit is not None and args.limit < 1) or args.snapshot_every < 1):
        parser.error("GPU indices must be nonnegative; limit and snapshot-every must be positive")

    # PyRoki builds shape-specific JAX programs. Persist their compiled binaries
    # so later clips and restarted workers can reuse them across processes.
    args.jax_cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(args.jax_cache_dir.resolve())
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")

    with args.manifest.open(newline="") as f:
        candidates = {r["soma_uniform_path"]: r for r in csv.DictReader(f) if r["status"] == "candidate"}
    args.stream.parent.mkdir(parents=True, exist_ok=True)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.stream.parent / (args.stream.name + ".lock")).open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("Another compact-stream converter is already running") from exc
    db = sqlite3.connect(args.database)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
    width = 13 + 2 * (model.njnt - 1)
    end = _prepare_db(db, args.stream, width)
    done = {r[0] for r in db.execute("SELECT source_path FROM clips WHERE status != 'error'")}
    # The exploratory accepted library predates the process database. Preserve
    # those NPZ clips without appending a duplicate compact-stream entry.
    accepted_stems = {path.stem for path in args.accepted_dir.glob("*.npz")}
    done.update(name for name in candidates
                if f"{Path(name).parent.name}__{Path(name).stem}" in accepted_stems)
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    queues = [ctx.Queue(maxsize=2) for _ in args.gpus]
    workers = [ctx.Process(target=_worker, args=(slot, gpu, jobs, results), daemon=True)
               for slot, (gpu, jobs) in enumerate(zip(args.gpus, queues))]
    for worker in workers:
        worker.start()
    pending: dict[str, tuple[dict, Path]] = {}
    selected = processed = accepted = failures = 0
    max_pending = len(workers) * 2

    def publish() -> None:
        subprocess.run([
            sys.executable, str(ROOT / "scripts/v10/build_compact_library.py"),
            "--accepted-dir", str(args.accepted_dir), "--output", str(args.snapshot_output),
            "--stream", str(args.stream), "--database", str(args.database),
            "--manifest", str(args.manifest),
        ], check=True)

    def receive() -> None:
        nonlocal processed, accepted, failures, end
        waited = 0
        while True:
            try:
                item = results.get(timeout=60)
                break
            except queue.Empty:
                waited += 60
                dead = [(slot, gpu, worker.exitcode)
                        for slot, (gpu, worker) in enumerate(zip(args.gpus, workers))
                        if not worker.is_alive()]
                if dead:
                    raise RuntimeError(f"GPU worker exited with pending clips: {dead}")
                if waited % 300 == 0:
                    print(f"Waiting for {len(pending)} long motions ({waited}s)", flush=True)
        if item[0] != "clip":
            raise RuntimeError(f"Unexpected GPU worker message: {item}")
        _, slot, gpu, name, status, reason, frames, rms, speed, elapsed, compact = item
        row, temp = pending.pop(name)
        try:
            output = ""
            if status == "accepted":
                source = Path(compact)
                if source.stat().st_size != frames * width * 2:
                    raise RuntimeError("Compact block byte count differs from accepted frames")
                if shutil.disk_usage(args.stream.parent).free < args.minimum_free_gb * 1e9:
                    raise RuntimeError("Free disk space is below the requested safety floor")
                with args.stream.open("ab") as dst, source.open("rb") as src:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
                db.execute("INSERT INTO compact_clips VALUES (?, ?, ?, ?, ?)",
                           (name, f"{Path(name).parent.name}__{Path(name).stem}", end, frames, width))
                end += frames
                output = str(args.stream.resolve())
            db.execute("""INSERT OR REPLACE INTO clips
                (source_path, category, status, reason, frames, position_rms_m,
                 max_joint_speed_rad_s, elapsed_s, output_path)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, row["category"], status, reason, frames, rms, speed, elapsed, output))
            db.commit()
            processed += 1
            accepted += status == "accepted"
            failures = failures + 1 if status == "error" else 0
            print(f"{processed}/{selected}: worker{slot}/GPU{gpu} {status} {name} ({elapsed:.1f}s) {reason}", flush=True)
            if accepted and accepted % args.snapshot_every == 0:
                print(f"Publishing trainable library snapshot after {accepted} accepted clips", flush=True)
                publish()
            if failures >= 20:
                raise RuntimeError("Twenty consecutive retargeting errors; stopping")
        finally:
            shutil.rmtree(temp)

    try:
        ready = set()
        while len(ready) < len(workers):
            try:
                item = results.get(timeout=90)
            except queue.Empty as exc:
                raise RuntimeError("GPU worker did not start") from exc
            if item[0] == "startup_error":
                raise RuntimeError(f"worker{item[1]}/GPU{item[2]} startup: {item[3]}")
            if item[0] != "ready" or item[3] != width:
                raise RuntimeError(f"Unexpected worker startup message: {item}")
            ready.add(item[1])
        print(f"Workers ready on GPUs {args.gpus}; completed before run: {len(done)}", flush=True)
        with tarfile.open(args.archive, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name.removeprefix("./")
                row = candidates.get(name)
                if row is None or name in done:
                    continue
                parts = Path(name).parts
                if not parts or parts[0] != "soma_uniform" or ".." in parts or Path(name).is_absolute():
                    raise ValueError(f"Unsafe archive member: {name}")
                if args.limit is not None and selected >= args.limit:
                    break
                while len(pending) >= max_pending:
                    receive()
                if shutil.disk_usage(args.stream.parent).free < args.minimum_free_gb * 1e9:
                    raise RuntimeError("Free disk space is below the requested safety floor")
                temp = Path(tempfile.mkdtemp(prefix="v10-motion-"))
                try:
                    source = temp / name
                    source.parent.mkdir(parents=True)
                    payload = tar.extractfile(member)
                    if payload is None:
                        raise ValueError(f"Unable to read archive member {name}")
                    with source.open("wb") as dst:
                        shutil.copyfileobj(payload, dst)
                    if source.stat().st_size != member.size:
                        raise ValueError("Incomplete BVH extraction")
                    pending[name] = (row, temp)
                    queues[selected % len(queues)].put((name, str(source), int(row["frames"])))
                    selected += 1
                except BaseException:
                    shutil.rmtree(temp)
                    raise
        while pending:
            receive()
        if processed:
            print("Publishing final trainable library snapshot", flush=True)
            publish()
    finally:
        for jobs in queues:
            try:
                jobs.put_nowait(None)
            except queue.Full:
                pass
        for worker in workers:
            worker.join(timeout=3)
            if worker.is_alive():
                worker.terminate()
        for _, temp in pending.values():
            shutil.rmtree(temp, ignore_errors=True)
        db.close()
        lock.close()
    print(f"Processed {processed} clips; accepted {accepted}; compact stream: {args.stream}", flush=True)


if __name__ == "__main__":
    main()
