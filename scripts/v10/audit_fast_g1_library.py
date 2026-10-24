"""Audit the resumed V10 compact stream and final trainable snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/retargeting"))
from apps.humanoid_retargeter.quality import evaluate_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-db", type=Path,
                        default=ROOT / "data/v10/manifests/v10_process.sqlite")
    parser.add_argument("--fast-db", type=Path,
                        default=ROOT / "data/v10/fast_g1/process.sqlite")
    parser.add_argument("--stream", type=Path,
                        default=ROOT / "data/v10/library/accepted_stream.f16")
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "data/v10/library/accepted_compact.npy")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/v10/manifests/bones_seed_prefilter.csv")
    parser.add_argument("--geometry-samples", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/v10/manifests/fast_g1_final_audit.json")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
    expected_width = 13 + 2 * (model.njnt - 1)
    with sqlite3.connect(args.main_db) as db:
        rows = db.execute("""SELECT compact_clips.source_path,name,start,length,width,
                           clips.reason FROM compact_clips JOIN clips USING(source_path)
                           ORDER BY start""").fetchall()
    offset = 0
    for source, name, start, length, width, reason in rows:
        if start != offset or length < 2 or width != expected_width:
            raise RuntimeError(f"Invalid compact index at {source}")
        offset += length
    expected_bytes = offset * expected_width * 2
    if args.stream.stat().st_size != expected_bytes:
        raise RuntimeError("Compact stream byte count differs from the committed index")
    stream = np.memmap(args.stream, dtype=np.float16, mode="r",
                       shape=(offset, expected_width))
    if not np.isfinite(stream).all():
        raise RuntimeError("Compact stream contains nonfinite reference values")
    fast = [(source, name, start, length) for source, name, start, length, width, reason
            in rows if reason == "fast_g1_calibrated"]
    speed_failures = []
    for source, name, start, length in fast:
        joint = stream[start:start+length, 7:7+model.njnt-1].astype(np.float32)
        speed = float(np.max(np.abs(np.diff(joint, axis=0)), initial=0.0) * 50.0)
        if speed > 20.0:
            speed_failures.append((source, speed))
    if speed_failures:
        raise RuntimeError(f"Stored fast clips exceed the velocity limit: {speed_failures[:10]}")
    indices = np.linspace(0, len(fast)-1, min(args.geometry_samples, len(fast)),
                          dtype=int) if fast else []
    geometry_failures = []
    for index in indices:
        source, name, start, length = fast[index]
        pose = stream[start:start+length, :7+model.njnt-1].astype(np.float64)
        quality = evaluate_reference(model, pose, 50.0,
                                     np.zeros(length), np.zeros(length))
        if not quality["geometric_pass"]:
            geometry_failures.append((source, quality["ground_penetration_frames"],
                                      quality["self_collision_frames"]))
    if geometry_failures:
        raise RuntimeError(f"Stored sample failed geometry: {geometry_failures[:10]}")
    with sqlite3.connect(args.fast_db) as db:
        statuses = dict(db.execute("SELECT status,COUNT(*) FROM clips GROUP BY status"))
    with args.manifest.open(newline="") as f:
        candidates = {r["soma_uniform_path"] for r in csv.DictReader(f)
                      if r["status"] == "candidate"}
    done = {row[0] for row in rows}
    snapshot_meta = json.loads(args.snapshot.with_suffix(".json").read_text())
    snapshot = np.load(args.snapshot, mmap_mode="r", allow_pickle=False)
    npz_count = len(list((ROOT / "data/v10/accepted").glob("*.npz")))
    if (snapshot.shape != (snapshot_meta["frames"], expected_width)
            or len(snapshot_meta["names"]) != len(rows) + npz_count
            or len(snapshot_meta["names"]) != len(snapshot_meta["starts"])
            or len(set(snapshot_meta["names"])) != len(snapshot_meta["names"])):
        raise RuntimeError("Trainable snapshot and its index differ")
    report = {
        "eligible_metadata_candidates": len(candidates),
        "stream_clips": len(rows), "fast_stream_clips": len(fast),
        "stream_frames": offset, "stream_bytes": expected_bytes,
        "fast_status_counts": statuses,
        "sampled_geometry_clips": len(indices),
        "candidates_not_in_compact_stream_including_rejections": len(candidates - done),
        "snapshot_clips": len(snapshot_meta["names"]),
        "snapshot_frames": snapshot_meta["frames"],
        "snapshot_bytes": args.snapshot.stat().st_size,
        "dynamic_balance_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
