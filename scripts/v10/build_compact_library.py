"""Pack accepted V10 references into one memory-mapped fp16 training library."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-dir", type=Path, default=ROOT / "data/v10/accepted")
    parser.add_argument("--output", type=Path, default=ROOT / "data/v10/library/accepted_compact.npy")
    parser.add_argument("--stream", type=Path, default=ROOT / "data/v10/library/accepted_stream.f16")
    parser.add_argument("--database", type=Path, default=ROOT / "data/v10/manifests/v10_process.sqlite")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/v10/manifests/bones_seed_prefilter.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    with args.manifest.open(newline="") as f:
        allowed_paths = {row["soma_uniform_path"] for row in csv.DictReader(f)
                         if row["status"] == "candidate"}
    allowed_names = {f"{Path(path).parent.name}__{Path(path).stem}" for path in allowed_paths}
    files = sorted(path for path in args.accepted_dir.glob("*.npz")
                   if path.stem in allowed_names)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
    expected_joints = [model.joint(i).name for i in range(1, model.njnt)]
    expected_bodies = [model.body(i).name for i in range(1, model.nbody)]
    width = 13 + 2 * len(expected_joints)
    streamed: list[tuple[str, int, int]] = []
    if args.stream.is_file() and args.database.is_file():
        with sqlite3.connect(args.database) as db:
            exists = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compact_clips'").fetchone()
            if exists:
                streamed = [(name, start, length) for source, name, start, length in
                            db.execute("SELECT source_path, name, start, length FROM compact_clips ORDER BY start")
                            if source in allowed_paths]
    existing_names = {path.stem for path in files}
    if any(name in existing_names for name, _, _ in streamed):
        raise ValueError("A motion exists in both accepted NPZ and compact stream")
    if len({name for name, _, _ in streamed}) != len(streamed):
        raise ValueError("Duplicate names in compact stream")
    if args.limit is not None:
        files = files[: args.limit]
        streamed = streamed[: max(0, args.limit - len(files))]
    if not files and not streamed:
        raise SystemExit("No accepted V10 references")
    names, starts, lengths = [], [], []
    total = 0
    for i, path in enumerate(files):
        with np.load(path, allow_pickle=False) as data:
            if [str(x) for x in data["joint_names"]] != expected_joints:
                raise ValueError(f"{path}: joint order mismatch")
            if [str(x) for x in data["body_names"]] != expected_bodies:
                raise ValueError(f"{path}: body order mismatch")
            if not np.isclose(float(data["fps"]), 50.0):
                raise ValueError(f"{path}: fps is not 50")
            n = int(data["joint_pos"].shape[0])
        names.append(path.stem)
        starts.append(total)
        lengths.append(n)
        total += n
        if (i + 1) % 10000 == 0:
            print(f"Indexed {i+1}/{len(files)}", flush=True)

    streamed_end = 0
    for name, offset, n in streamed:
        if offset < streamed_end or n < 2:
            raise ValueError("Compact stream index is inconsistent")
        streamed_end = offset + n
        names.append(name)
        starts.append(total)
        lengths.append(n)
        total += n
    if streamed and args.stream.stat().st_size < streamed_end * width * 2:
        raise ValueError("Compact stream is shorter than its index")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".npy.part")
    meta_tmp = args.output.with_suffix(".json.part")
    try:
        ref = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                        shape=(total, width))
        root = expected_bodies.index("pelvis_link")
        for i, (path, start, n) in enumerate(zip(files, starts, lengths)):
            with np.load(path, allow_pickle=False) as data:
                block = np.concatenate((
                    data["body_pos_w"][:, root], data["body_quat_w"][:, root],
                    data["joint_pos"], data["body_lin_vel_w"][:, root],
                    data["body_ang_vel_w"][:, root], data["joint_vel"],
                ), axis=1)
            if block.shape != (n, ref.shape[1]) or not np.isfinite(block).all():
                raise ValueError(f"{path}: invalid reference block")
            ref[start:start+n] = block.astype(np.float16)
            if (i + 1) % 10000 == 0:
                ref.flush()
                print(f"Packed {i+1}/{len(files)}", flush=True)
        if streamed:
            source = np.memmap(args.stream, mode="r", dtype=np.float16,
                               shape=(streamed_end, width))
            for i, (_, offset, n) in enumerate(streamed):
                block = source[offset:offset+n]
                if not np.isfinite(block).all():
                    raise ValueError(f"Compact stream clip {i} contains nonfinite values")
                ref[starts[len(files)+i]:starts[len(files)+i]+n] = block
                if (i + 1) % 10000 == 0:
                    ref.flush()
                    print(f"Packed {i+1}/{len(streamed)} stream clips", flush=True)
            del source
        ref.flush()
        del ref
        metadata = {"robot": "v10", "root_body": "pelvis_link", "fps": 50.0,
                    "names": names, "starts": starts, "lengths": lengths, "frames": total,
                    "reference_dimensions": width}
        meta_tmp.write_text(json.dumps(metadata))
        os.replace(tmp, args.output)
        os.replace(meta_tmp, args.output.with_suffix(".json"))
    finally:
        tmp.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)
    print(f"Packed {len(files) + len(streamed)} clips, {total} frames, {args.output.stat().st_size / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
