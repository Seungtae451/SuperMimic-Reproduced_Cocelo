"""Retarget one prefiltered BONES-SEED SOMA BVH directly to Humanoid V10.

Run with PYTHONPATH=tools/retargeting and a Python environment containing
MuJoCo, SciPy, JAX and the pinned PyRoki revision in the V10 config notes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/retargeting"))


def candidate_row(path: Path, manifest: Path) -> dict[str, str]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if path.as_posix().endswith("/" + row["soma_uniform_path"]):
                if row["status"] != "candidate":
                    raise ValueError(f"Motion was prefiltered: {row['status']} ({row['reason']})")
                return row
    raise ValueError(f"Motion is not present in the BONES-SEED manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bvh", type=Path, help="Extracted SOMA BVH from the selected archive")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/v10/manifests/bones_seed_prefilter.csv")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/robots/humanoid_v10/bones_seed_pyroki.json")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "data/v10/retargeted")
    parser.add_argument("--max-source-frames", type=int, default=None,
                        help="Use a short prefix for a quick pilot; omit for full motion")
    args = parser.parse_args()
    if args.max_source_frames is not None and args.max_source_frames < 2:
        parser.error("--max-source-frames must be at least 2")
    path = args.bvh.resolve()
    if not path.is_file():
        parser.error(f"BVH not found: {path}")
    row = candidate_row(path, args.manifest)
    from apps.external.formats import load_bvh
    from apps.humanoid_retargeter.pyroki_retargeter import retarget_motion_clip

    clip = load_bvh(path, length_scale=0.01)
    if clip.frame_count != int(row["frames"]):
        raise ValueError(f"BVH frame count {clip.frame_count} differs from manifest {row['frames']}")
    clip_dir = path.stem if args.max_source_frames is None else f"{path.stem}__pilot_{args.max_source_frames}"
    output_dir = args.output_root / Path(row["soma_uniform_path"]).parent.name / clip_dir
    result = retarget_motion_clip(
        clip, output_dir, source_motion_path=path, source_kind="BONES-SEED SOMA BVH",
        config_path=args.config, max_frames=args.max_source_frames,
        output_fps=50.0, progress=lambda message: print(message, flush=True),
    )
    print(f"NPZ: {result.npz}\nFrames: {result.frames}\nPosition RMS: {result.position_rms_m:.4f} m")


if __name__ == "__main__":
    main()
