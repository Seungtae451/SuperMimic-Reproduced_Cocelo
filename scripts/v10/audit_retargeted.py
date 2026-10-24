"""Audit V10 retargeted BONES-SEED clips and prepare a flat accepted library.

Only complete clips that pass the PyRoki geometry check and basic tracking
thresholds are linked into ``data/v10/accepted``. Dynamic tracking is still a
separate simulation/training-stage check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/retargeting"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retargeted-dir", type=Path, default=ROOT / "data/v10/retargeted")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/v10/manifests/bones_seed_prefilter.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/v10/manifests/v10_motion_quality.csv")
    parser.add_argument("--accepted-dir", type=Path, default=ROOT / "data/v10/accepted")
    parser.add_argument("--max-position-rms-m", type=float, default=0.10)
    parser.add_argument("--max-joint-velocity-rad-s", type=float, default=20.0)
    parser.add_argument("--link-accepted", action="store_true")
    args = parser.parse_args()
    if args.max_position_rms_m <= 0 or args.max_joint_velocity_rad_s <= 0:
        parser.error("Quality thresholds must be positive")

    by_path = {}
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_path[row["soma_uniform_path"]] = row
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
    expected_joints = [model.joint(i).name for i in range(1, model.njnt)]
    expected_bodies = [model.body(i).name for i in range(1, model.nbody)]
    limits = model.jnt_range[1:]
    records = []
    for npz in sorted(args.retargeted_dir.glob("**/external_50hz_retargeted.npz")):
        name = npz.parent.parent.name + "__" + npz.parent.name
        reason = ""
        rms = velocity = acceleration = math.nan
        frames = 0
        try:
            metadata = json.loads(npz.with_suffix(".json").read_text())
            source = Path(metadata["source_motion"])
            match = next((row for path, row in by_path.items()
                          if source.as_posix().endswith("/" + path)), None)
            if match is None or match["status"] != "candidate":
                reason = "not_prefilter_candidate"
            else:
                with np.load(npz, allow_pickle=False) as archive:
                    joints = archive["joint_names"].tolist()
                    bodies = archive["body_names"].tolist()
                    positions = archive["joint_pos"]
                    velocities = archive["joint_vel"]
                    frames = int(len(positions))
                    fps = float(archive["fps"].item())
                    if joints != expected_joints or bodies != expected_bodies:
                        reason = "robot_schema_mismatch"
                    elif not np.isclose(fps, 50.0) or frames < 2:
                        reason = "invalid_fps_or_frames"
                    elif not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                        reason = "nonfinite_values"
                    elif np.any(positions < limits[:, 0] - 1e-6) or np.any(positions > limits[:, 1] + 1e-6):
                        reason = "joint_limit"
                    else:
                        # The source count is checked against the BVH itself to
                        # reject pilot prefixes before adding a training clip.
                        from apps.external.formats import load_bvh
                        source_count = load_bvh(source).frame_count
                        if source_count != int(match["frames"]):
                            reason = "source_frame_mismatch"
                        else:
                            expected_frames = int(math.floor((source_count - 1) * 50.0 /
                                float(metadata["source_fps"]) + 1e-9)) + 1
                            if abs(frames - expected_frames) > 1:
                                reason = "partial_clip"
                rms = float(metadata.get("position_rms_m", math.nan))
                velocity = float(metadata.get("quality", {}).get("joint_velocity_max_rad_s", math.nan))
                acceleration = float(metadata.get("quality", {}).get("joint_acceleration_max_rad_s2", math.nan))
                if not reason and not metadata.get("quality", {}).get("geometric_pass", False):
                    reason = "geometric_fail"
                if not reason and (not math.isfinite(rms) or rms > args.max_position_rms_m):
                    reason = "position_error"
                if not reason and (not math.isfinite(velocity) or velocity > args.max_joint_velocity_rad_s):
                    reason = "joint_velocity"
        except (OSError, ValueError, KeyError, IndexError) as exc:
            reason = f"invalid_artifact:{type(exc).__name__}"
        keep = int(not reason)
        records.append(dict(name=name, keep=keep, reason=reason,
                            frames=frames, position_rms_m=rms,
                            joint_velocity_max_rad_s=velocity, joint_acceleration_max_rad_s2=acceleration,
                            dynamic_balance_verified=0, npz=str(npz.resolve())))
        if keep and args.link_accepted:
            args.accepted_dir.mkdir(parents=True, exist_ok=True)
            target = args.accepted_dir / f"{name}.npz"
            if target.exists() or target.is_symlink():
                if target.resolve() != npz.resolve():
                    raise FileExistsError(f"Refusing to replace existing accepted clip: {target}")
            else:
                target.symlink_to(npz.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "keep", "reason", "frames",
            "position_rms_m", "joint_velocity_max_rad_s",
            "joint_acceleration_max_rad_s2", "dynamic_balance_verified", "npz"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Audited {len(records)} clips; accepted {sum(r['keep'] for r in records)}. {args.output}")


if __name__ == "__main__":
    main()
