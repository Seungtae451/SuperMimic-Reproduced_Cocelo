"""Search small V10 root-pitch/foot-grounding corrections by PD replay.

A complete PD replay is a conservative pilot for a balance-corrected motion,
not proof that a SuperMimic policy will reproduce it. Retain the source NPZ;
export only complete candidates, ordered by the smallest absolute pitch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from apply_root_pitch_trial import correct, corrected_qpos
from check_dynamic_tracking import replay_reference


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PITCHES = (-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-dir", type=Path, default=ROOT / "data/v10/accepted")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/v10/manifests/v10_root_pitch_search.csv")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--pitches", type=float, nargs="+", default=list(DEFAULT_PITCHES))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clips", nargs="+", default=None,
                        help="Optional NPZ stems to inspect")
    parser.add_argument("--ground-height-m", type=float, default=0.005)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("max-frames must be at least two")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if not args.pitches or any(not np.isfinite(x) or abs(x) > 0.2 for x in args.pitches):
        parser.error("pitches must be finite and within ±0.2 rad")

    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
    control = json.loads((ROOT / "configs/robots/humanoid_v10/control_profile.json").read_text())
    substeps = round(1.0 / (50.0 * model.opt.timestep))
    files = sorted(args.accepted_dir.glob("*.npz"))
    if args.clips is not None:
        wanted = set(args.clips)
        files = [path for path in files if path.stem in wanted]
        if {path.stem for path in files} != wanted:
            raise FileNotFoundError("Requested clips are missing from accepted-dir")
    if args.limit is not None:
        files = files[: args.limit]
    rows = []
    exported = []
    for source in files:
        with np.load(source, allow_pickle=False) as clip:
            joints = np.asarray(clip["joint_pos"], dtype=np.float64)
            if args.max_frames is not None and len(joints) > args.max_frames:
                continue
            joint_vel = np.asarray(clip["joint_vel"], dtype=np.float64)
            joint_names = [str(x) for x in clip["joint_names"]]
            body_names = [str(x) for x in clip["body_names"]]
            pelvis = body_names.index("pelvis_link")
            root_pos = np.asarray(clip["body_pos_w"][:, pelvis], dtype=np.float64)
            root_quat = np.asarray(clip["body_quat_w"][:, pelvis], dtype=np.float64)
            fps = float(clip["fps"])
        clip_rows = []
        for pitch in sorted(set(args.pitches), key=lambda value: (abs(value), value)):
            qpos = corrected_qpos(model, joints, root_pos, root_quat,
                                  pitch_rad=pitch, ground_height_m=args.ground_height_m)
            result = replay_reference(source.name, joint_names, joints, joint_vel,
                                      qpos[:, :3], qpos[:, 3:7], fps, model, control,
                                      substeps=substeps)
            row = {
                "clip": source.stem,
                "pitch_rad": pitch,
                "frames": result["frames"],
                "simulated_frames": result["simulated_frames"],
                "complete": int(result["failure_frame"] is None),
                "failure_frame": result["failure_frame"],
                "joint_rms_rad": result["joint_rms_rad"],
                "root_rms_m": result["root_rms_m"],
                "peak_unclipped_torque_ratio": result["peak_unclipped_torque_ratio"],
                "mean_root_height_shift_m": float(np.mean(qpos[:, 2] - root_pos[:, 2])),
            }
            rows.append(row)
            clip_rows.append(row)
            print(f"{source.stem}: pitch {pitch:+.3f}, "
                  f"frames {result['simulated_frames']}/{result['frames']}", flush=True)
        complete = [row for row in clip_rows if row["complete"]]
        if complete and args.export_dir is not None:
            winner = min(complete, key=lambda row: (abs(row["pitch_rad"]),
                                                    row["root_rms_m"]))
            output = args.export_dir / source.name
            if output.exists():
                raise FileExistsError(f"Refusing to replace an existing trial: {output}")
            correct(source, output, pitch_rad=float(winner["pitch_rad"]),
                    ground_height_m=args.ground_height_m)
            exported.append(str(output))

    if not rows:
        raise SystemExit("No accepted clips matched the frame limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Searched {len(rows)} variants of {len({row['clip'] for row in rows})} clips; "
          f"{len(exported)} complete exports; wrote {args.output}")


if __name__ == "__main__":
    main()
