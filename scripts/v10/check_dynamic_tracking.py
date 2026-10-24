"""Replay accepted V10 references with bounded V9-style joint PD in MuJoCo.

This is an actuator/contact preflight, not a trained SuperMimic policy. It
measures whether the reference can be followed from its first frame without
pinning the floating base or applying external support forces.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def replay_reference(
    name: str, joint_names: list[str], q_ref: np.ndarray, qd_ref: np.ndarray,
    root_ref: np.ndarray, root_quat: np.ndarray, fps: float,
    model: mujoco.MjModel, control: dict, *, substeps: int,
) -> dict:
    expected = [model.joint(i).name for i in range(1, model.njnt)]
    if joint_names != expected or q_ref.shape != qd_ref.shape or fps != 50.0:
        raise ValueError(f"Reference does not match the V10 50 Hz model: {name}")
    if abs(model.opt.timestep * substeps - 1.0 / fps) > 1e-9:
        raise ValueError("MuJoCo timestep and substeps must match the reference fps")

    kp = np.array([control["joints"][name]["kp"] for name in joint_names])
    kd = np.array([control["joints"][name]["kd"] for name in joint_names])
    effort = np.array([control["joints"][name]["effort_limit_nm"] for name in joint_names])
    qadr = np.asarray(model.jnt_qposadr[1:], dtype=int)
    vadr = np.asarray(model.jnt_dofadr[1:], dtype=int)
    model.dof_armature[vadr] = np.array([control["joints"][name]["armature"] for name in joint_names])
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = root_ref[0]
    data.qpos[3:7] = root_quat[0]
    data.qpos[qadr] = q_ref[0]
    data.qvel[vadr] = qd_ref[0]
    mujoco.mj_forward(model, data)

    joint_errors = []
    root_errors = []
    torque_ratios = []
    floor_contacts = 0
    nsteps = 0
    failure_frame = None
    for frame in range(len(q_ref)):
        for _ in range(substeps):
            # V9 gains drive a position target; velocity damping is on actual speed.
            torque = kp * (q_ref[frame] - data.qpos[qadr]) - kd * data.qvel[vadr]
            torque_ratios.append(float(np.max(np.abs(torque) / effort)))
            data.qfrc_applied[vadr] = np.clip(torque, -effort, effort)
            mujoco.mj_step(model, data)
            nsteps += 1
            floor_contacts += int(any(c.geom1 == 0 or c.geom2 == 0 for c in data.contact))
        joint_errors.append(float(np.sqrt(np.mean((data.qpos[qadr] - q_ref[frame]) ** 2))))
        root_errors.append(float(np.linalg.norm(data.qpos[:3] - root_ref[frame])))
        if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.35:
            failure_frame = frame
            break

    return {
        "clip": name,
        "frames": len(q_ref),
        "simulated_frames": len(joint_errors),
        "failure_frame": failure_frame,
        "joint_rms_rad": float(np.sqrt(np.mean(np.square(joint_errors)))),
        "root_rms_m": float(np.sqrt(np.mean(np.square(root_errors)))),
        "peak_unclipped_torque_ratio": max(torque_ratios),
        "fraction_substeps_with_floor_contact": floor_contacts / nsteps,
        "final_root_height_m": float(data.qpos[2]),
    }


def replay(path: Path, model: mujoco.MjModel, control: dict, *, substeps: int) -> dict:
    with np.load(path, allow_pickle=False) as clip:
        joint_names = [str(x) for x in clip["joint_names"]]
        body_names = [str(x) for x in clip["body_names"]]
        q_ref = np.asarray(clip["joint_pos"], dtype=np.float64)
        qd_ref = np.asarray(clip["joint_vel"], dtype=np.float64)
        root_ref = np.asarray(clip["body_pos_w"][:, body_names.index("pelvis_link")], dtype=np.float64)
        root_quat = np.asarray(clip["body_quat_w"][:, body_names.index("pelvis_link")], dtype=np.float64)
        fps = float(clip["fps"])
    return replay_reference(path.name, joint_names, q_ref, qd_ref, root_ref, root_quat,
                            fps, model, control, substeps=substeps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-dir", type=Path, default=ROOT / "data/v10/accepted")
    parser.add_argument("--output", type=Path, default=ROOT / "data/v10/manifests/v10_dynamic_preflight.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
    control = json.loads((ROOT / "configs/robots/humanoid_v10/control_profile.json").read_text())
    substeps = round(1.0 / (50.0 * model.opt.timestep))
    files = sorted(args.accepted_dir.glob("*.npz"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No accepted V10 NPZ clips found")
    rows = [replay(path, model, control, substeps=substeps) for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
