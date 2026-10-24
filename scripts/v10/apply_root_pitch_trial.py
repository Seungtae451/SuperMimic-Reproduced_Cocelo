"""Create a complete V10 reference with a constant world-frame root-pitch trial.

This is a diagnostic balance correction for one motion, not an archive-wide
retargeting rule. It recomputes every body pose and velocity after changing the
root orientation and grounding the feet, leaving the source NPZ untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
FOOT_GEOMS = ("left_ankle_roll_link_collision", "right_ankle_roll_link_collision")


def _foot_meshes(model: mujoco.MjModel) -> list[tuple[int, np.ndarray]]:
    meshes = []
    for name in FOOT_GEOMS:
        geom = model.geom(name).id
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        meshes.append((geom, model.mesh_vert[start : start + count].copy()))
    return meshes


def _lowest_foot_z(data: mujoco.MjData, meshes: list[tuple[int, np.ndarray]]) -> float:
    return min(float(np.min(vertices @ data.geom_xmat[geom].reshape(3, 3).T
                            + data.geom_xpos[geom], axis=0)[2])
               for geom, vertices in meshes)


def corrected_qpos(
    model: mujoco.MjModel, joints: np.ndarray, root_pos: np.ndarray,
    root_quat: np.ndarray, *, pitch_rad: float, ground_height_m: float,
) -> np.ndarray:
    """Apply world-Y pitch and ground each frame on the actual V10 foot meshes."""
    delta = Rotation.from_euler("y", pitch_rad)
    original = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]])
    turned = (delta * original).as_quat()
    qpos = np.tile(model.qpos0, (len(joints), 1))
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = turned[:, [3, 0, 1, 2]]
    qpos[:, 7:] = joints
    meshes = _foot_meshes(model)
    data = mujoco.MjData(model)
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        qpos[frame, 2] += ground_height_m - _lowest_foot_z(data, meshes)
    return qpos


def correct(source: Path, output: Path, *, pitch_rad: float, ground_height_m: float) -> dict:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
    with np.load(source, allow_pickle=False) as clip:
        joint_names = [str(x) for x in clip["joint_names"]]
        body_names = [str(x) for x in clip["body_names"]]
        joints = np.asarray(clip["joint_pos"], dtype=np.float64)
        joint_vel = np.asarray(clip["joint_vel"], dtype=np.float64)
        pelvis = body_names.index("pelvis_link")
        root_pos = np.asarray(clip["body_pos_w"][:, pelvis], dtype=np.float64).copy()
        root_quat = np.asarray(clip["body_quat_w"][:, pelvis], dtype=np.float64).copy()
        fps = float(clip["fps"])
    if joint_names != [model.joint(i).name for i in range(1, model.njnt)]:
        raise ValueError("Source joint order differs from the V10 model")
    if body_names != [model.body(i).name for i in range(1, model.nbody)]:
        raise ValueError("Source body order differs from the V10 model")
    if joints.shape != joint_vel.shape or not np.isclose(fps, 50.0):
        raise ValueError("Expected a complete 50 Hz V10 named reference")
    if not np.isfinite(pitch_rad) or not np.isfinite(ground_height_m):
        raise ValueError("Pitch and ground height must be finite")

    qpos = corrected_qpos(model, joints, root_pos, root_quat,
                          pitch_rad=pitch_rad, ground_height_m=ground_height_m)
    meshes = _foot_meshes(model)
    data = mujoco.MjData(model)

    # MuJoCo's free-joint difference handles quaternion angular velocity.
    qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
    dt = 1.0 / fps
    for frame in range(len(qpos)):
        if frame == 0:
            mujoco.mj_differentiatePos(model, qvel[frame], dt, qpos[0], qpos[1])
        elif frame == len(qpos) - 1:
            mujoco.mj_differentiatePos(model, qvel[frame], dt, qpos[-2], qpos[-1])
        else:
            mujoco.mj_differentiatePos(model, qvel[frame], 2.0 * dt,
                                       qpos[frame - 1], qpos[frame + 1])
    qvel[:, 6:] = joint_vel

    nbody = model.nbody - 1
    body_pos = np.empty((len(qpos), nbody, 3), dtype=np.float32)
    body_quat = np.empty((len(qpos), nbody, 4), dtype=np.float32)
    body_lin = np.empty((len(qpos), nbody, 3), dtype=np.float32)
    body_ang = np.empty((len(qpos), nbody, 3), dtype=np.float32)
    velocity = np.empty(6)
    minimum_foot_z = float("inf")
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        minimum_foot_z = min(minimum_foot_z, _lowest_foot_z(data, meshes))
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]
        for body in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body,
                                     velocity, 0)
            angular = velocity[:3]
            # mj_objectVelocity is centered at the COM; the named reference
            # uses velocity at the body's link-frame origin.
            linear = velocity[3:] - np.cross(angular, data.xipos[body] - data.xpos[body])
            body_lin[frame, body - 1] = linear
            body_ang[frame, body - 1] = angular

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fps=np.asarray(fps),
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
        joint_pos=joints.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
    )
    details = {
        "source_npz": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pitch_rad": pitch_rad,
        "ground_height_m": ground_height_m,
        "minimum_foot_z_m": minimum_foot_z,
        "frames": len(qpos),
        "fps": fps,
        "method": "world Y-axis root pitch, per-frame foot-mesh grounding, MuJoCo FK and velocities",
    }
    output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pitch-rad", type=float, required=True)
    parser.add_argument("--ground-height-m", type=float, default=0.005)
    args = parser.parse_args()
    print(json.dumps(correct(args.source, args.output, pitch_rad=args.pitch_rad,
                             ground_height_m=args.ground_height_m), indent=2))


if __name__ == "__main__":
    main()
