"""Common template-exact nine-key simulation export with a JSON sidecar.

Final joint/link states use MuJoCo FK; sensor fixed transforms come from USD.
The simulation_arrays helper also supports internal diagnostic calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


SCHEMA_VERSION = "humanoid_reference_v2"
SUPPORTS = (
    ("left_foot", "left_ankle_roll_link"),
    ("right_foot", "right_ankle_roll_link"),
    ("back", "torso_roll_link"),
    ("left_hand", "left_wrist_link"),
    ("right_hand", "right_wrist_link"),
    ("left_knee", "left_knee_link"),
    ("right_knee", "right_knee_link"),
)


def _gradient(values, fps):
    return (np.gradient(values, 1. / fps, axis=0) if len(values) > 1 else np.zeros_like(values))


def _continuous_quaternions(values):
    values = np.array(values, dtype=np.float64, copy=True)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Cannot export a zero quaternion.")
    values /= norms
    for frame in range(1, len(values)):
        flip = np.sum(values[frame - 1] * values[frame], axis=-1) < 0
        values[frame] *= np.where(flip, -1., 1.)[..., None]
    return values


def simulation_arrays(model, qpos, fps, *, left_contact=None, right_contact=None,
                      support_contacts=None, contact_height_m=.005,
                      tolerance_m=.0005, progress=None):
    """Return native-order joint state, every robot link state and typed contacts."""
    qpos = np.array(qpos, dtype=np.float64, copy=True)
    fps = float(fps)
    if (qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos) or
            not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Simulation export requires finite [N,nq] qpos and positive fps.")
    if model.nq != 33 or model.nv != 32 or model.njnt != 27 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Simulation export currently supports Humanoid Light v2's free base + 26 hinges.")
    qpos[:, 3:7] = _continuous_quaternions(qpos[:, 3:7])
    frames = len(qpos)
    body_ids = np.arange(1, model.nbody, dtype=np.int32)
    joint_ids = np.arange(1, model.njnt, dtype=np.int32)
    joint_qpos = model.jnt_qposadr[joint_ids].copy()
    joint_dofs = model.jnt_dofadr[joint_ids].copy()
    names = np.asarray([model.joint(int(i)).name for i in joint_ids])
    body_names = np.asarray([model.body(int(i)).name for i in body_ids])
    qvel = np.zeros((frames, model.nv))
    for frame in range(frames):
        lo, hi = max(0, frame - 1), min(frames - 1, frame + 1)
        if hi > lo:
            mujoco.mj_differentiatePos(model, qvel[frame], (hi - lo) / fps, qpos[lo], qpos[hi])
            # mj_differentiatePos expresses free-base angular increments in
            # qpos[lo]'s body frame. Transport them to this sample's frame.
            first_rotation, current_rotation = np.empty(9), np.empty(9)
            mujoco.mju_quat2Mat(first_rotation, qpos[lo, 3:7])
            mujoco.mju_quat2Mat(current_rotation, qpos[frame, 3:7])
            qvel[frame, 3:6] = current_rotation.reshape(3, 3).T @ first_rotation.reshape(3, 3) @ qvel[frame, 3:6]
    qacc = _gradient(qvel, fps)
    shape = (frames, len(body_ids), 3)
    body_pos, body_vel, body_ang_vel, body_com, body_com_vel = (np.empty(shape) for _ in range(5))
    body_quat = np.empty((frames, len(body_ids), 4))
    body_ground = np.zeros((frames, len(body_ids)))
    collision_geoms = [i for i in range(model.ngeom) if
                       "_collision" in model.geom(i).name and "_mesh_visual" not in model.geom(i).name]
    by_body = [[g for g in collision_geoms if model.geom_bodyid[g] == b] for b in body_ids]
    ground_valid = np.asarray([bool(geoms) for geoms in by_body])
    ground = model.geom("ground").id
    support_ids = np.asarray([model.body(body).id for _, body in SUPPORTS], dtype=np.int32)
    support_geom_ids = np.asarray([model.geom(body + "_collision").id for _, body in SUPPORTS], dtype=np.int32)
    contact_distance = np.empty((frames, len(SUPPORTS)))
    contact_points = np.empty((frames, len(SUPPORTS), 3))
    reference = np.zeros((frames, len(SUPPORTS)), dtype=np.float32)
    reference_available = np.zeros(len(SUPPORTS), dtype=bool)
    masks = {**dict(support_contacts or {}), "left_foot": left_contact, "right_foot": right_contact}
    for column, (name, _) in enumerate(SUPPORTS):
        if masks.get(name) is not None:
            mask = np.asarray(masks[name], dtype=np.float32).reshape(-1)
            if mask.shape != (frames,) or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
                raise ValueError(f"Invalid final support contact mask: {name}.")
            reference[:, column] = mask
            reference_available[column] = True
    offsets = [0]
    collisions = {name: [] for name in ("geom_ids", "body_ids", "pos_w", "frame_w", "distance_m", "friction", "dim")}
    data = mujoco.MjData(model)
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for frame, pose in enumerate(qpos):
        if progress and frame % 200 == 0:
            progress(f"시뮬레이션 상태 내보내기: {frame}/{frames} 프레임 · {len(body_ids)} 링크")
        data.qpos[:] = pose
        data.qvel[:] = qvel[frame]
        mujoco.mj_fwdPosition(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]
        body_com[frame] = data.xipos[body_ids]
        distances = {g: float(mujoco.mj_geomDistance(model, data, g, ground, 2., np.zeros(6)))
                     for g in collision_geoms}
        for index, (body, geoms) in enumerate(zip(body_ids, by_body)):
            mujoco.mj_jacBody(model, data, jac_pos, jac_rot, int(body))
            body_vel[frame, index] = jac_pos @ qvel[frame]
            body_ang_vel[frame, index] = jac_rot @ qvel[frame]
            mujoco.mj_jacBodyCom(model, data, jac_pos, None, int(body))
            body_com_vel[frame, index] = jac_pos @ qvel[frame]
            if geoms:
                body_ground[frame, index] = min(distances[g] for g in geoms)
        for index, geom in enumerate(support_geom_ids):
            fromto = np.zeros(6)
            contact_distance[frame, index] = mujoco.mj_geomDistance(model, data, int(geom), ground, 2., fromto)
            contact_points[frame, index] = fromto[:3]
        for contact in data.contact[:data.ncon]:
            geoms = np.asarray(contact.geom, dtype=np.int32).copy()
            collisions["geom_ids"].append(geoms)
            collisions["body_ids"].append(model.geom_bodyid[geoms].copy())
            collisions["pos_w"].append(contact.pos.copy())
            collisions["frame_w"].append(contact.frame.reshape(3, 3).copy())
            collisions["distance_m"].append(float(contact.dist))
            collisions["friction"].append(contact.friction.copy())
            collisions["dim"].append(int(contact.dim))
        offsets.append(len(collisions["dim"]))
    body_quat = _continuous_quaternions(body_quat)
    mass = model.body_mass[body_ids].copy()
    com = np.einsum("nbi,b->ni", body_com, mass) / mass.sum()
    com_vel = np.einsum("nbi,b->ni", body_com_vel, mass) / mass.sum()
    proximity = ground_valid[None] & (body_ground <= contact_height_m + tolerance_m)
    body_index = {int(body): i for i, body in enumerate(body_ids)}
    root_index = body_index[int(model.jnt_bodyid[0])]
    root_angular = body_ang_vel[:, root_index]
    joint_pos, joint_vel, joint_acc = qpos[:, joint_qpos], qvel[:, joint_dofs], qacc[:, joint_dofs]
    rows = len(collisions["dim"])
    result = dict(
        schema_version=np.asarray(SCHEMA_VERSION), robot_name=np.asarray("humanoid_light_v2"),
        fps=np.asarray(fps), dt=np.asarray(1. / fps), time_s=np.arange(frames) / fps,
        model_timestep_s=np.asarray(model.opt.timestep),
        frame_index=np.arange(frames, dtype=np.int64), num_frames=np.asarray(frames),
        num_joints=np.asarray(len(names)), num_bodies=np.asarray(len(body_ids)),
        qpos=qpos, qvel=qvel, qacc=qacc,
        qpos_names=np.asarray(["root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz", *names]),
        qvel_names=np.asarray(["root_vx_w", "root_vy_w", "root_vz_w", "root_wx_b", "root_wy_b", "root_wz_b", *names]),
        qacc_names=np.asarray(["root_ax_w", "root_ay_w", "root_az_w", "root_alpha_x_b", "root_alpha_y_b", "root_alpha_z_b", *names]),
        joint_names=names, joint_angle_names=names, mujoco_qpos_joint_names=names,
        joint_ids=joint_ids, joint_qpos_indices=joint_qpos, joint_qvel_indices=joint_dofs,
        joint_pos=joint_pos, joint_vel=joint_vel, joint_acc=joint_acc,
        dof_pos=joint_pos, dof_vel=joint_vel, dof_acc=joint_acc,
        joint_velocities_rad_s=joint_vel, joint_accelerations_rad_s2=joint_acc,
        joint_position_limits_rad=model.jnt_range[joint_ids].copy(),
        joint_position_limited=model.jnt_limited[joint_ids].astype(bool),
        base_frame_pos=qpos[:, :3], base_frame_wxyz=qpos[:, 3:7], joint_angles=joint_pos,
        root_pos=qpos[:, :3], root_quat_w=qpos[:, 3:7],
        root_position_m=qpos[:, :3], root_quaternion_wxyz=qpos[:, 3:7],
        root_lin_vel=body_vel[:, root_index], root_ang_vel=root_angular,
        root_lin_vel_w=body_vel[:, root_index], root_ang_vel_w=root_angular,
        root_lin_acc_w=_gradient(body_vel[:, root_index], fps), root_ang_acc_w=_gradient(root_angular, fps),
        body_names=body_names, body_ids=body_ids,
        body_parent_ids=model.body_parentid[body_ids].copy(),
        body_parent_indices=np.asarray([body_index.get(int(model.body_parentid[b]), -1) for b in body_ids], dtype=np.int32),
        body_pos_w=body_pos, body_quat_w=body_quat,
        body_lin_vel_w=body_vel, body_ang_vel_w=body_ang_vel,
        body_lin_acc_w=_gradient(body_vel, fps), body_ang_acc_w=_gradient(body_ang_vel, fps),
        body_pos=body_pos, body_vel=body_vel,
        body_com_pos_w=body_com, body_com_lin_vel_w=body_com_vel,
        body_mass_kg=mass, body_inertia_kg_m2=model.body_inertia[body_ids].copy(),
        body_inertial_pos_b=model.body_ipos[body_ids].copy(),
        body_inertial_quat_b=model.body_iquat[body_ids].copy(),
        com_pos_w=com, com_lin_vel_w=com_vel, com_lin_acc_w=_gradient(com_vel, fps),
        gravity_w=model.opt.gravity.copy(),
        body_ground_distance_m=body_ground, body_ground_distance_valid=ground_valid,
        body_contact=proximity, body_ground_contact=proximity,
        contact_names=np.asarray([name for name, _ in SUPPORTS]),
        contact_body_ids=support_ids, contact_body_names=np.asarray([body for _, body in SUPPORTS]),
        contact_geom_ids=support_geom_ids, contact_reference=reference >= .5,
        contact_reference_weight=reference, contact_reference_available=reference_available,
        contact_ground_distance_m=contact_distance, contact_pos_w=contact_points,
        contact_detected=contact_distance <= contact_height_m + tolerance_m,
        foot_contact=proximity[:, [body_index[int(b)] for b in support_ids[:2]]],
        left_foot_contact=reference[:, 0], right_foot_contact=reference[:, 1],
        collision_frame_offsets=np.asarray(offsets, dtype=np.int64),
        collision_count=np.diff(offsets).astype(np.int32),
        collision_geom_ids=np.asarray(collisions["geom_ids"], dtype=np.int32).reshape(rows, 2),
        collision_body_ids=np.asarray(collisions["body_ids"], dtype=np.int32).reshape(rows, 2),
        collision_pos_w=np.asarray(collisions["pos_w"], dtype=np.float64).reshape(rows, 3),
        collision_frame_w=np.asarray(collisions["frame_w"], dtype=np.float64).reshape(rows, 3, 3),
        collision_distance_m=np.asarray(collisions["distance_m"], dtype=np.float64),
        collision_friction=np.asarray(collisions["friction"], dtype=np.float64).reshape(rows, 5),
        collision_dim=np.asarray(collisions["dim"], dtype=np.int32),
        geom_ids=np.arange(model.ngeom, dtype=np.int32),
        geom_names=np.asarray([model.geom(i).name for i in range(model.ngeom)]),
        geom_body_ids=model.geom_bodyid.copy(), ground_geom_id=np.asarray(ground),
        contact_height_threshold_m=np.asarray(contact_height_m + tolerance_m),
        contact_force_available=np.asarray(False), dynamics_validated=np.asarray(False),
        reset_qpos=qpos[0], reset_qvel=qvel[0],
    )
    result["collision_normal_w"] = result["collision_frame_w"][:, 0]
    result["collision_is_ground"] = np.any(result["collision_geom_ids"] == ground, axis=1)
    result["contact_count"] = result["collision_count"]
    return result


def _schema(arrays):
    """Describe every stored key, including names, legacy fields and ragged data."""
    manifest = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        units = "1"
        if "quat" in key or key.endswith("wxyz"):
            units = "unit quaternion WXYZ"
        elif key in {"joint_pos", "joint_angles", "dof_pos", "joint_positions_rad", "controller_joint_positions_rad"} or key.endswith("_rad"):
            units = "rad"
        elif key in {"joint_vel", "dof_vel", "joint_velocities_rad_s"} or "ang_vel" in key:
            units = "rad/s"
        elif key in {"joint_acc", "dof_acc", "joint_accelerations_rad_s2"} or "ang_acc" in key:
            units = "rad/s^2"
        elif "lin_vel" in key or key == "body_vel":
            units = "m/s"
        elif "lin_acc" in key or key == "gravity_w":
            units = "m/s^2"
        elif key.endswith(("_pos_w", "_pos_b")) or "distance_m" in key or key in {"base_frame_pos", "root_pos", "root_position_m", "body_pos", "target_keypoints_m", "root_z_correction_m", "contact_height_threshold_m"}:
            units = "m"
        elif key in {"time_s", "dt", "model_timestep_s"}:
            units = "s"
        elif key == "fps":
            units = "Hz"
        elif key == "body_mass_kg":
            units = "kg"
        elif key == "body_inertia_kg_m2":
            units = "kg*m^2"
        elif key in {"qpos", "qvel", "qacc"}:
            units = {"qpos": "m, WXYZ quaternion, rad", "qvel": "m/s, base-local rad/s, rad/s", "qacc": "m/s^2, base-local rad/s^2, rad/s^2"}[key]
        entry = dict(shape=list(array.shape), dtype=str(array.dtype), units=units)
        if key.startswith("body_") and array.ndim >= 2:
            entry["body_axis"] = "body_names (robot links only; world excluded)"
        if key.startswith("collision_") and key not in {"collision_frame_offsets", "collision_count"}:
            entry["records"] = "collision_frame_offsets[t]:collision_frame_offsets[t+1]"
        if key in {"collision_body_ids", "contact_body_ids", "body_parent_ids", "geom_body_ids"}:
            entry["id_space"] = "MuJoCo global body IDs; 0=world; robot array rows mapped by body_ids/body_names"
        if key in {"collision_geom_ids", "contact_geom_ids"}:
            entry["id_space"] = "MuJoCo geom IDs mapped by geom_ids/geom_names"
        if key.startswith("contact_") and key not in {"contact_count", "contact_force_available", "contact_height_threshold_m"}:
            entry["contact_axis"] = "contact_names"
        if key in {"joint_pos", "joint_vel", "joint_acc", "dof_pos", "dof_vel", "dof_acc", "joint_angles"}:
            entry["joint_axis"] = "joint_names (template order)"
        if key in {"joint_positions_rad", "controller_joint_positions_rad"}:
            entry["joint_axis"] = "controller_joint_names"
        manifest[key] = entry
    return manifest


def _save(path, arrays, metadata):
    """Save the exact simulation contract; all descriptions remain in JSON."""
    from .named_reference import KEYS
    if tuple(arrays) != KEYS:
        raise ValueError("Final reference NPZ must contain exactly the nine template keys.")
    path = Path(path)
    metadata = dict(metadata)
    metadata["npz_schema"] = _schema(arrays)
    metadata["schema_version"] = "test_data_named_reference_v1"
    # These fields described the superseded rich archive, not the current NPZ.
    metadata.pop("policy_npz_contract", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def write_simulation_archive(path, model, qpos, fps, metadata, *, arrays=None,
                             left_contact=None, right_contact=None, support_contacts=None,
                             contact_height_m=.005, tolerance_m=.0005, progress=None,
                             template=None, robot=None):
    """Every workflow writes the same nine-key NPZ and a separate JSON."""
    from .named_reference import arrays_from_qpos, default_robot
    from apps.paths import CONFIG_ROOT
    if progress:
        progress("시뮬레이션 상태 저장: MuJoCo FK + USD 카메라·IMU, test_data 9개 키")
    robot = default_robot(model) if robot is None else robot
    states = arrays_from_qpos(model, qpos, fps, template=template, robot=robot)
    metadata = dict(metadata)
    auxiliary = dict(metadata.get("diagnostics", {}))
    # Preview targets/support labels are development information, not fields
    # of test_data.npz. Keep them in the separate JSON only.
    for key, value in (arrays or {}).items():
        if key in {"target_keypoints_m", "root_z_correction_m"}:
            auxiliary[key] = np.asarray(value).tolist()
        elif key in {"source_start_frame", "original_num_frames", "entry_num_frames", "initial_num_frames", "settle_num_frames", "settle_support_end_frame", "settle_return_end_frame"}:
            metadata[key] = np.asarray(value).item()
    for name, mask in {**dict(support_contacts or {}), "left_foot": left_contact, "right_foot": right_contact}.items():
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != (len(qpos),) or not np.isfinite(mask).all():
                raise ValueError(f"Invalid final support contact mask: {name}.")
            auxiliary[name + "_contact"] = mask.tolist()
    metadata["diagnostics"] = auxiliary
    metadata["simulation_export"] = dict(
        schema_version="test_data_named_reference_v1", keys=list(states),
        joint_order="joint_names indexes joint_pos/joint_vel", body_order="body_names indexes all body states",
        mujoco_version=mujoco.__version__, coordinate_frame="Z-up world, meters, radians; quaternion WXYZ; link origins",
        velocity_method="mj_differentiatePos, central differences and one-sided endpoints; world body Jacobian @ qvel",
        sensor_fixed_frames=json.loads((CONFIG_ROOT / "reference_fixed_frames.json").read_text()) if template is None else
                            {name: frame.as_dict() for name, frame in robot.fixed_frames.items()},
        frames=len(qpos), joints=len(states["joint_names"]), bodies=len(states["body_names"]),
        metadata_storage="separate JSON; no extra NPZ keys",
    )
    initial = dict(metadata.get("initial_pose", {}))
    initial.update(reset_qpos=np.asarray(qpos)[0].tolist(), reset_qvel=robot.differentiate(np.asarray(qpos), fps)[0].tolist())
    metadata["initial_pose"] = initial
    return _save(path, states, metadata)


def update_archive_metadata(path, metadata):
    """Update only the JSON sidecar; never add metadata keys to the NPZ."""
    from .named_reference import KEYS
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != KEYS:
            raise ValueError("Expected a nine-key named reference archive.")
        schema = _schema({key: archive[key] for key in archive.files})
    metadata = {**metadata, "npz_schema": schema, "schema_version": "test_data_named_reference_v1"}
    metadata.pop("policy_npz_contract", None)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Re-export retargeted poses in the exact test_data.npz nine-key format.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a different --output path to preserve the original export.")
    from .viewer import load_reference_motion, _read_metadata
    reference = load_reference_motion(args.input, args.model)
    model = mujoco.MjModel.from_xml_path(str(reference.model_path))
    metadata = _read_metadata(args.input)
    metadata["target_mjcf"] = str(reference.model_path)
    with np.load(args.input, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata["simulation_export_source"] = dict(path=str(args.input.resolve()),
        sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), optimized_trajectory_preserved=True,
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_simulation_archive(args.output, model, reference.qpos, reference.fps, metadata,
        arrays=arrays, left_contact=arrays.get("left_foot_contact"), right_contact=arrays.get("right_foot_contact"))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
