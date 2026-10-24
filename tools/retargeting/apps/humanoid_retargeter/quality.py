"""Independent geometric certification of the final, sampled reference."""
from __future__ import annotations

import mujoco
import numpy as np

from .physical_constraints import _collision_pairs


def evaluate_reference(model, qpos, fps, left_contact, right_contact,
                       *, contact_height_m=0.005, tolerance_m=0.0005,
                       max_stance_tilt_degrees=None):
    qpos = np.asarray(qpos, dtype=np.float64)
    masks = np.column_stack((left_contact, right_contact)) >= 0.5
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError("Invalid final reference qpos.")
    if masks.shape != (len(qpos), 2) or fps <= 0:
        raise ValueError("Contact masks/rate must match the final reference.")
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    geoms = [i for i in range(model.ngeom)
             if "_collision" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
             and "_mesh_visual" not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                            f"{side}_ankle_roll_link_collision") for side in ("left", "right")]
    if ground < 0 or min(feet) < 0:
        raise ValueError("Missing ground/foot geometry.")
    pairs = _collision_pairs(model)
    data, points = mujoco.MjData(model), np.zeros(6)
    minimum_ground, minimum_self = np.inf, np.inf
    ground_frames = self_frames = support_error_frames = 0
    maximum_support_error = 0.0
    lowest_foot = []
    foot_bodies = [int(model.geom_bodyid[foot]) for foot in feet]
    stance_tilts = []
    tilt_error_frames = 0
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        distances = {g: float(mujoco.mj_geomDistance(model, data, g, ground, 2.0, points)) for g in geoms}
        floor = min(distances.values())
        # Bounding spheres only cull distant pairs; convex distances check all
        # remaining pairs, regardless of MuJoCo's collision enable masks.
        self_distance = min((float(mujoco.mj_geomDistance(model, data, a, b, 2.0, points))
                             for a, b in pairs
                             if np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b])
                             < model.geom_rbound[a] + model.geom_rbound[b] + 0.01), default=1.0)
        foot_distances = np.asarray([distances[g] for g in feet])
        error = max((max(0.0, foot_distances[side] - contact_height_m)
                     for side in range(2) if masks[frame, side]), default=0.0)
        minimum_ground = min(minimum_ground, floor)
        minimum_self = min(minimum_self, self_distance)
        maximum_support_error = max(maximum_support_error, error)
        ground_frames += floor < -tolerance_m
        self_frames += self_distance < -tolerance_m
        support_error_frames += error > tolerance_m
        lowest_foot.append(float(min(foot_distances)))
        tilts = [float(np.rad2deg(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 1], -1, 1))))
                 for side, body in enumerate(foot_bodies) if masks[frame, side]]
        stance_tilts.extend(tilts)
        tilt_error_frames += (max_stance_tilt_degrees is not None and
                              max(tilts, default=0.0) > max_stance_tilt_degrees + .01)
    velocity = np.diff(qpos[:, 7:], axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    return {
        "frames_checked": len(qpos), "fps": float(fps),
        "ground_penetration_frames": int(ground_frames),
        "minimum_ground_distance_m": float(minimum_ground),
        "self_collision_frames": int(self_frames),
        "minimum_self_distance_m": float(minimum_self),
        "checked_self_collision_pairs": len(pairs),
        "self_collision_scope": "non-adjacent collision meshes excluding overlaps present in manufactured zero pose",
        "support_height_error_frames": int(support_error_frames),
        "maximum_support_height_error_m": float(maximum_support_error),
        "unsupported_mask_frames": int(np.count_nonzero(~masks.any(axis=1))),
        "maximum_lowest_foot_height_m": float(max(lowest_foot)),
        "stance_foot_tilt_mean_deg": float(np.mean(stance_tilts)) if stance_tilts else 0.0,
        "stance_foot_tilt_p95_deg": float(np.percentile(stance_tilts, 95)) if stance_tilts else 0.0,
        "stance_foot_tilt_max_deg": float(max(stance_tilts, default=0.0)),
        "foot_tilt_error_frames": int(tilt_error_frames),
        "joint_velocity_max_rad_s": float(np.max(np.abs(velocity), initial=0.0)),
        "joint_acceleration_max_rad_s2": float(np.max(np.abs(acceleration), initial=0.0)),
        "quaternion_norm_max_error": float(np.max(abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1))),
        "geometric_pass": not (ground_frames or self_frames or support_error_frames or tilt_error_frames),
        "dynamic_balance_verified": False,
    }


def require_valid_reference(quality):
    if not quality["geometric_pass"]:
        raise RuntimeError(f"Final reference failed geometric certification: {quality}")
