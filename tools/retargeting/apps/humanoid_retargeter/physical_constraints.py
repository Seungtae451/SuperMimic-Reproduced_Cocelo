"""Project PyRoki poses onto the robot's actual ground and collision geometry.

The kinematic URDF used by PyRoki deliberately has no collision meshes.  This
last stage uses the authoritative MJCF convex meshes, including after temporal
resampling, where interpolated poses can introduce new penetrations.
"""

from __future__ import annotations

from collections.abc import Mapping, Callable
from itertools import combinations

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d


def refine_supported_trajectory(model, trajectory, joint_qpos_addresses, lower, upper,
                                left_contact, right_contact, *, fps, passes=2,
                                fixed_pose_mask=None, **kwargs):
    """Alternate temporal smoothing and exact projection without losing support.

    Only accept a lower peak acceleration with small changes to the already
    fitted reference. Every accepted trial has passed the same mesh projection.
    """
    original = np.asarray(trajectory, dtype=np.float64)
    best = original.copy()
    indices = np.r_[2, joint_qpos_addresses]
    zero_holds = np.all(original[:, joint_qpos_addresses] == 0.0, axis=1)
    if fixed_pose_mask is not None:
        zero_holds |= np.asarray(fixed_pose_mask, dtype=bool)

    def peak(q):
        return float(np.max(abs(np.diff(q[:, joint_qpos_addresses], n=2, axis=0)), initial=0.0) * fps**2)

    before = peak(best)
    accepted = 0
    failure = None
    for _ in range(passes):
        proposal = best.copy()
        filtered = gaussian_filter1d(best[:, indices], sigma=1.2, axis=0, mode="nearest")
        radius = np.r_[.012, np.full(len(joint_qpos_addresses), .06)]
        proposal[:, indices] = np.clip(filtered, original[:, indices] - radius, original[:, indices] + radius)
        proposal[:, joint_qpos_addresses] = np.clip(proposal[:, joint_qpos_addresses], lower, upper)
        proposal[zero_holds] = original[zero_holds]
        proposal[[0, -1]] = original[[0, -1]]
        try:
            candidate, _ = project_physical_trajectory(
                model, proposal, joint_qpos_addresses, lower, upper,
                left_contact, right_contact, **kwargs)
        except RuntimeError as exc:
            failure = str(exc)
            break
        drift = np.max(abs(candidate[:, joint_qpos_addresses] - original[:, joint_qpos_addresses]))
        velocity = np.max(abs(np.diff(candidate[:, joint_qpos_addresses], axis=0)), initial=0.0) * fps
        previous_velocity = np.max(abs(np.diff(best[:, joint_qpos_addresses], axis=0)), initial=0.0) * fps
        if (peak(candidate) < peak(best) and drift < .1 and velocity <= previous_velocity + .05
                and np.array_equal(candidate[zero_holds, 7:], original[zero_holds, 7:])):
            best = candidate
            accepted += 1
        else:
            break
    return best, {"requested_passes": passes, "accepted_passes": accepted,
                  "peak_joint_acceleration_before_rad_s2": before,
                  "peak_joint_acceleration_after_rad_s2": peak(best),
                  "maximum_joint_change_rad": float(np.max(abs(best[:, joint_qpos_addresses] - original[:, joint_qpos_addresses]))),
                  "rejected_projection_error": failure}


def _collision_pairs(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """Keep non-neighbouring collision meshes, excluding neutral mesh overlaps."""
    geoms = [
        i for i in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
        and "_collision" in name and "_mesh_visual" not in name
    ]
    neutral = mujoco.MjData(model)
    mujoco.mj_forward(model, neutral)
    fromto = np.zeros(6)
    pairs = []
    for a, b in combinations(geoms, 2):
        body_a, body_b = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        if body_a == body_b:
            continue
        # The manufactured meshes of directly joined links overlap at their
        # bearings.  Those overlaps are present even in the reference pose.
        if model.body_parentid[body_a] == body_b or model.body_parentid[body_b] == body_a:
            continue
        if mujoco.mj_geomDistance(model, neutral, a, b, 1.0, fromto) < -1e-4:
            continue
        pairs.append((a, b))
    return pairs


def project_physical_trajectory(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    joint_qpos_addresses: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    *,
    support_contacts: Mapping[str, np.ndarray] | None = None,
    contact_height_m: float = 0.005,
    support_height_m: float = 0.015,
    collision_margin_m: float = 0.001,
    tolerance_m: float = 0.0005,
    progress: Callable[[str], None] | None = None,
    max_stance_tilt_degrees: float | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Constrain active foot/body supports and non-neighbouring self collision.

    Root orientation and horizontal travel remain fixed.  The root height and
    joint angles are adjusted as little as possible.  An infeasible frame is
    reported rather than silently exporting an invalid reference.
    """
    output = np.asarray(trajectory, dtype=np.float64).copy()
    joint_qpos_addresses = np.asarray(joint_qpos_addresses, dtype=int)
    lower, upper = np.asarray(lower), np.asarray(upper)
    n = len(output)
    left_contact = np.asarray(left_contact).reshape(-1)
    right_contact = np.asarray(right_contact).reshape(-1)
    if len(left_contact) != n or len(right_contact) != n:
        raise ValueError("Contact masks must match the output trajectory length.")
    if contact_height_m <= 0 or support_height_m <= 0 or collision_margin_m < 0 or tolerance_m <= 0:
        raise ValueError("Physical constraint distances must be positive.")
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    feet = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_ankle_roll_link_collision")
        for side in ("left", "right")
    )
    if ground < 0 or any(geom < 0 for geom in feet):
        raise ValueError("MJCF is missing the ground or a foot collision mesh.")
    if max_stance_tilt_degrees is not None and not 0 < max_stance_tilt_degrees < 90:
        raise ValueError("max_stance_tilt_degrees must be between 0 and 90.")
    tilt_cosine = (np.cos(np.deg2rad(max_stance_tilt_degrees))
                   if max_stance_tilt_degrees is not None else -1.0)
    foot_bodies = [int(model.geom_bodyid[foot]) for foot in feet]
    support_geom_names = {
        "back": "torso_roll_link_collision",
        "left_hand": "left_wrist_link_collision",
        "right_hand": "right_wrist_link_collision",
        "left_knee": "left_knee_link_collision",
        "right_knee": "right_knee_link_collision",
    }
    support_geoms: dict[str, tuple[int, np.ndarray]] = {}
    for name, mask in (support_contacts or {}).items():
        if name not in support_geom_names:
            raise ValueError(f"Unknown support contact {name!r}.")
        values = np.asarray(mask).reshape(-1)
        if len(values) != n:
            raise ValueError(f"Support mask {name!r} must match trajectory length.")
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, support_geom_names[name])
        if geom < 0:
            raise ValueError(f"MJCF is missing support geometry {support_geom_names[name]!r}.")
        support_geoms[name] = geom, values
    pairs = _collision_pairs(model)
    ground_geoms = [
        i for i in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
        and "_collision" in name and "_mesh_visual" not in name
    ]
    data = mujoco.MjData(model)
    fromto = np.zeros(6)
    q_indices = np.r_[2, joint_qpos_addresses]
    dof_indices = np.r_[2, [model.jnt_dofadr[model.jnt_qposadr == q][0] for q in joint_qpos_addresses]]
    bounds = [(None, None), *zip(lower.tolist(), upper.tolist())]
    corrected = 0
    maximum_penetration = 0.0
    maximum_contact_error = 0.0

    def put(x: np.ndarray, pose: np.ndarray) -> None:
        data.qpos[:] = pose
        data.qpos[q_indices] = x
        mujoco.mj_forward(model, data)

    def distance(a: int, b: int) -> float:
        return float(mujoco.mj_geomDistance(model, data, a, b, 2.0, fromto))

    def near_pairs() -> list[tuple[int, int]]:
        active = []
        for a, b in pairs:
            # A conservative bounding-sphere check avoids expensive convex
            # mesh queries for body parts that are far apart.
            radius = model.geom_rbound[a] + model.geom_rbound[b] + 0.12
            if np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b]) < radius:
                active.append((a, b))
        return active

    def near_ground_geoms() -> list[int]:
        return [i for i in ground_geoms
                if data.geom_xpos[i, 2] - model.geom_rbound[i] < 0.12]

    for frame, pose in enumerate(output):
        if progress is not None and frame % 200 == 0:
            progress(f"MuJoCo 메시 접지·충돌 보정: {frame}/{n} 프레임")
        original = pose[q_indices].copy()
        put(original, pose)
        contacts = (left_contact[frame] >= 0.5, right_contact[frame] >= 0.5)
        active_support = [(geom, support_height_m) for geom, mask in support_geoms.values()
                          if mask[frame] >= 0.5]
        active_upper = [(foot, contact_height_m) for foot, contact in zip(feet, contacts) if contact]
        active_upper.extend(active_support)
        active = near_pairs()
        stance_bodies = [body for body, contact in zip(foot_bodies, contacts) if contact]

        def tilt_valid():
            return (max_stance_tilt_degrees is None or
                    all(data.xmat[body].reshape(3, 3)[2, 1] >= tilt_cosine - 1e-7
                        for body in stance_bodies))
        ground_distances = [distance(geom, ground) for geom in ground_geoms]
        upper_distances = [distance(geom, ground) for geom, _ in active_upper]
        pair_distances = [distance(a, b) for a, b in active]
        if (min(ground_distances) >= collision_margin_m - tolerance_m
            and all(d <= height + tolerance_m for d, (_, height) in zip(upper_distances, active_upper))
            and min(pair_distances, default=1.0) >= collision_margin_m - tolerance_m
            and tilt_valid()):
            maximum_penetration = max(
                maximum_penetration,
                max(0.0, collision_margin_m - min(ground_distances),
                    collision_margin_m - min(pair_distances, default=1.0)),
            )
            maximum_contact_error = max(
                maximum_contact_error,
                max((max(0.0, d - height) for d, (_, height) in zip(upper_distances, active_upper)), default=0.0),
            )
            continue

        # A shared vertical translation exactly fixes ordinary floating feet
        # without changing joint angles, stance XY, or any self distances.
        if min(pair_distances, default=1.0) >= collision_margin_m - tolerance_m and tilt_valid():
            stance_distances = np.asarray(upper_distances)
            shift = (min(height - d for d, (_, height) in zip(upper_distances, active_upper))
                     if len(stance_distances)
                     else collision_margin_m - min(ground_distances))
            shifted_ground = np.asarray(ground_distances) + shift
            shifted_stance = stance_distances + shift
            if (np.min(shifted_ground) >= collision_margin_m - tolerance_m
                and all(d <= height + tolerance_m for d, (_, height) in zip(shifted_stance, active_upper))):
                pose[2] += shift
                corrected += 1
                maximum_penetration = max(
                    maximum_penetration, max(0.0, collision_margin_m - np.min(shifted_ground))
                )
                maximum_contact_error = max(
                    maximum_contact_error,
                    max((max(0.0, d - height) for d, (_, height) in zip(shifted_stance, active_upper)), default=0.0),
                )
                continue

        # Only pairs near this pose enter the local optimization.  Recheck all
        # pairs afterwards because moving a limb can activate a new pair.
        for _ in range(3):
            active = near_pairs()
            constraints = [(geom, ground, collision_margin_m, None) for geom in near_ground_geoms()]
            constraints += [(geom, ground, collision_margin_m, height)
                            for geom, height in active_upper]
            constraints += [(a, b, collision_margin_m, None) for a, b in active]
            cached_x: np.ndarray | None = None
            cached_values: np.ndarray | None = None
            cached_jac: np.ndarray | None = None

            def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                nonlocal cached_x, cached_values, cached_jac
                if cached_x is not None and np.array_equal(x, cached_x):
                    return cached_values, cached_jac  # type: ignore[return-value]
                put(x, pose)
                values = np.empty(len(constraints))
                jac = np.empty((len(constraints), len(x)))
                for i, (a, b, _, _) in enumerate(constraints):
                    values[i] = distance(a, b)
                    delta = fromto[3:] - fromto[:3]
                    normal = delta / max(abs(values[i]), 1e-8)
                    if values[i] < 0:
                        normal = -normal
                    jac_a = np.zeros((3, model.nv))
                    jac_b = np.zeros((3, model.nv))
                    mujoco.mj_jac(model, data, jac_a, None, fromto[:3], int(model.geom_bodyid[a]))
                    mujoco.mj_jac(model, data, jac_b, None, fromto[3:], int(model.geom_bodyid[b]))
                    jac[i] = normal @ (jac_b[:, dof_indices] - jac_a[:, dof_indices])
                cached_x, cached_values, cached_jac = x.copy(), values, jac
                return values, jac

            low = np.asarray([c[2] for c in constraints])
            high_indices = [i for i, c in enumerate(constraints) if c[3] is not None]
            high = np.asarray([constraints[i][3] for i in high_indices])
            scale = np.r_[0.08, np.full(len(original) - 1, 0.35)]

            def evaluate_tilt(x):
                put(x, pose)
                values, jacobian = [], []
                for body in stance_bodies:
                    normal = data.xmat[body].reshape(3, 3)[:, 1]
                    rotation_jac = np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, data, None, rotation_jac, body)
                    values.append(normal[2] - tilt_cosine)
                    jacobian.append(np.array([normal[1], -normal[0], 0.0]) @ rotation_jac[:, dof_indices])
                return np.asarray(values), np.asarray(jacobian).reshape(-1, len(x))

            tilt_constraints = ([{"type": "ineq", "fun": lambda x: evaluate_tilt(x)[0],
                                  "jac": lambda x: evaluate_tilt(x)[1]}]
                                if max_stance_tilt_degrees is not None and stance_bodies else [])

            result = minimize(
                lambda x: 0.5 * np.sum(np.square((x - original) / scale)),
                pose[q_indices].copy(),
                jac=lambda x: (x - original) / np.square(scale),
                bounds=bounds,
                constraints=[
                    {"type": "ineq", "fun": lambda x: evaluate(x)[0] - low,
                     "jac": lambda x: evaluate(x)[1]},
                    {"type": "ineq", "fun": lambda x: high - evaluate(x)[0][high_indices],
                     "jac": lambda x: -evaluate(x)[1][high_indices]},
                ] + tilt_constraints,
                method="SLSQP",
                options={"maxiter": 45, "ftol": 1e-8},
            )
            pose[q_indices] = result.x
            put(result.x, pose)
            active = near_pairs()
            ground_distances = [distance(geom, ground) for geom in ground_geoms]
            upper_distances = [distance(geom, ground) for geom, _ in active_upper]
            pair_distances = [distance(a, b) for a, b in active]
            penetration = max(0.0, collision_margin_m - min(ground_distances), collision_margin_m - min(pair_distances, default=1.0))
            contact_error = max((max(0.0, d - height) for d, (_, height) in zip(upper_distances, active_upper)), default=0.0)
            if penetration <= tolerance_m and contact_error <= tolerance_m and tilt_valid():
                break
        else:
            worst_contact = max(
                ((d - height, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
                 for d, (geom, height) in zip(upper_distances, active_upper)),
                default=(0.0, "none"),
            )
            worst_collision = min(
                ((d, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, a),
                  mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b))
                 for d, (a, b) in zip(pair_distances, active)),
                default=(1.0, "none", "none"),
            )
            raise RuntimeError(
                f"Physical projection failed at frame {frame}: penetration {penetration:.4f} m, "
                f"contact height error {contact_error:.4f} m; "
                f"worst contact {worst_contact[1]} (+{worst_contact[0]:.4f} m), "
                f"closest self pair {worst_collision[1]}/{worst_collision[2]} "
                f"({worst_collision[0]:.4f} m); optimizer: {result.message}."
            )
        corrected += 1
        maximum_penetration = max(maximum_penetration, penetration)
        maximum_contact_error = max(maximum_contact_error, contact_error)

    return output, {
        "corrected_frames": corrected,
        "checked_self_collision_pairs": len(pairs),
        "maximum_remaining_penetration_m": maximum_penetration,
        "maximum_remaining_contact_height_error_m": maximum_contact_error,
        "maximum_root_height_correction_m": float(np.max(abs(output[:, 2] - np.asarray(trajectory)[:, 2]))),
        "maximum_joint_correction_rad": float(np.max(abs(output[:, joint_qpos_addresses] - np.asarray(trajectory)[:, joint_qpos_addresses]))),
    }
