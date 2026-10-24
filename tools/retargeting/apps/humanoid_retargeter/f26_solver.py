"""Full-sequence F26-style PyRoki retargeting objective.

The F26 reference objective is extended with ground-height and pelvis
orientation terms.  The adapter supplies robot-frame keypoints and measured
collision-mesh support offsets while this module owns trajectory optimization.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
from loguru import logger


# jaxls' per-factor analysis messages are useful when developing a cost, but
# turn a GUI launch terminal into thousands of lines for a long video.
logger.disable("jaxls")


F26_TARGET_LINKS: tuple[tuple[str, str], ...] = (
    ("pelvis", "pelvis_link"),
    ("left_hip", "left_hip_pitch_link"),
    ("right_hip", "right_hip_pitch_link"),
    ("left_knee", "left_knee_link"),
    ("right_knee", "right_knee_link"),
    ("left_ankle", "left_ankle_roll_link"),
    ("right_ankle", "right_ankle_roll_link"),
    ("left_foot", "left_foot_link"),
    ("right_foot", "right_foot_link"),
    ("left_shoulder", "left_shoulder_pitch_link"),
    ("right_shoulder", "right_shoulder_pitch_link"),
    ("left_elbow", "left_elbow_link"),
    ("right_elbow", "right_elbow_link"),
    ("left_wrist", "left_wrist_link"),
    ("right_wrist", "right_wrist_link"),
    ("head", "head_link"),
)

DIRECT_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot"),
    ("right_ankle", "right_foot"),
)


@jaxls.Cost.create_factory
def _joint_velocity_limit_cost(
    values: jaxls.VarValues,
    joints_current: jaxls.Var[jnp.ndarray],
    joints_previous: jaxls.Var[jnp.ndarray],
    max_velocity_rad_s: float,
    dt_s: float,
    weight: float,
) -> jax.Array:
    velocity = (values[joints_current] - values[joints_previous]) / dt_s
    excess = jnp.maximum(jnp.abs(velocity) - max_velocity_rad_s, 0.0)
    return excess.flatten() * weight


@jaxls.Cost.create_factory
def _joint_acceleration_cost(values: jaxls.VarValues,
                             current: jaxls.Var[jnp.ndarray],
                             previous: jaxls.Var[jnp.ndarray],
                             before: jaxls.Var[jnp.ndarray], dt_s: float,
                             weight: float) -> jax.Array:
    return ((values[current] - 2 * values[previous] + values[before]) / dt_s**2).flatten() * weight


@jaxls.Cost.create_factory
def _root_vertical_velocity_cost(
    values: jaxls.VarValues,
    root_current: jaxls.SE3Var,
    root_previous: jaxls.SE3Var,
    max_velocity_m_s: float,
    dt_s: float,
    weight: float,
) -> jax.Array:
    velocity = (
        values[root_current].translation()[..., 2]
        - values[root_previous].translation()[..., 2]
    ) / dt_s
    return jnp.atleast_1d(
        jnp.maximum(jnp.abs(velocity) - max_velocity_m_s, 0.0) * weight
    ).flatten()


@jaxls.Cost.create_factory
def _root_orientation_cost(
    values: jaxls.VarValues,
    root: jaxls.SE3Var,
    target_rotation: jnp.ndarray,
    weight: float,
) -> jax.Array:
    """Keep a supine pelvis aligned instead of letting landmarks flip the base."""
    target = jaxlie.SO3.from_matrix(target_rotation)
    return (target.inverse() @ values[root].rotation()).log() * weight


@jaxls.Cost.create_factory
def _foot_contact_cost(
    values: jaxls.VarValues,
    root_current: jaxls.SE3Var,
    root_previous: jaxls.SE3Var,
    joints_current: jaxls.Var[jnp.ndarray],
    joints_previous: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_contact: jnp.ndarray,
    right_contact: jnp.ndarray,
    retarget_link_indices: jnp.ndarray,
    foot_target_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    """Keep ankle and calibrated toe landmark still while a foot is in contact."""
    current = values[root_current] @ jaxlie.SE3(
        robot.forward_kinematics(cfg=values[joints_current])
    )
    previous = values[root_previous] @ jaxlie.SE3(
        robot.forward_kinematics(cfg=values[joints_previous])
    )
    current_positions = current.translation()
    previous_positions = previous.translation()
    left_ankle, right_ankle, left_foot, right_foot = foot_target_indices
    selected = retarget_link_indices[jnp.array([left_ankle, right_ankle, left_foot, right_foot])]
    velocity = current_positions[selected] - previous_positions[selected]
    contact_weights = jnp.array(
        [left_contact[0], right_contact[0], left_contact[0], right_contact[0]]
    )
    return (velocity * contact_weights[:, None]).flatten() * weight


@jaxls.Cost.create_factory
def _foot_tilt_cost(
    values: jaxls.VarValues,
    root: jaxls.SE3Var,
    joints: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_contact: jnp.ndarray,
    right_contact: jnp.ndarray,
    retarget_link_indices: jnp.ndarray,
    ankle_target_indices: jnp.ndarray,
    sole_offsets_m: jnp.ndarray,
    foot_anchor_xy_m: jnp.ndarray,
    support_contact: jnp.ndarray,
    support_link_indices: jnp.ndarray,
    support_offsets_m: jnp.ndarray,
    tilt_weight: float,
    ground_weight: float,
    ground_height_m: float,
    anchor_weight: float,
    support_weight: float,
    support_height_m: float,
) -> jax.Array:
    """Keep stance feet upright and their sole contact points on the ground."""
    world_links = values[root] @ jaxlie.SE3(robot.forward_kinematics(cfg=values[joints]))
    rotations = world_links.rotation().as_matrix()
    left_ankle, right_ankle = ankle_target_indices
    link_indices = retarget_link_indices[jnp.array([left_ankle, right_ankle])]
    # The model's foot sole normal is the ankle-roll link local +Y direction,
    # matching the F26 retargeting URDF used as the reference implementation.
    upright = rotations[link_indices, :, 1] - jnp.array([0.0, 0.0, 1.0])
    contacts = jnp.array([left_contact[0], right_contact[0]])
    sole_height = (
        world_links.translation()[link_indices, 2]
        + jnp.sum(rotations[link_indices, 2, :] * sole_offsets_m, axis=1)
    )
    support_height = (
        world_links.translation()[support_link_indices, 2]
        + jnp.sum(rotations[support_link_indices, 2, :] * support_offsets_m, axis=1)
    )
    return jnp.concatenate((
        (upright * contacts[:, None] * tilt_weight).flatten(),
        (sole_height - ground_height_m) * contacts * ground_weight,
        ((world_links.translation()[link_indices, :2] - foot_anchor_xy_m)
         * contacts[:, None] * anchor_weight).flatten(),
        (support_height - support_height_m) * support_contact * support_weight,
    ))


def _foot_contact_anchors(
    target_keypoints: np.ndarray,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    left_ankle_index: int,
    right_ankle_index: int,
) -> np.ndarray:
    """Hold each planted ankle at the median observed XY of its contact run."""
    anchors = np.stack((
        target_keypoints[:, left_ankle_index, :2],
        target_keypoints[:, right_ankle_index, :2],
    ), axis=1).copy()
    for side, mask in enumerate((left_contact, right_contact)):
        active = np.asarray(mask).reshape(-1) >= 0.5
        start = 0
        while start < len(active):
            if not active[start]:
                start += 1
                continue
            end = start + 1
            while end < len(active) and active[end]:
                end += 1
            anchors[start:end, side] = np.median(anchors[start:end, side], axis=0)
            start = end
    return anchors


def solve_f26_full_sequence(
    robot: pk.Robot,
    target_keypoints: np.ndarray,
    root_rotation_matrices: np.ndarray,
    left_foot_contact: np.ndarray,
    right_foot_contact: np.ndarray,
    *,
    fps: float,
    weights: Mapping[str, float],
    sole_offsets_m: np.ndarray,
    ground_geometry: dict | None = None,
    support_contacts: np.ndarray | None = None,
    support_offsets_m: np.ndarray | None = None,
    support_ground_weight: float = 0.0,
    support_height_m: float = 0.008,
    root_orientation_weight: float = 0.0,
    initial_root_orientation_weight: float = 0.0,
    root_vertical_velocity_limit_m_s: float = 0.0,
    root_vertical_velocity_weight: float = 0.0,
    ground_height_m: float = 0.005,
    max_iterations: int = 800,
    reference_joint_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one globally coupled F26-style trajectory.

    ``target_keypoints`` uses ``F26_TARGET_LINKS`` ordering.  This intentionally
    has no chunk boundaries: temporal smoothness, contact constraints, and
    velocity constraints are applied continuously across the full motion.
    """
    target_keypoints = np.asarray(target_keypoints, dtype=np.float64)
    root_rotation_matrices = np.asarray(root_rotation_matrices, dtype=np.float64)
    left_foot_contact = np.asarray(left_foot_contact, dtype=np.float64)
    right_foot_contact = np.asarray(right_foot_contact, dtype=np.float64)
    frame_count, target_count, dimensions = target_keypoints.shape
    if (target_count, dimensions) != (len(F26_TARGET_LINKS), 3):
        raise ValueError(
            f"Expected keypoints [frames, {len(F26_TARGET_LINKS)}, 3], got "
            f"{target_keypoints.shape}."
        )
    if frame_count < 2:
        raise ValueError("Full-sequence PyRoki retargeting needs at least two frames.")
    if root_rotation_matrices.shape != (frame_count, 3, 3):
        raise ValueError("root_rotation_matrices must have shape [frames, 3, 3].")
    if left_foot_contact.shape != (frame_count, 1) or right_foot_contact.shape != (
        frame_count,
        1,
    ):
        raise ValueError("Foot contacts must have shape [frames, 1].")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite.")
    sole_offsets_m = np.asarray(sole_offsets_m, dtype=np.float64)
    if sole_offsets_m.shape != (2, 3):
        raise ValueError("sole_offsets_m must have shape [2, 3].")
    support_contacts = (
        np.zeros((frame_count, 5), dtype=np.float64)
        if support_contacts is None else np.asarray(support_contacts, dtype=np.float64)
    )
    support_offsets_m = (
        np.zeros((5, 3), dtype=np.float64)
        if support_offsets_m is None else np.asarray(support_offsets_m, dtype=np.float64)
    )
    if support_contacts.shape != (frame_count, 5) or support_offsets_m.shape != (5, 3):
        raise ValueError("Support contacts must be [frames, 5] and offsets [5, 3].")

    target_names = [name for name, _ in F26_TARGET_LINKS]
    target_link_names = [link for _, link in F26_TARGET_LINKS]
    missing_links = [name for name in target_link_names if name not in robot.links.names]
    if missing_links:
        raise ValueError("PyRoki robot is missing F26 retarget links: " + ", ".join(missing_links))
    link_indices = jnp.asarray([robot.links.names.index(name) for name in target_link_names])
    retarget_mask = jnp.zeros((target_count, target_count))
    for start_name, end_name in DIRECT_PAIRS:
        start, end = target_names.index(start_name), target_names.index(end_name)
        retarget_mask = retarget_mask.at[start, end].set(1.0)
        retarget_mask = retarget_mask.at[end, start].set(1.0)

    joints_that_move_less = [
        name
        for name in ("torso_roll_joint", "right_wrist_joint", "left_wrist_joint")
        if name in robot.joints.actuated_names
    ]
    reduced_motion_indices = jnp.asarray(
        [robot.joints.actuated_names.index(name) for name in joints_that_move_less], dtype=jnp.int32
    )
    left_ankle = target_names.index("left_ankle")
    right_ankle = target_names.index("right_ankle")
    left_foot = target_names.index("left_foot")
    right_foot = target_names.index("right_foot")
    ankle_indices = jnp.asarray([left_ankle, right_ankle], dtype=jnp.int32)
    foot_indices = jnp.asarray([left_ankle, right_ankle, left_foot, right_foot], dtype=jnp.int32)
    foot_anchor_xy = _foot_contact_anchors(
        target_keypoints, left_foot_contact, right_foot_contact, left_ankle, right_ankle
    )
    support_link_names = (
        "torso_roll_link", "left_wrist_link", "right_wrist_link",
        "left_knee_link", "right_knee_link",
    )
    support_link_indices = jnp.asarray(
        [robot.links.names.index(name) for name in support_link_names], dtype=jnp.int32
    )

    joint_var = robot.joint_var_cls(jnp.arange(frame_count))
    root_var = jaxls.SE3Var(jnp.arange(frame_count))
    if reference_joint_positions is not None:
        reference_joint_positions = np.asarray(reference_joint_positions, dtype=np.float64)
        if reference_joint_positions.shape != (frame_count, robot.joints.num_actuated_joints):
            raise ValueError("Reference joints must match the frame count and PyRoki joint order.")
        if not np.isfinite(reference_joint_positions).all():
            raise ValueError("Reference joints must be finite.")

    @jaxls.Cost.create_factory
    def local_bone_alignment(
        values: jaxls.VarValues,
        roots: jaxls.SE3Var,
        joints: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        world_links = values[roots] @ jaxlie.SE3(robot.forward_kinematics(cfg=values[joints]))
        robot_positions = world_links.translation()[link_indices]
        delta_target = keypoints[:, None] - keypoints[None, :]
        delta_robot = robot_positions[:, None] - robot_positions[None, :]
        scale = jnp.ones((target_count, target_count, 1))
        off_diagonal = 1.0 - jnp.eye(target_count)
        position_residual = (
            (delta_target - delta_robot * scale)
            * off_diagonal[..., None]
            * retarget_mask[..., None]
        )
        target_normalized = delta_target / jnp.linalg.norm(
            delta_target + 1e-6, axis=-1, keepdims=True
        )
        robot_normalized = delta_robot / jnp.linalg.norm(delta_robot + 1e-6, axis=-1, keepdims=True)
        angular_residual = (target_normalized - robot_normalized) * retarget_mask[..., None]
        return jnp.concatenate([position_residual.flatten(), angular_residual.flatten()]) * float(
            weights["local_alignment"]
        )

    @jaxls.Cost.create_factory
    def global_point_alignment(
        values: jaxls.VarValues,
        roots: jaxls.SE3Var,
        joints: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        world_links = values[roots] @ jaxlie.SE3(robot.forward_kinematics(cfg=values[joints]))
        return (world_links.translation()[link_indices] - keypoints).flatten() * float(
            weights["global_alignment"]
        )

    @jaxls.Cost.create_factory
    def root_smoothness(
        values: jaxls.VarValues,
        current: jaxls.SE3Var,
        previous: jaxls.SE3Var,
    ) -> jax.Array:
        return (values[current].inverse() @ values[previous]).log().flatten() * float(
            weights["root_smoothness"]
        )

    initial_roots = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(jnp.asarray(root_rotation_matrices)),
        jnp.asarray(target_keypoints[:, 0]),
    )
    rest_weights = jnp.full((robot.joints.num_actuated_joints,), 0.02)
    if reduced_motion_indices.size:
        rest_weights = rest_weights.at[reduced_motion_indices].set(
            float(weights["joint_rest_penalty"])
        )
    batched_robot = jax.tree.map(lambda value: value[None], robot)
    costs: list[jaxls.Cost] = [
        local_bone_alignment(root_var, joint_var, jnp.asarray(target_keypoints)),
        pk.costs.limit_cost(batched_robot, joint_var, 100.0),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, frame_count)),
            robot.joint_var_cls(jnp.arange(0, frame_count - 1)),
            float(weights["joint_smoothness"]),
        ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, frame_count)),
            jaxls.SE3Var(jnp.arange(0, frame_count - 1)),
        ),
        global_point_alignment(root_var, joint_var, jnp.asarray(target_keypoints)),
        pk.costs.rest_cost(joint_var, joint_var.default_factory()[None], rest_weights[None]),
        _joint_velocity_limit_cost(
            robot.joint_var_cls(jnp.arange(1, frame_count)),
            robot.joint_var_cls(jnp.arange(0, frame_count - 1)),
            float(weights.get("joint_velocity_limit_rad_s", 20.0)),
            1.0 / float(fps),
            float(weights["joint_vel_limit"]),
        ),
    ]
    if reference_joint_positions is not None:
        @jaxls.Cost.create_factory
        def reference_joint_cost(values, joints, reference):
            return (values[joints] - reference) * float(weights.get("joint_reference", 10.0))

        costs.append(reference_joint_cost(joint_var, jnp.asarray(reference_joint_positions)))
    if frame_count > 2 and weights.get("joint_acceleration", 0.0) > 0.0:
        costs.append(_joint_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, frame_count)),
            robot.joint_var_cls(jnp.arange(1, frame_count - 1)),
            robot.joint_var_cls(jnp.arange(frame_count - 2)), 1.0 / fps,
            float(weights["joint_acceleration"]),
        ))
    if root_orientation_weight > 0.0:
        costs.append(_root_orientation_cost(root_var, jnp.asarray(root_rotation_matrices),
                                            float(root_orientation_weight)))
    if initial_root_orientation_weight > 0.0:
        costs.append(_root_orientation_cost(jaxls.SE3Var(0), jnp.asarray(root_rotation_matrices[0]),
                                            float(initial_root_orientation_weight)))
    if ground_geometry is not None and weights.get("mesh_ground", 0.0) > 0.0:
        mesh_links = jnp.asarray(ground_geometry["links"])
        mesh_vertices = jnp.asarray(ground_geometry["vertices"])
        foot_meshes = jnp.asarray(ground_geometry["feet"])

        @jaxls.Cost.create_factory
        def mesh_ground_cost(values: jaxls.VarValues, roots: jaxls.SE3Var, joints: jaxls.Var[jnp.ndarray], contact: jnp.ndarray):
            world = values[roots] @ jaxlie.SE3(robot.forward_kinematics(cfg=values[joints]))
            rotations = world.rotation().as_matrix()[mesh_links]
            heights = (jnp.einsum("gi,gvi->gv", rotations[:, 2, :], mesh_vertices)
                       + world.translation()[mesh_links, 2, None])
            lowest = jnp.min(heights, axis=1)
            return jnp.concatenate((
                jnp.minimum(lowest - float(ground_height_m), 0.0) * float(weights["mesh_ground"]),
                (lowest[foot_meshes] - float(ground_height_m)) * contact * float(weights.get("mesh_support", 60.0)),
            ))

        costs.append(mesh_ground_cost(root_var, joint_var,
                                     jnp.asarray(np.column_stack((left_foot_contact[:, 0], right_foot_contact[:, 0])))))
    if root_vertical_velocity_weight > 0.0:
        costs.append(_root_vertical_velocity_cost(
            jaxls.SE3Var(jnp.arange(1, frame_count)),
            jaxls.SE3Var(jnp.arange(0, frame_count - 1)),
            float(root_vertical_velocity_limit_m_s),
            1.0 / float(fps),
            float(root_vertical_velocity_weight),
        ))
    costs.append(_foot_contact_cost(
        jaxls.SE3Var(jnp.arange(1, frame_count)), jaxls.SE3Var(jnp.arange(frame_count - 1)),
        robot.joint_var_cls(jnp.arange(1, frame_count)), robot.joint_var_cls(jnp.arange(frame_count - 1)),
        batched_robot, jnp.asarray(left_foot_contact[1:]), jnp.asarray(right_foot_contact[1:]),
        link_indices[None], foot_indices[None], float(weights["foot_contact"]),
    ))
    costs.append(_foot_tilt_cost(
        root_var, joint_var, batched_robot, jnp.asarray(left_foot_contact), jnp.asarray(right_foot_contact),
        link_indices[None], ankle_indices[None], jnp.asarray(sole_offsets_m)[None], jnp.asarray(foot_anchor_xy),
        jnp.asarray(support_contacts), support_link_indices[None], jnp.asarray(support_offsets_m)[None],
        float(weights["foot_tilt"]), float(weights.get("foot_ground", 30.0)), float(ground_height_m),
        float(weights.get("foot_anchor", 20.0)), float(support_ground_weight), float(support_height_m),
    ))

    variables = [joint_var, root_var]
    initial_joints = (joint_var if reference_joint_positions is None else
                      joint_var.with_value(jnp.asarray(reference_joint_positions)))
    initial_values = [initial_joints, root_var.with_value(initial_roots)]
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=variables).analyze()
    solution = problem.solve(
        initial_vals=jaxls.VarValues.make(
            initial_values
        ),
        termination=jaxls.TerminationConfig(max_iterations=int(max_iterations)),
        verbose=False,
    )
    return np.asarray(solution[root_var].wxyz_xyz), np.asarray(solution[joint_var])


def _blend_quaternions_wxyz(
    previous: np.ndarray, current: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Blend an overlap while preserving the unit quaternion convention."""
    current = np.asarray(current, dtype=np.float64).copy()
    previous = np.asarray(previous, dtype=np.float64)
    current[np.sum(previous * current, axis=1) < 0.0] *= -1.0
    blended = previous * (1.0 - alpha[:, None]) + current * alpha[:, None]
    return blended / np.linalg.norm(blended, axis=1, keepdims=True)


def solve_f26_trajectory(
    robot: pk.Robot,
    target_keypoints: np.ndarray,
    root_rotation_matrices: np.ndarray,
    left_foot_contact: np.ndarray,
    right_foot_contact: np.ndarray,
    *,
    fps: float,
    weights: Mapping[str, float],
    sole_offsets_m: np.ndarray,
    ground_geometry: dict | None = None,
    support_contacts: np.ndarray | None = None,
    support_offsets_m: np.ndarray | None = None,
    support_ground_weight: float = 0.0,
    support_height_m: float = 0.008,
    root_orientation_weight: float = 0.0,
    initial_root_orientation_weight: float = 0.0,
    root_vertical_velocity_limit_m_s: float = 0.0,
    root_vertical_velocity_weight: float = 0.0,
    ground_height_m: float = 0.005,
    max_iterations: int = 800,
    long_motion_max_iterations: int | None = None,
    max_full_sequence_frames: int = 450,
    window_frames: int = 320,
    overlap_frames: int = 32,
    progress: Callable[[str], None] | None = None,
    reference_joint_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the F26 objective without allocating an unbounded JAX problem.

    A short clip uses one globally coupled problem.  A long clip uses the
    same extended F26 objective on overlapping windows
    and smoothly stitches their shared frames.  This prevents a 30+ second
    50 Hz video from allocating an unbounded full-sequence factor graph.
    """
    frame_count = int(np.asarray(target_keypoints).shape[0])
    if support_contacts is None:
        support_contacts = np.zeros((frame_count, 5), dtype=np.float64)
    if frame_count <= max_full_sequence_frames:
        if progress:
            progress(f"F26 전체 시퀀스 최적화: {frame_count} 프레임")
        return solve_f26_full_sequence(
            robot,
            target_keypoints,
            root_rotation_matrices,
            left_foot_contact,
            right_foot_contact,
            fps=fps,
            weights=weights,
            sole_offsets_m=sole_offsets_m,
            ground_geometry=ground_geometry,
            support_contacts=support_contacts,
            support_offsets_m=support_offsets_m,
            support_ground_weight=support_ground_weight,
            support_height_m=support_height_m,
            root_orientation_weight=root_orientation_weight,
            initial_root_orientation_weight=initial_root_orientation_weight,
            root_vertical_velocity_limit_m_s=root_vertical_velocity_limit_m_s,
            root_vertical_velocity_weight=root_vertical_velocity_weight,
            ground_height_m=ground_height_m,
            max_iterations=max_iterations,
            reference_joint_positions=reference_joint_positions,
        )
    if window_frames < 2:
        raise ValueError("window_frames must be at least two.")
    if overlap_frames < 1 or overlap_frames >= window_frames - 1:
        raise ValueError("overlap_frames must be between 1 and window_frames - 2.")

    step_frames = window_frames - overlap_frames
    window_iterations = (
        int(long_motion_max_iterations)
        if long_motion_max_iterations is not None
        else int(max_iterations)
    )
    if window_iterations <= 0:
        raise ValueError("long_motion_max_iterations must be positive.")
    window_count = 1 + int(np.ceil((frame_count - window_frames) / step_frames))
    output_bases: np.ndarray | None = None
    output_joints: np.ndarray | None = None
    start = 0
    window_index = 0
    while True:
        end = min(start + window_frames, frame_count)
        window_index += 1
        if progress:
            progress(
                f"F26 메모리 안전 최적화: 구간 {window_index}/{window_count} "
                f"({start + 1}–{end}/{frame_count} 프레임)"
            )
        bases, joints = solve_f26_full_sequence(
            robot,
            target_keypoints[start:end],
            root_rotation_matrices[start:end],
            left_foot_contact[start:end],
            right_foot_contact[start:end],
            fps=fps,
            weights=weights,
            sole_offsets_m=sole_offsets_m,
            ground_geometry=ground_geometry,
            support_contacts=support_contacts[start:end],
            support_offsets_m=support_offsets_m,
            support_ground_weight=support_ground_weight,
            support_height_m=support_height_m,
            root_orientation_weight=root_orientation_weight,
            initial_root_orientation_weight=initial_root_orientation_weight,
            root_vertical_velocity_limit_m_s=root_vertical_velocity_limit_m_s,
            root_vertical_velocity_weight=root_vertical_velocity_weight,
            ground_height_m=ground_height_m,
            max_iterations=window_iterations,
            reference_joint_positions=(None if reference_joint_positions is None else
                                       reference_joint_positions[start:end]),
        )
        # Each window defines different factor closures. Retaining all LLVM
        # executables across windows/candidates can exhaust desktop RAM.
        # Results are already host arrays, so release compilation caches now.
        jax.clear_caches()
        import gc
        gc.collect()
        if output_bases is None or output_joints is None:
            output_bases = np.empty((frame_count, bases.shape[1]), dtype=np.float64)
            output_joints = np.empty((frame_count, joints.shape[1]), dtype=np.float64)
            output_bases[start:end] = bases
            output_joints[start:end] = joints
        else:
            shared = min(overlap_frames, end - start)
            alpha = (np.arange(shared, dtype=np.float64) + 1.0) / (shared + 1.0)
            output_bases[start : start + shared, :4] = _blend_quaternions_wxyz(
                output_bases[start : start + shared, :4], bases[:shared, :4], alpha
            )
            output_bases[start : start + shared, 4:] = (
                output_bases[start : start + shared, 4:] * (1.0 - alpha[:, None])
                + bases[:shared, 4:] * alpha[:, None]
            )
            output_joints[start : start + shared] = (
                output_joints[start : start + shared] * (1.0 - alpha[:, None])
                + joints[:shared] * alpha[:, None]
            )
            output_bases[start + shared : end] = bases[shared:]
            output_joints[start + shared : end] = joints[shared:]
        if end == frame_count:
            break
        start = end - overlap_frames

    assert output_bases is not None and output_joints is not None
    return output_bases, output_joints
