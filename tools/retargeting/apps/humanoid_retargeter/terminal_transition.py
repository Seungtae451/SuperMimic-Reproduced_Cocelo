"""Robot-only zero-stand transitions around retargeted human motion."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class SafeStandTransition:
    """Metadata for a robot-only suffix that finishes at exact joint zero."""

    qpos: np.ndarray
    source_frames: int
    final_hold_frames: int
    transition_frames: int
    zero_hold_frames: int
    target_qpos: np.ndarray
    stand_diagnostics: dict[str, float | str]

    @property
    def added_frames(self) -> int:
        return self.qpos.shape[0] - self.source_frames

    def metadata(self, fps: float) -> dict:
        return {
            "enabled": True,
            "mode": "zero_stand",
            "description": (
                "Keeps the final world XY and heading, levels floating-base roll and pitch, "
                "calculates a ground-clear base height from the zero-joint collision geometry, "
                "and transitions all 26 controlled joints exactly to 0.0 radians."
            ),
            "source_frames": self.source_frames,
            "added_frames": self.added_frames,
            "final_hold_frames": self.final_hold_frames,
            "transition_frames": self.transition_frames,
            "zero_hold_frames": self.zero_hold_frames,
            "final_hold_s": self.final_hold_frames / fps,
            "transition_s": self.transition_frames / fps,
            "zero_hold_s": self.zero_hold_frames / fps,
            "target_root_pose_mode": "keep_final_xy_and_yaw_grounded_zero_stand",
            "target_root_position_m": self.target_qpos[:3].tolist(),
            "target_root_quaternion_wxyz": self.target_qpos[3:7].tolist(),
            "target_joint_max_abs_rad": float(
                np.max(np.abs(self.target_qpos[7:]), initial=0.0)
            ),
            **self.stand_diagnostics,
        }


def _frames(seconds: float, fps: float, label: str) -> int:
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError(f"terminal_transition.{label} must be a non-negative finite number.")
    return int(round(seconds * fps))


@dataclass(frozen=True)
class InitialStandTransition:
    """A grounded zero-stand prefix leading into the original first pose."""

    qpos: np.ndarray
    source_frames: int
    stand: SafeStandTransition

    @property
    def added_frames(self) -> int:
        return self.qpos.shape[0] - self.source_frames

    def metadata(self, fps: float) -> dict:
        return {
            "enabled": True,
            "mode": "zero_stand",
            "source_frames": self.source_frames,
            "added_frames": self.added_frames,
            "source_start_frame": self.added_frames,
            "zero_hold_frames": self.stand.zero_hold_frames,
            "transition_frames": self.stand.transition_frames,
            "initial_hold_frames": self.stand.final_hold_frames,
            "zero_hold_s": self.stand.zero_hold_frames / fps,
            "transition_s": self.stand.transition_frames / fps,
            "initial_hold_s": self.stand.final_hold_frames / fps,
            "start_root_pose_mode": "keep_first_xy_and_yaw_grounded_zero_stand",
            "start_root_position_m": self.stand.target_qpos[:3].tolist(),
            "start_root_quaternion_wxyz": self.stand.target_qpos[3:7].tolist(),
            **self.stand.stand_diagnostics,
            "heading_mode": "first_yaw_only_roll_pitch_level",
        }


def prepend_safe_stand_transition(
    trajectory: np.ndarray,
    model: mujoco.MjModel,
    joint_qpos_addresses: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    fps: float,
    zero_hold_s: float = 1.0,
    transition_s: float = 2.0,
    initial_hold_s: float = 0.5,
    ground_clearance_m: float = 0.005,
) -> InitialStandTransition:
    """Prepend zero stand → first pose, preserving every original motion frame.

    Time-reversing the existing terminal blend preserves its minimum-jerk
    easing, quaternion interpolation, and geometry-based stand height.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] != model.nq:
        raise ValueError(f"trajectory must have shape [frames, {model.nq}].")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory must be finite.")
    stand = append_safe_stand_transition(
        trajectory[:1], model, joint_qpos_addresses, lower, upper,
        fps=fps, final_hold_s=initial_hold_s, transition_s=transition_s,
        zero_hold_s=zero_hold_s, ground_clearance_m=ground_clearance_m,
    )
    return InitialStandTransition(
        np.vstack((stand.qpos[:0:-1], trajectory)), trajectory.shape[0], stand
    )


def _minimum_jerk(progress: np.ndarray) -> np.ndarray:
    """Quintic easing with zero velocity and acceleration at both endpoints."""
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def _root_yaw_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Preserve heading only; terminal roll/pitch cannot be part of zero stand."""
    rotation = Rotation.from_quat(np.asarray(quaternion_wxyz, dtype=np.float64)[[1, 2, 3, 0]])
    matrix = rotation.as_matrix()
    yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    yaw_only = Rotation.from_euler("z", yaw).as_quat()
    return yaw_only[[3, 0, 1, 2]]


def _mesh_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Return the world-space lowest vertex of one MuJoCo mesh geometry."""
    mesh_id = int(model.geom_dataid[geom_id])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
    if vertex_count == 0:
        raise ValueError(f"Collision mesh {geom_id} has no vertices.")
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world_vertices = vertices @ rotation.T + data.geom_xpos[geom_id]
    return float(np.min(world_vertices[:, 2]))


def _geom_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Return a conservative world-space lower bound for a collision geometry."""
    geom_type = int(model.geom_type[geom_id])
    position = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        return _mesh_lowest_z(model, data, geom_id)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(position[2] - size[0])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(position[2] - size[0] - size[1] * abs(rotation[2, 2]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radial_z = size[0] * np.hypot(rotation[2, 0], rotation[2, 1])
        return float(position[2] - radial_z - size[1] * abs(rotation[2, 2]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(position[2] - np.sum(np.abs(rotation[2]) * size))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return float(position[2] - np.linalg.norm(rotation[2] * size))
    raise ValueError(f"Unsupported robot collision geometry type {geom_type} for zero stand.")


def _grounded_zero_stand_target(
    start: np.ndarray,
    model: mujoco.MjModel,
    joint_qpos_addresses: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    ground_clearance_m: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Build exact-zero joints at final XY/yaw and solve base Z from collision meshes."""
    if not np.isfinite(ground_clearance_m) or ground_clearance_m < 0:
        raise ValueError("terminal_transition.ground_clearance_m must be finite and non-negative.")
    if np.any(0.0 < lower - 1e-12) or np.any(0.0 > upper + 1e-12):
        raise ValueError("Humanoid Light zero stand is outside a controlled-joint limit.")

    target = start.copy()
    target[:2] = start[:2]
    target[2] = 0.0
    target[3:7] = _root_yaw_quaternion_wxyz(start[3:7])
    target[joint_qpos_addresses] = 0.0

    data = mujoco.MjData(model)
    data.qpos[:] = target
    mujoco.mj_forward(model, data)
    lower_surface_z = [
        _geom_lowest_z(model, data, geom_id)
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] != 0
        and model.geom_contype[geom_id]
        and model.geom_conaffinity[geom_id]
    ]
    if not lower_surface_z:
        raise ValueError("Humanoid Light model has no robot collision geometry for zero stand.")
    lowest_z = float(min(lower_surface_z))
    target[2] = ground_clearance_m - lowest_z
    return target, {
        "ground_clearance_m": float(ground_clearance_m),
        "zero_joint_collision_lowest_z_at_root_zero_m": lowest_z,
        "computed_zero_stand_base_height_m": float(target[2]),
        "target_joint_value_rad": 0.0,
        "heading_mode": "final_yaw_only_roll_pitch_level",
    }


def grounded_zero_supine_target(
    start: np.ndarray,
    model: mujoco.MjModel,
    joint_qpos_addresses: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    ground_clearance_m: float = 0.005,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Place the zero-joint robot on its back, with its local forward axis up."""
    if not np.isfinite(ground_clearance_m) or ground_clearance_m < 0:
        raise ValueError("ground_clearance_m must be finite and non-negative.")
    if np.any(0.0 < lower - 1e-12) or np.any(0.0 > upper + 1e-12):
        raise ValueError("Humanoid Light zero pose is outside a controlled-joint limit.")
    target = np.asarray(start, dtype=np.float64).copy()
    target[2] = 0.0
    # Upright Humanoid Light faces local +X.  -90 degrees about world Y
    # points the face (+X) upward and the head (+Z) toward world -X.
    target[3:7] = Rotation.from_euler("y", -90.0, degrees=True).as_quat()[[3, 0, 1, 2]]
    target[joint_qpos_addresses] = 0.0
    data = mujoco.MjData(model)
    data.qpos[:] = target
    mujoco.mj_forward(model, data)
    collision_geoms = [
        i for i in range(model.ngeom)
        if model.geom_bodyid[i] != 0 and model.geom_contype[i] and model.geom_conaffinity[i]
    ]
    if not collision_geoms:
        raise ValueError("Humanoid Light model has no collision geometry.")
    lowest_z = min(_geom_lowest_z(model, data, i) for i in collision_geoms)
    target[2] = ground_clearance_m - lowest_z
    return target, {
        "ground_clearance_m": float(ground_clearance_m),
        "zero_joint_collision_lowest_z_at_root_zero_m": float(lowest_z),
        "computed_zero_supine_base_height_m": float(target[2]),
        "target_joint_value_rad": 0.0,
    }


@dataclass(frozen=True)
class InitialSupineTransition:
    qpos: np.ndarray
    source_frames: int
    target_qpos: np.ndarray
    zero_hold_frames: int
    transition_frames: int
    initial_hold_frames: int
    diagnostics: dict[str, float | str]

    @property
    def added_frames(self) -> int:
        return self.qpos.shape[0] - self.source_frames

    def metadata(self, fps: float) -> dict:
        return {
            "enabled": True,
            "mode": "zero_supine",
            "source_frames": self.source_frames,
            "added_frames": self.added_frames,
            "source_start_frame": self.added_frames,
            "zero_hold_frames": self.zero_hold_frames,
            "transition_frames": self.transition_frames,
            "initial_hold_frames": self.initial_hold_frames,
            "zero_hold_s": self.zero_hold_frames / fps,
            "transition_s": self.transition_frames / fps,
            "initial_hold_s": self.initial_hold_frames / fps,
            "start_root_position_m": self.target_qpos[:3].tolist(),
            "start_root_quaternion_wxyz": self.target_qpos[3:7].tolist(),
            **self.diagnostics,
        }


def prepend_zero_supine_transition(
    trajectory: np.ndarray,
    target_qpos: np.ndarray,
    *,
    fps: float,
    zero_hold_s: float = 0.5,
    transition_s: float = 1.0,
    initial_hold_s: float = 0.25,
    diagnostics: dict[str, float | str] | None = None,
) -> InitialSupineTransition:
    """Hold an exact zero-joint supine pose, then blend into source frame zero."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    target_qpos = np.asarray(target_qpos, dtype=np.float64)
    if trajectory.ndim != 2 or not len(trajectory) or target_qpos.shape != trajectory.shape[1:]:
        raise ValueError("Supine target and trajectory qpos shapes do not match.")
    hold = _frames(zero_hold_s, fps, "zero_hold_s")
    blend_frames = _frames(transition_s, fps, "transition_s")
    settled = _frames(initial_hold_s, fps, "initial_hold_s")
    if hold < 1 or blend_frames < 2:
        raise ValueError("Supine start requires at least one hold and two transition frames.")
    alpha = _minimum_jerk(np.arange(1, blend_frames + 1) / blend_frames)
    first = trajectory[0]
    transition = target_qpos[None] * (1.0 - alpha[:, None]) + first[None] * alpha[:, None]
    quats = Rotation.from_quat(np.stack((target_qpos[3:7], first[3:7]))[:, [1, 2, 3, 0]])
    transition[:, 3:7] = Slerp([0, 1], quats)(alpha).as_quat()[:, [3, 0, 1, 2]]
    prefix = np.vstack((
        np.repeat(target_qpos[None], hold, axis=0),
        transition,
        np.repeat(first[None], settled, axis=0),
    ))
    return InitialSupineTransition(
        np.vstack((prefix, trajectory)), len(trajectory), target_qpos.copy(),
        hold, blend_frames, settled, diagnostics or {},
    )


def append_safe_stand_transition(
    trajectory: np.ndarray,
    model: mujoco.MjModel,
    joint_qpos_addresses: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    fps: float,
    final_hold_s: float = 0.5,
    transition_s: float = 2.0,
    zero_hold_s: float = 1.0,
    ground_clearance_m: float = 0.005,
) -> SafeStandTransition:
    """Append a smooth final-pose → grounded exact-zero-stand suffix.

    Final world XY and yaw are retained.  Roll, pitch, and base height are
    intentionally allowed to change because keeping a tilted, low base while
    commanding every joint to zero produces an unstable-looking posture.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    joint_qpos_addresses = np.asarray(joint_qpos_addresses, dtype=np.int32)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] != model.nq:
        raise ValueError(f"trajectory must have shape [frames, {model.nq}].")
    if not np.isfinite(trajectory).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("trajectory and fps must be finite, and fps must be positive.")
    final_hold_frames = _frames(final_hold_s, fps, "final_hold_s")
    transition_frames = _frames(transition_s, fps, "transition_s")
    zero_hold_frames = _frames(zero_hold_s, fps, "zero_hold_s")
    if transition_frames < 2:
        raise ValueError("terminal_transition.transition_s must contain at least two policy frames.")

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != joint_qpos_addresses.shape or upper.shape != joint_qpos_addresses.shape:
        raise ValueError("lower and upper must match the controlled joint qpos addresses.")
    start = trajectory[-1].copy()
    target, stand_diagnostics = _grounded_zero_stand_target(
        start,
        model,
        joint_qpos_addresses,
        lower,
        upper,
        ground_clearance_m=ground_clearance_m,
    )

    progress = np.arange(1, transition_frames + 1, dtype=np.float64) / transition_frames
    blend = _minimum_jerk(progress)
    transition = np.repeat(start[None], transition_frames, axis=0)
    transition[:, :3] = start[:3] * (1.0 - blend[:, None]) + target[:3] * blend[:, None]
    transition[:, :2] = start[:2]
    transition[:, joint_qpos_addresses] = (
        start[joint_qpos_addresses] * (1.0 - blend[:, None])
        + target[joint_qpos_addresses] * blend[:, None]
    )
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_quat(np.vstack((start[3:7], target[3:7]))[:, [1, 2, 3, 0]]),
    )(blend).as_quat()
    transition[:, 3:7] = rotations[:, [3, 0, 1, 2]]

    chunks = [trajectory]
    if final_hold_frames:
        chunks.append(np.repeat(start[None], final_hold_frames, axis=0))
    chunks.append(transition)
    if zero_hold_frames:
        chunks.append(np.repeat(target[None], zero_hold_frames, axis=0))
    result = SafeStandTransition(
        np.vstack(chunks),
        trajectory.shape[0],
        final_hold_frames,
        transition_frames,
        zero_hold_frames,
        target,
        stand_diagnostics,
    )
    if not np.array_equal(result.target_qpos[joint_qpos_addresses], np.zeros_like(lower)):
        raise RuntimeError("Zero stand target must contain exact 0.0 for all controlled joints.")
    return result
