"""Shared geometry, model, and export utilities for the PyRoki pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree

import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from apps.paths import CONFIG_ROOT, WORKSPACE_ROOT

from .motion_clip import MotionClip

DEFAULT_CONFIG = CONFIG_ROOT / "gemx_pyroki_retargeter.json"
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class RetargetResult:
    """Files produced by a successful 50 Hz F26 reference export."""

    npz: Path
    csv: Path
    metadata: Path
    source_motion: Path
    frames: int
    fps: float
    position_rms_m: float


def _load_config(config_path: str | Path | None) -> tuple[dict, Path]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Retargeter config was not found: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream), path.resolve()


def _mujoco_from_source_positions(values: np.ndarray) -> np.ndarray:
    """Map GEM-X's source Y-up vector to MuJoCo's Z-up world frame."""
    array = np.asarray(values, dtype=np.float64)
    return array @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


_BVH_TO_MUJOCO = Rotation.from_euler("X", 90.0, degrees=True)


def _mujoco_from_source_rotations(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Change GEM-X source orientation basis from Y-up to MuJoCo Z-up."""
    source = Rotation.from_quat(np.asarray(quaternions_xyzw, dtype=np.float64))
    return (_BVH_TO_MUJOCO * source * _BVH_TO_MUJOCO.inv()).as_quat()


def _align_source_heading(
    positions_m: np.ndarray,
    rotations_xyzw: np.ndarray,
    heading_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Align GEM-X source axes with the target robot's forward/lateral axes."""
    heading = Rotation.from_euler("Z", heading_degrees, degrees=True)
    aligned_positions = heading.apply(np.asarray(positions_m).reshape(-1, 3)).reshape(
        positions_m.shape
    )
    aligned_rotations = (
        (heading * Rotation.from_quat(np.asarray(rotations_xyzw).reshape(-1, 4)) * heading.inv())
        .as_quat()
        .reshape(rotations_xyzw.shape)
    )
    return aligned_positions, aligned_rotations


def _savgol_window(frame_count: int, requested_window: int) -> int:
    """Return a valid odd Savitzky-Golay window, or zero when disabled."""
    window = max(1, int(requested_window))
    window = min(window, frame_count if frame_count % 2 else frame_count - 1)
    return window if window >= 3 else 0


def _smooth_rotation_sequence(rotations: Rotation, requested_window: int) -> Rotation:
    """Denoise a rotation trajectory while preserving its first frame."""
    window = _savgol_window(len(rotations), requested_window)
    if not window:
        return rotations
    anchor = rotations[0]
    relative = (anchor.inv() * rotations).as_rotvec()
    filtered = savgol_filter(
        relative, window_length=window, polyorder=min(2, window - 1), axis=0, mode="interp"
    )
    filtered -= filtered[0]
    return anchor * Rotation.from_rotvec(filtered)


def _source_pelvis_rotations(
    source_positions: np.ndarray,
    source_names: list[str],
    source_root_rotations: Rotation,
    config: dict,
) -> Rotation:
    """Build the configured, anatomically calibrated pelvis orientation series."""
    root_orientation = config.get("root_orientation", {})
    mode = str(root_orientation.get("mode", "pelvis_landmark_3d"))
    smoothing_window = int(root_orientation.get("smoothing_window_frames", 1))
    if mode == "full_3d":
        return _smooth_rotation_sequence(source_root_rotations, smoothing_window)
    if mode == "yaw_only":
        matrix = source_root_rotations.as_matrix()
        yaw = np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
        return _smooth_rotation_sequence(Rotation.from_euler("Z", yaw), smoothing_window)
    if mode == "pelvis_landmark_3d":
        labels = {
            "hips": str(root_orientation.get("hips_source", "Hips")),
            "spine": str(root_orientation.get("spine_source", "Spine1")),
            "left": str(root_orientation.get("left_hip_source", "LeftLeg")),
            "right": str(root_orientation.get("right_hip_source", "RightLeg")),
        }
        try:
            indices = {key: source_names.index(value) for key, value in labels.items()}
        except ValueError as exc:
            raise ValueError(
                "pelvis_landmark_3d requires Hips, Spine1, LeftLeg and RightLeg source joints."
            ) from exc
        frames: list[np.ndarray] = []
        for frame in range(source_positions.shape[0]):
            right = source_positions[frame, indices["right"]] - source_positions[frame, indices["left"]]
            right_norm = float(np.linalg.norm(right))
            if right_norm < 1e-6:
                raise ValueError("Cannot derive pelvis orientation: left/right hip points overlap.")
            right /= right_norm
            up = source_positions[frame, indices["spine"]] - source_positions[frame, indices["hips"]]
            up -= right * np.dot(up, right)
            up_norm = float(np.linalg.norm(up))
            if up_norm < 1e-6:
                raise ValueError(
                    "Cannot derive pelvis orientation: Hips→Spine1 is collinear with the hip line."
                )
            up /= up_norm
            forward = np.cross(up, right)
            forward /= np.linalg.norm(forward)
            frames.append(np.column_stack([right, forward, np.cross(right, forward)]))
        return _smooth_rotation_sequence(Rotation.from_matrix(np.stack(frames)), smoothing_window)
    if mode != "hip_line_yaw":
        raise ValueError(
            "root_orientation.mode must be 'pelvis_landmark_3d', 'hip_line_yaw', 'yaw_only', or 'full_3d'."
        )
    try:
        left_index = source_names.index(str(root_orientation.get("left_hip_source", "LeftLeg")))
        right_index = source_names.index(str(root_orientation.get("right_hip_source", "RightLeg")))
    except ValueError as exc:
        raise ValueError("hip_line_yaw requires configured hip source joints.") from exc
    hip_line_xy = source_positions[:, right_index, :2] - source_positions[:, left_index, :2]
    if np.any(np.linalg.norm(hip_line_xy, axis=1) < 1e-6):
        raise ValueError("Cannot derive hip-line heading: left/right hip points overlap.")
    heading = np.unwrap(np.arctan2(hip_line_xy[:, 0], -hip_line_xy[:, 1]))
    heading -= heading[0]
    window = _savgol_window(heading.size, smoothing_window)
    if window:
        heading = savgol_filter(heading, window_length=window, polyorder=min(2, window - 1), mode="interp")
        heading -= heading[0]
    return Rotation.from_euler("Z", heading)


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    identifier = mujoco.mj_name2id(model, kind, name)
    if identifier < 0:
        label = "body" if kind == mujoco.mjtObj.mjOBJ_BODY else "joint"
        raise ValueError(f"MJCF model does not contain {label} {name!r}.")
    return identifier


def _ensure_retarget_model(model_path: Path, config: dict) -> Path:
    """Create an MJCF sibling containing the configured massless target sites."""
    site_specs = config.get("retarget_sites", [])
    if not site_specs:
        return model_path
    suffix = str(config.get("retarget_model", {}).get("generated_suffix", "_retarget"))
    derived_path = model_path.with_name(f"{model_path.stem}{suffix}{model_path.suffix}")
    try:
        tree = ElementTree.parse(model_path)
    except ElementTree.ParseError as exc:
        raise ValueError(f"Target MJCF could not be parsed: {model_path}") from exc
    root = tree.getroot()
    bodies = {body.get("name"): body for body in root.findall(".//body") if body.get("name")}
    existing_sites = {site.get("name") for site in root.findall(".//site") if site.get("name")}
    for spec in site_specs:
        name = str(spec.get("name", ""))
        body_name = str(spec.get("body", ""))
        position = spec.get("pos_m")
        if not name or not body_name or not isinstance(position, list) or len(position) != 3:
            raise ValueError("Each retarget site requires name, body, and a three-value pos_m.")
        if body_name not in bodies:
            raise ValueError(f"Retarget site {name!r} refers to missing body {body_name!r}.")
        if name in existing_sites:
            continue
        ElementTree.SubElement(
            bodies[body_name],
            "site",
            {
                "name": name,
                "type": "sphere",
                "size": "0.008",
                "pos": " ".join(f"{float(value):.9g}" for value in position),
                "rgba": "0.95 0.55 0.04 0.75",
                "group": "4",
            },
        )
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = derived_path.with_suffix(f"{derived_path.suffix}.tmp")
    tree.write(temporary_path, encoding="utf-8", xml_declaration=True)
    temporary_path.replace(derived_path)
    return derived_path.resolve()


def _validate_and_index_model(
    model: mujoco.MjModel, config: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return qpos addresses and physical hinge limits in controller order."""
    joint_order = [str(name) for name in config.get("joint_order", [])]
    if len(joint_order) != 26 or len(set(joint_order)) != 26:
        raise ValueError("The PyRoki policy must define exactly 26 unique controlled joints.")
    qpos_addresses: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for name in joint_order:
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Controlled joint {name!r} must be a one-DoF hinge joint.")
        if not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"Controlled joint {name!r} must define physical MJCF limits.")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        lower.append(float(model.jnt_range[joint_id, 0]))
        upper.append(float(model.jnt_range[joint_id, 1]))
    return (
        np.asarray(qpos_addresses, dtype=np.int32),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def _contact_masks(
    source_positions: np.ndarray,
    clip: MotionClip,
    config: dict,
) -> dict[str, np.ndarray]:
    """Identify sustained low-velocity source-foot intervals for F26 contacts."""
    contact_config = config.get("contacts", {})
    if not bool(contact_config.get("enabled", False)):
        return {}
    masks: dict[str, np.ndarray] = {}
    maximum_speed = float(contact_config.get("max_speed_mps", 0.22))
    height_margin = float(contact_config.get("height_margin_m", 0.08))
    minimum_frames = max(1, int(contact_config.get("minimum_frames", 3)))
    for name in contact_config.get("feet", []):
        if name not in clip.names:
            continue
        index = clip.names.index(name)
        positions = source_positions[:, index]
        smoothing_window = _savgol_window(
            positions.shape[0], int(contact_config.get("smoothing_window_frames", 1))
        )
        if smoothing_window:
            positions = savgol_filter(
                positions,
                window_length=smoothing_window,
                polyorder=min(2, smoothing_window - 1),
                axis=0,
                mode="interp",
            )
        speed = np.linalg.norm(np.diff(positions, axis=0, prepend=positions[:1]), axis=1) * clip.fps
        mask = speed <= maximum_speed
        mask &= positions[:, 2] <= np.percentile(positions[:, 2], 20.0) + height_margin
        start = 0
        while start < mask.size:
            if not mask[start]:
                start += 1
                continue
            end = start + 1
            while end < mask.size and mask[end]:
                end += 1
            if end - start < minimum_frames:
                mask[start:end] = False
            start = end
        masks[str(name)] = mask
    return masks


def _write_outputs(
    output_dir: Path,
    model_path: Path,
    config_path: Path,
    config: dict,
    clip: MotionClip,
    qpos: np.ndarray,
    joint_qpos_addresses: np.ndarray,
    joint_order: list[str],
    mujoco_joint_order: list[str],
    position_rms_m: float,
    position_errors_by_effector: dict[str, dict[str, float]],
    contact_summary: dict[str, dict[str, int]],
    position_chain_scales: dict[str, float],
    source_motion_path: Path,
    source_fps: float | None = None,
    terminal_transition: dict[str, object] | None = None,
    initial_transition: dict[str, object] | None = None,
    initial_alignment_qpos: np.ndarray | None = None,
    *,
    left_contact: np.ndarray | None = None,
    right_contact: np.ndarray | None = None,
    support_contacts: dict[str, np.ndarray] | None = None,
    target_keypoints_m: np.ndarray | None = None,
    progress: ProgressCallback | None = None,
    output_stem: str = "gemx_50hz_retargeted",
) -> RetargetResult:
    """Write the F26-compatible NPZ, inspection CSV, and metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    npz_path, csv_path, metadata_path = (stem.with_suffix(suffix) for suffix in (".npz", ".csv", ".json"))
    joint_positions = qpos[:, joint_qpos_addresses]
    if len(mujoco_joint_order) != len(joint_order) or set(mujoco_joint_order) != set(joint_order):
        raise ValueError("MuJoCo and controller joint orders must name the same 26 joints.")
    source_frame_count = int(clip.frame_count)
    frame_count = int(qpos.shape[0])
    terminal_transition = terminal_transition or {"enabled": False}
    initial_transition = initial_transition or {"enabled": False}
    initial_frames = int(initial_transition.get("added_frames", 0))
    timeline = dict(
        original_num_frames=source_frame_count, source_start_frame=initial_frames,
        initial_num_frames=initial_frames, settle_support_end_frame=initial_frames + source_frame_count,
        settle_return_end_frame=frame_count, settle_num_frames=int(terminal_transition.get("added_frames", 0)),
    )
    headers = ["frame", "time_s", "root_x_m", "root_y_m", "root_z_m", "root_qw", "root_qx", "root_qy", "root_qz"]
    headers.extend(f"{name}_rad" for name in joint_order)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for frame, values in enumerate(joint_positions):
            writer.writerow([frame, frame / clip.fps, *qpos[frame, :7], *values])
    metadata = {
        "format": "f26_pyroki_retargeted_motion_v1",
        "robot": str(config.get("robot_name", "humanoid_light_v2")),
        "source_motion": str(source_motion_path.resolve()),
        "source_gemx_motion": str(source_motion_path.resolve()),
        "target_mjcf": str(model_path.resolve()),
        "retargeter_config": str(config_path),
        "retarget_settings": {
            "source_scale": float(config.get("source_scale", 1.0)),
            "source_heading_alignment_degrees": float(config.get("source_heading_alignment_degrees", 0.0)),
            "root_orientation": config.get("root_orientation", {}),
            "pelvis_frame_convention": "F26_forward_left_up",
            "f26_solver": config.get("f26_solver", {}),
            "retarget_model": config.get("retarget_model", {}),
            "retarget_sites": config.get("retarget_sites", []),
            "contacts": config.get("contacts", {}),
            "source_joint_aliases": config.get("source_joint_aliases", {}),
            "initial_pose": config.get("initial_pose", {"mode": "upright"}),
            "initial_alignment_qpos": (
                np.asarray(initial_alignment_qpos, dtype=np.float64).tolist()
                if initial_alignment_qpos is not None else None
            ),
            "position_chain_scale_factors": position_chain_scales,
        },
        "coordinate_frame": "MuJoCo Z-up; root quaternion is WXYZ; first-frame orientation is aligned to initial_pose.mode",
        "frames": frame_count,
        "source_motion_frames": source_frame_count,
        "source_start_frame": initial_frames,
        "fps": float(clip.fps),
        "source_fps": float(source_fps if source_fps is not None else clip.fps),
        "joint_order": joint_order,
        "mujoco_qpos_joint_order": mujoco_joint_order,
        "position_rms_m": float(position_rms_m),
        "position_errors_by_effector_m": position_errors_by_effector,
        "stance_contact_summary": contact_summary,
        "terminal_transition": terminal_transition,
        "initial_transition": initial_transition,
        "safety": "All 26 commanded hinges are constrained to MJCF limits. This is a kinematic reference, not a hardware command.",
    }
    metadata.update(timeline)
    arrays = {} if target_keypoints_m is None else {"target_keypoints_m": target_keypoints_m}
    from .simulation_export import write_simulation_archive
    physics = config.get("physical_constraints", {})
    write_simulation_archive(
        npz_path, mujoco.MjModel.from_xml_path(str(model_path)), qpos, clip.fps, metadata,
        arrays=arrays, left_contact=left_contact, right_contact=right_contact,
        support_contacts=support_contacts,
        contact_height_m=float(physics.get("contact_height_m", .005)),
        tolerance_m=float(physics.get("tolerance_m", .0005)), progress=progress,
    )
    return RetargetResult(
        npz_path, csv_path, metadata_path, source_motion_path, frame_count, clip.fps, position_rms_m
    )
