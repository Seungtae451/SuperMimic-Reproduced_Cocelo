"""PyRoki-based retargeting for Humanoid Light v2.

GEM-X and Kimodo supply an articulated human-motion source in memory. This
module uses PyRoki/JAX for the *robot* inverse-kinematics optimization. The
MJCF remains authoritative for output and playback; a minimal kinematic URDF
is generated only for PyRoki.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
import warnings
from xml.etree import ElementTree

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .motion_clip import MotionClip
from .motion_io import save_visualization_motion
from .shared import (
    WORKSPACE_ROOT,
    RetargetResult,
    _align_source_heading,
    _contact_masks,
    _ensure_retarget_model,
    _id,
    _load_config,
    _mujoco_from_source_positions,
    _mujoco_from_source_rotations,
    _source_pelvis_rotations,
    _validate_and_index_model,
    _write_outputs,
)


ProgressCallback = Callable[[str], None]
PYROKI_GIT_REVISION = "388e43e1fc0d0ee382968d3dd72970fd62a0450c"

# The canonical 16-point F26 source interface.  GEM-X uses every canonical
# name; Kimodo's compact SOMA-30 uses ToeBase aliases resolved below.
_F26_SOURCE_JOINTS: tuple[tuple[str, str], ...] = (
    ("pelvis", "Hips"),
    ("left_hip", "LeftLeg"),
    ("right_hip", "RightLeg"),
    ("left_knee", "LeftShin"),
    ("right_knee", "RightShin"),
    ("left_ankle", "LeftFoot"),
    ("right_ankle", "RightFoot"),
    ("left_foot", "LeftToeEnd"),
    ("right_foot", "RightToeEnd"),
    ("left_shoulder", "LeftArm"),
    ("right_shoulder", "RightArm"),
    ("left_elbow", "LeftForeArm"),
    ("right_elbow", "RightForeArm"),
    ("left_wrist", "LeftHand"),
    ("right_wrist", "RightHand"),
    ("head", "Head"),
)
_F26_FOOT_MARKER_OFFSETS_M = {
    "left_foot_link": np.array([0.000749999890104, -0.0390000008047, 0.0959992855787]),
    "right_foot_link": np.array([-0.000749999890104, -0.0390000008047, 0.0959992855787]),
}


def _require_pyroki():
    """Import the optional runtime lazily so GUI startup stays lightweight."""
    try:
        import jax
        import jax.numpy as jnp
        import jaxlie
        import jaxls
        import pyroki as pk
        import yourdfpy
    except ImportError as exc:
        raise RuntimeError(
            "PyRoki가 설치되지 않았습니다. `python -m pip install -r apps/requirements-gui.txt`를 "
            "gemx_gui 환경에서 실행한 뒤 다시 시도하세요."
        ) from exc
    return jax, jnp, jaxlie, jaxls, pk, yourdfpy


def _rpy_from_mjcf_quaternion(value: str | None) -> str:
    quaternion = np.fromstring(value or "1 0 0 0", sep=" ", dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Invalid MJCF quaternion: {value!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rpy = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_euler("xyz")
    return " ".join(f"{component:.12g}" for component in rpy)


def _export_pyroki_urdf(mjcf_path: Path, output_dir: Path, base_link: str) -> Path:
    """Export the MJCF body's kinematic tree as a minimal PyRoki-compatible URDF.

    The robot has only one hinge per movable body and all hinge anchors are at
    the body origin, so this is an exact kinematic conversion for the
    Humanoid Light v2 model.  Meshes, inertias, actuators, and the MJCF
    floating joint are deliberately omitted: PyRoki optimizes a fixed-base
    tree and a separate SE(3) variable represents the floating base.
    """
    tree = ElementTree.parse(mjcf_path)
    root = tree.getroot()
    base_body = next(
        (body for body in root.findall(".//worldbody/body") if body.get("name") == base_link), None
    )
    if base_body is None:
        raise ValueError(
            f"MJCF model does not contain the requested PyRoki base body {base_link!r}."
        )

    urdf_root = ElementTree.Element("robot", {"name": f"{mjcf_path.stem}_pyroki"})
    seen_links: set[str] = set()

    def add_link(name: str) -> None:
        if name in seen_links:
            raise ValueError(f"Duplicate MJCF body name cannot be represented in URDF: {name!r}.")
        seen_links.add(name)
        ElementTree.SubElement(urdf_root, "link", {"name": name})

    def add_child(parent_name: str, body: ElementTree.Element) -> None:
        child_name = body.get("name")
        if not child_name:
            raise ValueError("Every body in the Humanoid Light v2 kinematic tree must have a name.")
        add_link(child_name)
        joints = body.findall("./joint")
        if len(joints) > 1:
            raise ValueError(
                f"Body {child_name!r} has multiple joints; PyRoki URDF export needs one."
            )
        joint = joints[0] if joints else None
        if joint is not None and joint.get("type", "hinge") not in {"hinge", "slide"}:
            raise ValueError(f"Unsupported MJCF joint type on {joint.get('name')!r}.")
        joint_name = joint.get("name") if joint is not None else f"{parent_name}_to_{child_name}"
        joint_type = (
            "revolute" if joint is not None and joint.get("type", "hinge") == "hinge" else "fixed"
        )
        urdf_joint = ElementTree.SubElement(
            urdf_root, "joint", {"name": str(joint_name), "type": joint_type}
        )
        ElementTree.SubElement(urdf_joint, "parent", {"link": parent_name})
        ElementTree.SubElement(urdf_joint, "child", {"link": child_name})
        ElementTree.SubElement(
            urdf_joint,
            "origin",
            {
                "xyz": body.get("pos", "0 0 0"),
                "rpy": _rpy_from_mjcf_quaternion(body.get("quat")),
            },
        )
        if joint is not None:
            ElementTree.SubElement(urdf_joint, "axis", {"xyz": joint.get("axis", "0 0 1")})
            lower, upper = joint.get("range", "-3.141592653589793 3.141592653589793").split()
            ElementTree.SubElement(
                urdf_joint,
                "limit",
                {"lower": lower, "upper": upper, "effort": "1000", "velocity": "100"},
            )
        for child in body.findall("./body"):
            add_child(child_name, child)

    add_link(base_link)
    for child_body in base_body.findall("./body"):
        add_child(base_link, child_body)

    # Match the fixed toe landmarks supplied by the proven F26 solver.  The
    # landmarks are kinematic-only (not new actuated joints) and locate the
    # forward face of each foot, which gives the contact objective a stable
    # point in addition to the ankle origin.
    for marker_name, parent_name in (
        ("left_foot_link", "left_ankle_roll_link"),
        ("right_foot_link", "right_ankle_roll_link"),
    ):
        if parent_name not in seen_links:
            raise ValueError(f"Cannot attach F26 foot marker: missing {parent_name!r}.")
        add_link(marker_name)
        marker_joint = ElementTree.SubElement(
            urdf_root, "joint", {"name": f"{marker_name}_joint", "type": "fixed"}
        )
        ElementTree.SubElement(marker_joint, "parent", {"link": parent_name})
        ElementTree.SubElement(marker_joint, "child", {"link": marker_name})
        ElementTree.SubElement(
            marker_joint,
            "origin",
            {
                "xyz": " ".join(
                    f"{value:.12g}" for value in _F26_FOOT_MARKER_OFFSETS_M[marker_name]
                ),
                "rpy": "0 0 0",
            },
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = output_dir / f"{mjcf_path.stem}_pyroki.urdf"
    ElementTree.ElementTree(urdf_root).write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path


def _resample_clip(clip: MotionClip, target_fps: float) -> MotionClip:
    """Resample a source motion with linear positions and quaternion SLERP."""
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("output_fps must be a positive finite value.")
    if np.isclose(clip.fps, target_fps):
        return clip
    if clip.frame_count == 1:
        return MotionClip(
            clip.names,
            clip.parents,
            clip.offsets_m,
            clip.channels,
            clip.local_positions_m.copy(),
            clip.local_rotations.copy(),
            target_fps,
        )
    source_times = np.arange(clip.frame_count, dtype=np.float64) / clip.fps
    frame_count = int(np.floor(source_times[-1] * target_fps + 1e-9)) + 1
    target_times = np.arange(frame_count, dtype=np.float64) / target_fps
    positions = np.empty((frame_count, len(clip.names), 3), dtype=np.float64)
    for joint in range(len(clip.names)):
        for axis in range(3):
            positions[:, joint, axis] = np.interp(
                target_times, source_times, clip.local_positions_m[:, joint, axis]
            )
    rotations = np.empty((frame_count, len(clip.names), 4), dtype=np.float64)
    for joint in range(len(clip.names)):
        rotations[:, joint] = Slerp(
            source_times, Rotation.from_quat(clip.local_rotations[:, joint])
        )(target_times).as_quat()
    return MotionClip(
        clip.names, clip.parents, clip.offsets_m, clip.channels, positions, rotations, target_fps
    )


def _resample_qpos_trajectory(qpos: np.ndarray, source_fps: float, target_fps: float,
                              target_frame_count: int | None = None) -> np.ndarray:
    """Resample a floating-base MuJoCo trajectory without interpolating quaternions linearly."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 7:
        raise ValueError("qpos must have shape [frames, nq] with a free-joint quaternion.")
    if np.isclose(source_fps, target_fps):
        return qpos.copy()
    source_times = np.arange(qpos.shape[0], dtype=np.float64) / source_fps
    target_count = target_frame_count or (int(np.floor(source_times[-1] * target_fps + 1e-9)) + 1)
    target_times = np.arange(target_count, dtype=np.float64) / target_fps
    result = np.empty((target_count, qpos.shape[1]), dtype=np.float64)
    for column in [*range(3), *range(7, qpos.shape[1])]:
        result[:, column] = np.interp(target_times, source_times, qpos[:, column])
    result[:, 3:7] = Slerp(source_times, Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(
        np.minimum(target_times, source_times[-1])
    ).as_quat()[:, [3, 0, 1, 2]]
    return result


def _prepare_clip(
    source_motion: MotionClip, config: dict, max_frames: int | None, output_fps: float
) -> tuple[MotionClip, np.ndarray, np.ndarray]:
    clip = source_motion
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")
        clip = MotionClip(
            clip.names,
            clip.parents,
            clip.offsets_m,
            clip.channels,
            clip.local_positions_m[:max_frames],
            clip.local_rotations[:max_frames],
            clip.fps,
        )
    if clip.frame_count == 0:
        raise ValueError("Source motion contains no frames.")
    clip = _resample_clip(clip, output_fps)
    positions, rotations = clip.global_transforms()
    positions = _mujoco_from_source_positions(positions)
    rotations = _mujoco_from_source_rotations(rotations.reshape(-1, 4)).reshape(
        clip.frame_count, -1, 4
    )
    return (
        clip,
        *_align_source_heading(
            positions, rotations, float(config.get("source_heading_alignment_degrees", 0.0))
        ),
    )


def _smooth_contact_mask(mask: np.ndarray, window_frames: int) -> np.ndarray:
    """Return the F26-style local average used to soften contact transitions."""
    window_frames = max(1, int(window_frames))
    if window_frames == 1:
        return mask.astype(np.float64)[:, None]
    radius = window_frames // 2
    values = np.empty(mask.shape[0], dtype=np.float64)
    for frame in range(mask.shape[0]):
        values[frame] = np.mean(mask[max(0, frame - radius) : min(mask.size, frame + radius + 1)])
    return values[:, None]


def _resolve_source_landmarks(clip: MotionClip, config: dict) -> dict[str, str]:
    """Resolve the F26 landmark names against a source skeleton.

    GEM-X already provides every F26 landmark under its canonical SOMA name.
    Kimodo's compact SOMA-30 skeleton intentionally omits the toe-end joints,
    so callers can supply a small alias map (``LeftToeEnd`` →
    ``LeftToeBase``) without creating a fake BVH/SOMA conversion step.
    """
    aliases = config.get("source_joint_aliases", {})
    if not isinstance(aliases, dict):
        raise ValueError("source_joint_aliases must be an object mapping F26 names to source names.")
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for _, canonical_name in _F26_SOURCE_JOINTS:
        source_name = str(aliases.get(canonical_name, canonical_name))
        if source_name not in clip.names:
            missing.append(
                canonical_name
                if source_name == canonical_name
                else f"{canonical_name} (alias: {source_name})"
            )
        else:
            resolved[canonical_name] = source_name
    if missing:
        raise ValueError("Source motion is missing F26 landmarks: " + ", ".join(missing))
    return resolved


def _contact_summary(mask: np.ndarray) -> dict[str, int]:
    segments = 0
    frame = 0
    while frame < mask.size:
        if not mask[frame]:
            frame += 1
            continue
        segments += 1
        while frame < mask.size and mask[frame]:
            frame += 1
    return {"contact_frames": int(mask.sum()), "contact_segments": segments}


def _supine_support_masks(
    clip: MotionClip,
    positions: np.ndarray,
    landmarks: dict[str, str],
    settings: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Detect back, hand and knee supports using a shared observed floor."""
    hips = positions[:, clip.names.index(landmarks["Hips"])]
    spine_name = str(settings.get("spine_source", "Spine1"))
    if spine_name not in clip.names:
        raise ValueError("Supine support detection needs the source Spine1 landmark.")
    spine = positions[:, clip.names.index(spine_name)]
    torso = spine - hips
    upright = np.abs(torso[:, 2]) / np.maximum(np.linalg.norm(torso, axis=1), 1e-8)
    names = {
        "left_hand": landmarks["LeftHand"],
        "right_hand": landmarks["RightHand"],
        "left_knee": landmarks["LeftShin"],
        "right_knee": landmarks["RightShin"],
    }
    candidate_indices = [clip.names.index(name) for name in names.values()]
    candidate_indices += [clip.names.index(landmarks[name]) for name in
                          ("LeftFoot", "RightFoot", "Hips")]
    floor = float(np.percentile(np.min(positions[:, candidate_indices, 2], axis=1), 5))
    masks: dict[str, np.ndarray] = {}
    masks["back"] = (upright < float(settings.get("back_max_upright", 0.20))) & (
        hips[:, 2] < floor + float(settings.get("back_hip_height_margin_m", 0.20))
    )
    maximum_speed = float(settings.get("max_support_speed_mps", 0.40))
    minimum_frames = max(1, int(settings.get("minimum_support_frames", 3)))
    for label, source_name in names.items():
        xyz = positions[:, clip.names.index(source_name)]
        velocity = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[:1]), axis=1) * clip.fps
        margin = float(settings.get("hand_height_margin_m", 0.01)) if "hand" in label else float(
            settings.get("knee_height_margin_m", 0.14)
        )
        candidate = (xyz[:, 2] <= floor + margin) & (velocity <= maximum_speed)
        if "knee" in label:
            candidate &= upright >= float(settings.get("knee_min_upright", 0.15))
            candidate &= upright < float(settings.get("knee_max_upright", 0.90))
        start = 0
        while start < len(candidate):
            end = start + 1
            while end < len(candidate) and candidate[end] == candidate[start]:
                end += 1
            if candidate[start] and end - start < minimum_frames:
                candidate[start:end] = False
            start = end
        masks[label] = candidate
    return upright, masks


def _sole_offsets_from_mjcf(model: mujoco.MjModel) -> np.ndarray:
    """Measure each foot mesh's neutral ground point in its ankle link frame."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ground = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    offsets = []
    for side in ("left", "right"):
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_ankle_roll_link_collision")
        fromto = np.zeros(6)
        mujoco.mj_geomDistance(model, data, geom, ground, 2.0, fromto)
        offsets.append(data.xmat[body].reshape(3, 3).T @ (fromto[:3] - data.xpos[body]))
    return np.asarray(offsets)


def _ground_geometry_from_mjcf(model: mujoco.MjModel, robot) -> dict:
    """Body-local support vertices for the JAX cost; exact meshes certify export."""
    directions = np.asarray([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1)
                             for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vertices, links, names = [], [], []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if "_collision" not in name or "_mesh_visual" in name:
            continue
        body = int(model.geom_bodyid[geom])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_name not in robot.links.names or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        points = model.mesh_vert[start:start + count]
        extreme = points[np.argmax(points @ directions.T, axis=0)]
        world = extreme @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        vertices.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
        links.append(robot.links.names.index(body_name))
        names.append(name)
    return {"vertices": np.asarray(vertices), "links": np.asarray(links),
            "feet": np.asarray([names.index(f"{side}_ankle_roll_link_collision")
                                for side in ("left", "right")])}


_SUPPORT_LINKS = (
    ("back", "torso_roll_link", "torso_roll_link_collision"),
    ("left_hand", "left_wrist_link", "left_wrist_link_collision"),
    ("right_hand", "right_wrist_link", "right_wrist_link_collision"),
    ("left_knee", "left_knee_link", "left_knee_link_collision"),
    ("right_knee", "right_knee_link", "right_knee_link_collision"),
)


def _support_offsets_from_mjcf(
    model: mujoco.MjModel, supine_qpos: np.ndarray
) -> np.ndarray:
    """Measure back/hand and knee ground-facing mesh points in link frames."""
    data = mujoco.MjData(model)
    ground = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    offsets = []
    for label, body_name, geom_name in _SUPPORT_LINKS:
        # The knee's support face is sampled in the upright leg configuration;
        # the back and wrist faces are sampled in the supine start posture.
        data.qpos[:] = model.qpos0 if "knee" in label else supine_qpos
        mujoco.mj_forward(model, data)
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        fromto = np.zeros(6)
        mujoco.mj_geomDistance(model, data, geom, ground, 2.0, fromto)
        offsets.append(data.xmat[body].reshape(3, 3).T @ (fromto[:3] - data.xpos[body]))
    return np.asarray(offsets)


def _supine_world_alignment(
    source_positions: np.ndarray,
    source_names: list[str],
    source_pelvis_rotations: Rotation,
    desired_supine_rotation: Rotation,
    spine_name: str,
) -> tuple[Rotation, Rotation] | None:
    """Align a get-up's first lying pose and final standing pose in one world frame.

    A fixed first-frame alignment alone maps the *change* from lying to
    standing around the wrong world axis.  Use the standing pelvis as the
    neutral robot orientation and choose a world yaw that makes the initial
    source rotation agree with the robot's grounded supine direction.  The
    world yaw also applies to every position target.
    """
    hips = source_positions[:, source_names.index("Hips")]
    spine = source_positions[:, source_names.index(spine_name)]
    torso = spine - hips
    upright = np.abs(torso[:, 2]) / np.maximum(np.linalg.norm(torso, axis=1), 1e-8)
    tail_start = max(1, int(0.75 * len(upright)))
    standing = np.flatnonzero((np.arange(len(upright)) >= tail_start) & (upright >= 0.85))
    if upright[0] >= 0.45 or len(standing) < 5:
        return None
    standing_rotation = source_pelvis_rotations[standing].mean()
    initial_delta = source_pelvis_rotations[0] * standing_rotation.inv()
    source_axis = initial_delta.as_rotvec()[:2]
    desired_axis = desired_supine_rotation.as_rotvec()[:2]
    if np.linalg.norm(source_axis) < 0.2 or np.linalg.norm(desired_axis) < 0.2:
        return None
    yaw = np.arctan2(desired_axis[1], desired_axis[0]) - np.arctan2(
        source_axis[1], source_axis[0]
    )
    world_alignment = Rotation.from_euler("z", yaw)
    standing_aligned = world_alignment * standing_rotation
    root_rotations = world_alignment * source_pelvis_rotations * standing_aligned.inv()
    return world_alignment, root_rotations


def _f26_target_trajectory(
    clip: MotionClip,
    source_positions: np.ndarray,
    source_rotations: np.ndarray,
    model: mujoco.MjModel,
    config: dict,
    initial_root_qpos: np.ndarray | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, int]],
    dict[str, float], dict[str, np.ndarray]
]:
    """Adapt source joints to the exact 16-point F26 full-sequence objective."""
    from .f26_solver import F26_TARGET_LINKS

    target_names = tuple(name for name, _ in F26_TARGET_LINKS)
    source_names = tuple(name for name, _ in _F26_SOURCE_JOINTS)
    if target_names != source_names:
        raise RuntimeError("GEM-X/F26 target ordering drifted; update the direct adapter.")
    landmark_names = _resolve_source_landmarks(clip, config)

    hips_index = clip.names.index(landmark_names["Hips"])
    source_indices = [
        clip.names.index(landmark_names[source_name]) for _, source_name in _F26_SOURCE_JOINTS
    ]
    root_qpos = model.qpos0 if initial_root_qpos is None else initial_root_qpos
    root_default = root_qpos[:3].copy()
    default_root_rotation = Rotation.from_quat(root_qpos[3:7][[1, 2, 3, 0]])
    source_root_rotations = Rotation.from_quat(source_rotations[:, hips_index])
    source_pelvis_rotations = _source_pelvis_rotations(
        source_positions, clip.names, source_root_rotations, config
    )
    root_orientation_mode = str(
        config.get("root_orientation", {}).get("mode", "pelvis_landmark_3d")
    )
    source_rotation_series = (
        source_root_rotations if root_orientation_mode == "full_3d" else source_pelvis_rotations
    )
    source_start_rotation = source_rotation_series[0]
    # Landmark frames use columns [anatomical right, forward, up]. F26's
    # pelvis uses [forward, left, up]. Align these anatomical conventions
    # explicitly; treating them as identical introduces a 90-degree error.
    if root_orientation_mode == "pelvis_landmark_3d":
        source_rotation_series = source_rotation_series * Rotation.from_euler("z", 90.0, degrees=True)
        source_start_rotation = source_rotation_series[0]
    initial_alignment = default_root_rotation * source_start_rotation.inv()
    calibrated = None
    if config.get("initial_pose", {}).get("mode") == "zero_supine":
        calibrated = _supine_world_alignment(
            source_positions, clip.names, source_rotation_series, default_root_rotation,
            str(config.get("supine_support", {}).get("spine_source", "Spine1")),
        )
    if calibrated is None:
        root_rotations = initial_alignment * source_rotation_series
        world_alignment = initial_alignment
    else:
        world_alignment, root_rotations = calibrated
    source_scale = float(config.get("source_scale", 1.0))
    root_displacement = source_positions[:, hips_index] - source_positions[0, hips_index]
    root_positions = root_default + source_scale * world_alignment.apply(root_displacement)

    # A constant initial-frame alignment brings GEM-X's human skeleton into
    # the Humanoid Light world axes.  The global F26 cost can then fit its own
    # per-bone scales instead of receiving old semantic-proxy targets.
    source_offsets = (
        source_positions[:, source_indices] - source_positions[:, hips_index : hips_index + 1]
    )
    keypoints = root_positions[:, None, :] + source_scale * world_alignment.apply(
        source_offsets.reshape(-1, 3)
    ).reshape(source_offsets.shape)

    segment_scales = {}
    if config.get("f26_solver", {}).get("anatomical_targets", False) and calibrated is None:
        # Transfer directions onto robot-length segments. A uniform human scale
        # otherwise asks the optimizer to shorten the legs by pitching the base.
        neutral = mujoco.MjData(model)
        mujoco.mj_forward(model, neutral)
        nominal = []
        for _, link in F26_TARGET_LINKS:
            parent = link.replace("_foot_link", "_ankle_roll_link") if link in _F26_FOOT_MARKER_OFFSETS_M else link
            body = _id(model, mujoco.mjtObj.mjOBJ_BODY, parent)
            point = neutral.xpos[body].copy()
            if link in _F26_FOOT_MARKER_OFFSETS_M:
                point += neutral.xmat[body].reshape(3, 3) @ _F26_FOOT_MARKER_OFFSETS_M[link]
            nominal.append(point)
        nominal = np.asarray(nominal)
        original = keypoints.copy()
        # Ordered parent indices in the canonical 16-point interface.
        parents = [-1, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 9, 10, 11, 12, 0]
        for child, parent in enumerate(parents):
            if parent < 0:
                continue
            delta = original[:, child] - original[:, parent]
            length = np.linalg.norm(nominal[child] - nominal[parent])
            segment_scales[target_names[child]] = float(np.median(
                source_scale * length / np.maximum(np.linalg.norm(delta, axis=1), 1e-8)))
            keypoints[:, child] = keypoints[:, parent] + delta * (
                length / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
            )
        # Remove monocular root-height drift using the lowest target ankle and
        # the measured neutral sole offset; retain vertical leg articulation.
        sole = _sole_offsets_from_mjcf(model)
        sole_z = np.einsum("fij,sj->fsi", root_rotations.as_matrix(), sole)[:, :, 2]
        floor = np.min(keypoints[:, [5, 6], 2] + sole_z, axis=1)
        keypoints[:, :, 2] += (float(config.get("physical_constraints", {}).get("contact_height_m", 0.005)) - floor)[:, None]

    contact_config = dict(config.get("contacts", {}))
    contact_config["feet"] = [
        landmark_names["LeftFoot"],
        landmark_names["LeftToeEnd"],
        landmark_names["RightFoot"],
        landmark_names["RightToeEnd"],
    ]
    contact_masks = _contact_masks(source_positions, clip, {**config, "contacts": contact_config})
    left_raw = np.logical_or(
        contact_masks.get(landmark_names["LeftFoot"], np.zeros(clip.frame_count, dtype=bool)),
        contact_masks.get(landmark_names["LeftToeEnd"], np.zeros(clip.frame_count, dtype=bool)),
    )
    right_raw = np.logical_or(
        contact_masks.get(landmark_names["RightFoot"], np.zeros(clip.frame_count, dtype=bool)),
        contact_masks.get(landmark_names["RightToeEnd"], np.zeros(clip.frame_count, dtype=bool)),
    )
    support_masks: dict[str, np.ndarray] = {}
    upright: np.ndarray | None = None
    if config.get("initial_pose", {}).get("mode") == "zero_supine":
        upright, support_masks = _supine_support_masks(
            clip, source_positions, landmark_names, dict(config.get("supine_support", {}))
        )
        # Supine feet may be close to the floor without bearing weight.  The
        # source's torso must rise before a foot becomes a support candidate.
        threshold = float(config.get("supine_support", {}).get("foot_min_upright", 0.40))
        left_raw &= upright >= threshold
        right_raw &= upright >= threshold
    if bool(contact_config.get("enabled", False)):
        # Noisy video landmarks can fail the velocity/minimum-duration test.
        # Recover a support foot only while the lower source foot is close to
        # its observed floor; frames with both feet airborne remain airborne.
        left_height = np.minimum(
            source_positions[:, clip.names.index(landmark_names["LeftFoot"]), 2],
            source_positions[:, clip.names.index(landmark_names["LeftToeEnd"]), 2],
        )
        right_height = np.minimum(
            source_positions[:, clip.names.index(landmark_names["RightFoot"]), 2],
            source_positions[:, clip.names.index(landmark_names["RightToeEnd"]), 2],
        )
        floor_height = np.percentile(np.minimum(left_height, right_height), 5)
        near_floor = np.minimum(left_height, right_height) <= (
            floor_height + float(contact_config.get("fallback_height_margin_m", 0.04))
        )
        missing_support = near_floor & ~(left_raw | right_raw)
        if upright is not None:
            missing_support &= upright >= float(
                config.get("supine_support", {}).get("fallback_foot_min_upright", 0.65)
            )
        bilateral_tolerance = float(contact_config.get("fallback_bilateral_tolerance_m", 0.015))
        left_raw |= missing_support & (left_height <= right_height + bilateral_tolerance)
        right_raw |= missing_support & (right_height <= left_height + bilateral_tolerance)
    smoothing_frames = int(config.get("f26_solver", {}).get("contact_smoothing_frames", 5))
    if contact_config.get("mode") == "supported_dance" and upright is None:
        # Explicit supported-dance mode: this clip has no flight phase. Use
        # relative foot height, which is insensitive to monocular floor drift.
        left_height = source_positions[:, clip.names.index(landmark_names["LeftFoot"]), 2]
        right_height = source_positions[:, clip.names.index(landmark_names["RightFoot"]), 2]
        from scipy.ndimage import median_filter
        difference = median_filter(left_height - right_height, size=max(1, int(clip.fps * 0.12) | 1))
        band = float(contact_config.get("bilateral_height_tolerance_m", 0.025))
        left_raw = difference <= band
        right_raw = difference >= -band
    contacts = {
        "left_foot": _smooth_contact_mask(left_raw, smoothing_frames),
        "right_foot": _smooth_contact_mask(right_raw, smoothing_frames),
    }
    summary = {"left_foot": _contact_summary(left_raw), "right_foot": _contact_summary(right_raw)}
    summary.update({name: _contact_summary(mask) for name, mask in support_masks.items()})
    scales = {name: source_scale for name in target_names}
    scales.update(segment_scales)
    return (
        keypoints,
        root_rotations.as_matrix(),
        contacts["left_foot"],
        contacts["right_foot"],
        summary,
        scales,
        support_masks,
    )


def _f26_position_errors(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    target_keypoints: np.ndarray,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Measure the same 16 landmarks optimized by the F26 solver."""
    from .f26_solver import F26_TARGET_LINKS

    direct_body_ids: dict[str, int] = {}
    for _, link_name in F26_TARGET_LINKS:
        if link_name in _F26_FOOT_MARKER_OFFSETS_M:
            continue
        direct_body_ids[link_name] = _id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    values: dict[str, list[float]] = {name: [] for name, _ in F26_TARGET_LINKS}
    squared_errors: list[np.ndarray] = []
    data = mujoco.MjData(model)
    for frame, qpos in enumerate(trajectory):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for target_index, (name, link_name) in enumerate(F26_TARGET_LINKS):
            if link_name in _F26_FOOT_MARKER_OFFSETS_M:
                parent_name = link_name.replace("_foot_link", "_ankle_roll_link")
                parent_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, parent_name)
                position = (
                    data.xpos[parent_id]
                    + data.xmat[parent_id].reshape(3, 3) @ _F26_FOOT_MARKER_OFFSETS_M[link_name]
                )
            else:
                position = data.xpos[direct_body_ids[link_name]]
            error = np.asarray(position) - target_keypoints[frame, target_index]
            squared_errors.append(np.square(error))
            values[name].append(float(np.linalg.norm(error)))
    summary = {
        name: {
            "rms": float(np.sqrt(np.mean(np.square(errors)))),
            "p95": float(np.percentile(errors, 95)),
            "max": float(np.max(errors)),
        }
        for name, errors in values.items()
    }
    return float(np.sqrt(np.mean(np.asarray(squared_errors)))), summary


def _retarget_motion(
    source_motion: MotionClip,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    max_frames: int | None = None,
    output_fps: float = 50.0,
    source_metadata_path: str | Path,
    source_kind: str = "GEM-X",
    source_joint_aliases: dict[str, str] | None = None,
    append_safe_stand: bool | None = None,
    prepend_safe_stand: bool | None = None,
    initial_pose_mode: str | None = None,
    progress: ProgressCallback | None = None,
) -> RetargetResult:
    """Fit one articulated human motion to a 50 Hz Humanoid Light v2 policy."""
    _, _, _, _, pk, yourdfpy = _require_pyroki()
    from .f26_solver import solve_f26_trajectory as solve_uncached

    source_metadata_path = Path(source_metadata_path).resolve()
    output_dir = Path(output_dir).resolve()
    config, resolved_config = _load_config(config_path)
    if initial_pose_mode is not None:
        config = dict(config)
        config["initial_pose"] = {**dict(config.get("initial_pose", {})), "mode": initial_pose_mode}
    pose_settings = dict(config.get("initial_pose", {}))
    pose_mode = str(pose_settings.get("mode", "upright"))
    if pose_mode not in {"upright", "zero_supine"}:
        raise ValueError("initial_pose.mode must be 'upright' or 'zero_supine'.")
    if source_joint_aliases:
        config = dict(config)
        config["source_joint_aliases"] = {
            **dict(config.get("source_joint_aliases", {})),
            **{str(key): str(value) for key, value in source_joint_aliases.items()},
        }
    if prepend_safe_stand is not None:
        config = dict(config)
        config["initial_transition"] = {
            **dict(config.get("initial_transition", {})),
            "enabled": bool(prepend_safe_stand),
        }
    if append_safe_stand is not None:
        config = dict(config)
        config["terminal_transition"] = {
            **dict(config.get("terminal_transition", {})),
            "enabled": bool(append_safe_stand),
        }
    raw_model_path = Path(config["model_path"])
    model_path = (
        raw_model_path if raw_model_path.is_absolute() else WORKSPACE_ROOT / raw_model_path
    ).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Target MJCF was not found: {model_path}")
    if progress:
        progress(f"{source_kind} 모션을 읽고 PyRoki 입력을 준비하는 중…")
    output_clip, output_source_positions, output_source_rotations = _prepare_clip(
        source_motion, config, max_frames, output_fps
    )
    model_path = _ensure_retarget_model(model_path, config)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos_addresses, lower, upper = _validate_and_index_model(model, config)
    initial_root_qpos = model.qpos0.copy()
    supine_diagnostics: dict[str, float | str] = {}
    if pose_mode == "zero_supine":
        from .terminal_transition import grounded_zero_supine_target

        initial_root_qpos, supine_diagnostics = grounded_zero_supine_target(
            model.qpos0, model, qpos_addresses, lower, upper,
            ground_clearance_m=float(pose_settings.get("ground_clearance_m", 0.005)),
        )
    solver_config = config.get("f26_solver", {})
    clip = output_clip
    source_positions = output_source_positions
    source_rotations = output_source_rotations
    max_full_sequence_frames = int(solver_config.get("max_full_sequence_frames", 450))
    requested_solver_fps = float(solver_config.get("long_motion_solver_fps", output_fps))
    if output_clip.frame_count > max_full_sequence_frames and requested_solver_fps < output_fps:
        solver_fps = max(1.0, requested_solver_fps)
        clip, source_positions, source_rotations = _prepare_clip(
            source_motion, config, max_frames, solver_fps
        )
        if progress:
            progress(
                f"긴 영상 입력은 {clip.fps:.0f} Hz에서 F26 최적화 후 "
                f"{output_fps:.0f} Hz 정책으로 보간합니다."
            )
    (
        target_keypoints,
        root_rotation_matrices,
        left_foot_contact,
        right_foot_contact,
        contact_summary,
        chain_scales,
        support_masks,
    ) = _f26_target_trajectory(
        clip, source_positions, source_rotations, model, config, initial_root_qpos
    )
    urdf_path = _export_pyroki_urdf(model_path, output_dir, "pelvis_link")
    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(str(urdf_path)))

    def solve_f26_trajectory(*args, **kwargs):
        # Resume only when every numerical input, robot, objective and setting
        # matches. Projection/export fixes need not repeat the expensive solve.
        import hashlib
        digest = hashlib.sha256(urdf_path.read_bytes())
        digest.update((Path(__file__).parent / "f26_solver.py").read_bytes())

        def fingerprint(value):
            if isinstance(value, dict):
                for key in sorted(value):
                    if key != "progress":
                        digest.update(key.encode())
                        fingerprint(value[key])
            elif value is None or isinstance(value, (str, int, float, bool)):
                digest.update(repr(value).encode())
            else:
                array = np.asarray(value)
                digest.update(str((array.shape, array.dtype)).encode())
                digest.update(np.ascontiguousarray(array).tobytes())

        for value in args[1:]:
            fingerprint(value)
        fingerprint(kwargs)
        key = digest.hexdigest()
        cache = output_dir / "solver_solution.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved["input_sha256"].item()) == key:
                    if progress:
                        progress("동일 입력·목적함수의 PyRoki 체크포인트를 재사용합니다.")
                    return saved["bases"].copy(), saved["joints"].copy()
        bases, joints = solve_uncached(*args, **kwargs)
        np.savez_compressed(cache, input_sha256=np.asarray(key), bases=bases, joints=joints)
        return bases, joints
    joint_order = list(config["joint_order"])
    robot_joint_index = {name: index for index, name in enumerate(robot.joints.actuated_names)}
    missing_joints = [name for name in joint_order if name not in robot_joint_index]
    if missing_joints:
        raise RuntimeError("PyRoki URDF is missing controlled joints: " + ", ".join(missing_joints))
    if progress:
        progress("F26 PyRoki 전체 시퀀스(접지·속도 제한 포함)를 최적화하는 중…")
    supine_support_settings = dict(config.get("supine_support", {}))
    solver_support_contacts = np.column_stack([
        support_masks.get(name, np.zeros(clip.frame_count, dtype=bool))
        for name, _, _ in _SUPPORT_LINKS
    ]).astype(np.float64)
    solved_bases, solved_joints = solve_f26_trajectory(
        robot,
        target_keypoints,
        root_rotation_matrices,
        left_foot_contact,
        right_foot_contact,
        fps=clip.fps,
        weights=solver_config.get("weights", {}),
        sole_offsets_m=_sole_offsets_from_mjcf(model),
        ground_geometry=_ground_geometry_from_mjcf(model, robot),
        support_contacts=solver_support_contacts,
        support_offsets_m=_support_offsets_from_mjcf(model, initial_root_qpos),
        support_ground_weight=(
            float(supine_support_settings.get("support_ground_weight", 60.0))
            if pose_mode == "zero_supine" else 0.0
        ),
        support_height_m=float(supine_support_settings.get("support_height_m", 0.015)) * 0.5,
        root_orientation_weight=(
            float(supine_support_settings.get("root_orientation_weight", 20.0))
            if pose_mode == "zero_supine" else float(solver_config.get("weights", {}).get("pelvis_orientation", 0.0))
        ),
        initial_root_orientation_weight=(
            float(supine_support_settings.get("initial_root_orientation_weight", 50.0))
            if pose_mode == "zero_supine" else 0.0
        ),
        root_vertical_velocity_limit_m_s=(
            float(supine_support_settings.get("root_vertical_velocity_limit_m_s", 0.8))
            if pose_mode == "zero_supine" else 0.0
        ),
        root_vertical_velocity_weight=(
            float(supine_support_settings.get("root_vertical_velocity_weight", 20.0))
            if pose_mode == "zero_supine" else 0.0
        ),
        ground_height_m=float(config.get("physical_constraints", {}).get("contact_height_m", 0.005)),
        max_iterations=int(solver_config.get("max_iterations", 800)),
        long_motion_max_iterations=int(
            solver_config.get(
                "long_motion_max_iterations", solver_config.get("max_iterations", 800)
            )
        ),
        max_full_sequence_frames=max_full_sequence_frames,
        window_frames=int(solver_config.get("window_frames", 320)),
        overlap_frames=int(solver_config.get("overlap_frames", 32)),
        progress=progress,
    )

    trajectory = np.tile(model.qpos0, (clip.frame_count, 1))
    trajectory[:, :3] = solved_bases[:, 4:]
    trajectory[:, 3:7] = solved_bases[:, :4]
    for index, name in enumerate(joint_order):
        trajectory[:, qpos_addresses[index]] = solved_joints[:, robot_joint_index[name]]
    trajectory[:, qpos_addresses] = np.clip(trajectory[:, qpos_addresses], lower, upper)
    trajectory[:, 3:7] /= np.linalg.norm(trajectory[:, 3:7], axis=1, keepdims=True)

    error_keypoints = target_keypoints
    output_left_contact = left_foot_contact.reshape(-1)
    output_right_contact = right_foot_contact.reshape(-1)
    if not np.isclose(clip.fps, output_clip.fps):
        trajectory = _resample_qpos_trajectory(trajectory, clip.fps, output_clip.fps,
                                             output_clip.frame_count)
        clip = output_clip
        (
            error_keypoints,
            _,
            output_left_contact,
            output_right_contact,
            contact_summary,
            chain_scales,
            support_masks,
        ) = _f26_target_trajectory(
            output_clip,
            output_source_positions,
            output_source_rotations,
            model,
            config,
            initial_root_qpos,
        )

    # _f26_target_trajectory returns [frames, 1]. np.pad on that 2-D array
    # would pad both axes and silently corrupt the full-rate contact labels.
    output_left_contact = np.asarray(output_left_contact).reshape(-1)
    output_right_contact = np.asarray(output_right_contact).reshape(-1)
    initial_settings = dict(config.get("initial_transition", {}))
    initial_metadata: dict[str, object] = {"enabled": False}
    if pose_mode == "zero_supine":
        from .terminal_transition import prepend_zero_supine_transition

        initial = prepend_zero_supine_transition(
            trajectory, initial_root_qpos,
            fps=output_clip.fps,
            zero_hold_s=float(pose_settings.get("zero_hold_s", 0.5)),
            transition_s=float(pose_settings.get("transition_s", 1.0)),
            initial_hold_s=float(pose_settings.get("initial_hold_s", 0.25)),
            diagnostics=supine_diagnostics,
        )
        trajectory = initial.qpos
        output_left_contact = np.pad(output_left_contact, (initial.added_frames, 0))
        output_right_contact = np.pad(output_right_contact, (initial.added_frames, 0))
        support_masks = {name: np.pad(mask, (initial.added_frames, 0))
                         for name, mask in support_masks.items()}
        initial_metadata = initial.metadata(output_clip.fps)
    elif bool(initial_settings.get("enabled", False)):
        from .terminal_transition import prepend_safe_stand_transition

        if progress:
            progress("첫 위치·방향의 정자세에서 모션으로 이어지는 시작 구간을 계산하는 중…")
        initial = prepend_safe_stand_transition(
            trajectory, model, qpos_addresses, lower, upper,
            fps=output_clip.fps,
            zero_hold_s=float(initial_settings.get("zero_hold_s", 1.0)),
            transition_s=float(initial_settings.get("transition_s", 2.0)),
            initial_hold_s=float(initial_settings.get("initial_hold_s", 0.5)),
            ground_clearance_m=float(initial_settings.get("ground_clearance_m", 0.005)),
        )
        trajectory = initial.qpos
        output_left_contact = np.pad(output_left_contact, (initial.added_frames, 0))
        output_right_contact = np.pad(output_right_contact, (initial.added_frames, 0))
        support_masks = {name: np.pad(mask, (initial.added_frames, 0))
                         for name, mask in support_masks.items()}
        initial_metadata = initial.metadata(output_clip.fps)
    terminal_settings = dict(config.get("terminal_transition", {}))
    terminal_metadata: dict[str, object] = {"enabled": False}
    if bool(terminal_settings.get("enabled", False)):
        from .terminal_transition import append_safe_stand_transition

        if progress:
            progress("마지막 XY·방향을 보존한 grounded zero stand 전이 구간을 계산하는 중…")
        terminal = append_safe_stand_transition(
            trajectory,
            model,
            qpos_addresses,
            lower,
            upper,
            fps=output_clip.fps,
            final_hold_s=float(terminal_settings.get("final_hold_s", 0.5)),
            transition_s=float(terminal_settings.get("transition_s", 2.0)),
            zero_hold_s=float(terminal_settings.get("zero_hold_s", 1.0)),
            ground_clearance_m=float(terminal_settings.get("ground_clearance_m", 0.005)),
        )
        trajectory = terminal.qpos
        output_left_contact = np.pad(output_left_contact, (0, terminal.added_frames))
        output_right_contact = np.pad(output_right_contact, (0, terminal.added_frames))
        support_masks = {name: np.pad(mask, (0, terminal.added_frames))
                         for name, mask in support_masks.items()}
        terminal_metadata = terminal.metadata(output_clip.fps)
        terminal_metadata["start_frame"] = terminal.source_frames
    physical_settings = dict(config.get("physical_constraints", {}))
    if config.get("contacts", {}).get("mode") == "supported_dance":
        # The standing prefix/suffix must also remain supported. Select the
        # lower foot during their blend, keeping exact zero-joint holds intact.
        data = mujoco.MjData(model)
        ground = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        feet = [_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_ankle_roll_link_collision")
                for side in ("left", "right")]
        start = int(initial_metadata.get("added_frames", 0))
        for frame in [*range(start), *range(start + clip.frame_count, len(trajectory))]:
            data.qpos[:] = trajectory[frame]
            mujoco.mj_forward(model, data)
            heights = [mujoco.mj_geomDistance(model, data, foot, ground, 2.0, np.zeros(6)) for foot in feet]
            output_left_contact[frame] = heights[0] <= min(heights) + 0.002
            output_right_contact[frame] = heights[1] <= min(heights) + 0.002
    physical_diagnostics: dict[str, object] = {"enabled": False}
    if bool(physical_settings.get("enabled", True)):
        from .physical_constraints import project_physical_trajectory

        if progress:
            progress("실제 MuJoCo 충돌 메시로 발 접지와 셀프 콜리션을 검사·보정하는 중…")
        trajectory, physical_diagnostics = project_physical_trajectory(
            model, trajectory, qpos_addresses, lower, upper,
            output_left_contact, output_right_contact,
            support_contacts=support_masks,
            contact_height_m=float(physical_settings.get("contact_height_m", 0.005)),
            support_height_m=float(config.get("supine_support", {}).get("support_height_m", 0.015)),
            collision_margin_m=float(physical_settings.get("collision_margin_m", 0.001)),
            tolerance_m=float(physical_settings.get("tolerance_m", 0.0005)),
            progress=progress,
            max_stance_tilt_degrees=physical_settings.get("max_stance_tilt_degrees"),
        )
        physical_diagnostics["enabled"] = True
        passes = int(physical_settings.get("temporal_refinement_passes", 0))
        if passes > 0:
            from .physical_constraints import refine_supported_trajectory
            fixed_holds = np.zeros(len(trajectory), dtype=bool)
            initial_end = int(initial_metadata.get("added_frames", 0))
            initial_hold = int(initial_metadata.get("initial_hold_frames", 0))
            if initial_hold:
                fixed_holds[initial_end - initial_hold:initial_end] = True
            terminal_start = int(terminal_metadata.get("start_frame", len(trajectory)))
            terminal_hold = int(terminal_metadata.get("final_hold_frames", 0))
            fixed_holds[terminal_start:terminal_start + terminal_hold] = True
            if progress:
                progress("접지·충돌 조건을 유지하면서 관절 궤적의 급격한 가속도를 줄이는 중…")
            trajectory, refinement = refine_supported_trajectory(
                model, trajectory, qpos_addresses, lower, upper,
                output_left_contact, output_right_contact,
                fps=output_clip.fps, passes=passes, fixed_pose_mask=fixed_holds,
                support_contacts=support_masks,
                contact_height_m=float(physical_settings.get("contact_height_m", 0.005)),
                support_height_m=float(config.get("supine_support", {}).get("support_height_m", 0.015)),
                collision_margin_m=float(physical_settings.get("collision_margin_m", 0.001)),
                tolerance_m=float(physical_settings.get("tolerance_m", 0.0005)),
                progress=progress,
                max_stance_tilt_degrees=physical_settings.get("max_stance_tilt_degrees"),
            )
            physical_diagnostics["temporal_refinement"] = refinement
    source_start_frame = int(initial_metadata.get("added_frames", 0))
    from .quality import evaluate_reference, require_valid_reference
    quality = evaluate_reference(model, trajectory, output_clip.fps,
                                 output_left_contact.reshape(-1), output_right_contact.reshape(-1),
                                 contact_height_m=float(physical_settings.get("contact_height_m", 0.005)),
                                 tolerance_m=float(physical_settings.get("tolerance_m", 0.0005)),
                                 max_stance_tilt_degrees=physical_settings.get("max_stance_tilt_degrees"))
    require_valid_reference(quality)
    hold_ranges = []
    if initial_metadata.get("enabled"):
        lead_end = int(initial_metadata["added_frames"])
        hold_ranges.extend([
            (0, int(initial_metadata.get("zero_hold_frames", 0))),
            (lead_end - int(initial_metadata.get("initial_hold_frames", 0)), lead_end),
        ])
    if terminal_metadata.get("enabled"):
        tail_start = int(terminal_metadata["start_frame"])
        hold_ranges.extend([
            (tail_start, tail_start + int(terminal_metadata.get("final_hold_frames", 0))),
            (len(trajectory) - int(terminal_metadata.get("zero_hold_frames", 0)), len(trajectory)),
        ])
    variation = max((float(np.max(np.ptp(trajectory[start:end], axis=0)))
                     for start, end in hold_ranges if end > start), default=0.0)
    quality["stationary_hold_max_coordinate_variation"] = variation
    if variation > 1e-6:
        raise RuntimeError(f"Stationary hold was changed by refinement: {variation:.6g}")
    position_rms_m, position_error_summary = _f26_position_errors(
        model, trajectory[source_start_frame:source_start_frame + clip.frame_count], error_keypoints
    )
    command_positions = trajectory[:, qpos_addresses]
    if np.any(command_positions < lower - 1e-7) or np.any(command_positions > upper + 1e-7):
        raise RuntimeError("PyRoki output violates a Humanoid Light v2 joint limit.")
    mujoco_joint_order = [
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        for joint_id in range(1, model.njnt)
    ]
    result = _write_outputs(
        output_dir,
        model_path,
        resolved_config,
        config,
        clip,
        trajectory,
        qpos_addresses,
        joint_order,
        mujoco_joint_order,
        position_rms_m,
        position_error_summary,
        contact_summary,
        chain_scales,
        source_metadata_path,
        source_motion.fps,
        terminal_metadata,
        initial_metadata,
        initial_root_qpos,
        left_contact=output_left_contact,
        right_contact=output_right_contact,
        support_contacts=support_masks,
        target_keypoints_m=error_keypoints,
        output_stem=("gemx" if source_kind == "GEM-X" else "kimodo" if source_kind == "Kimodo" else "external") + "_50hz_retargeted",
        progress=progress,
    )
    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    # ``source_gemx_motion`` remains for older exports and tools.  New code
    # reads the source-neutral field, allowing Kimodo and GEM-X to share the
    # exact output contract and viewer.
    metadata["source_motion"] = str(source_metadata_path)
    metadata["source_gemx_motion"] = str(source_metadata_path)
    metadata["source_kind"] = source_kind
    metadata["terminal_transition"] = terminal_metadata
    metadata["physical_constraints"] = physical_diagnostics
    metadata["quality"] = quality
    metadata["solver_settings"] = solver_config
    metadata["position_rms_3d_m"] = float(position_rms_m * np.sqrt(3.0))
    # Save the exact configuration and support labels beside the selected result.
    (output_dir / "retarget_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    metadata["retargeter_config"] = str(output_dir / "retarget_config.json")
    import hashlib
    metadata["provenance"] = {
        "source_motion_sha256": hashlib.sha256(source_metadata_path.read_bytes()).hexdigest(),
        "target_mjcf_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in ("f26_solver.py", "pyroki_retargeter.py", "physical_constraints.py", "quality.py", "terminal_transition.py", "simulation_export.py")},
    }
    reset_qvel = metadata["initial_pose"]["reset_qvel"]
    metadata["initial_pose"] = {
        "mode": pose_mode,
        "reset_qpos": trajectory[0].tolist(),
        "reset_qvel": reset_qvel,
    }
    metadata["retargeter"] = {
        "name": "PyRoki",
        "revision": PYROKI_GIT_REVISION,
        "kinematic_urdf": str(urdf_path.resolve()),
        "solver": "F26 full-sequence JAX least-squares followed by exact MuJoCo foot-ground and self-collision projection",
    }
    metadata["reference_policy"] = {
        "control_rate_hz": float(result.fps),
        "action_space": f"{len(joint_order)} {config.get('robot_name', 'robot')} joint-position targets in radians",
        "representation": "MuJoCo qpos reference trajectory",
        "note": "Kinematic reference policy input; not motor torques or a hardware-safe control policy.",
    }
    # The viewer reads the exported settings while building its source panel.
    result.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    from .simulation_export import update_archive_metadata
    update_archive_metadata(result.npz, metadata)
    if progress:
        progress(f"PyRoki {result.fps:.0f} Hz 레퍼런스 정책 저장 완료: {result.npz}")
    return result


def retarget_motion_clip(
    source_motion: MotionClip,
    output_dir: str | Path,
    *,
    source_motion_path: str | Path,
    source_kind: str,
    source_joint_aliases: dict[str, str] | None = None,
    append_safe_stand: bool | None = None,
    prepend_safe_stand: bool | None = None,
    initial_pose_mode: str | None = None,
    config_path: str | Path | None = None,
    max_frames: int | None = None,
    output_fps: float = 50.0,
    progress: ProgressCallback | None = None,
) -> RetargetResult:
    """Retarget an already-loaded human motion through the common F26 path.

    This is deliberately source-neutral.  GEM-X and Kimodo differ only in
    how they create ``MotionClip``; the solver and nine-key simulation export
    are shared with external human datasets.
    """
    return _retarget_motion(
        source_motion,
        output_dir,
        config_path=config_path,
        max_frames=max_frames,
        output_fps=output_fps,
        source_metadata_path=source_motion_path,
        source_kind=source_kind,
        source_joint_aliases=source_joint_aliases,
        append_safe_stand=append_safe_stand,
        prepend_safe_stand=prepend_safe_stand,
        initial_pose_mode=initial_pose_mode,
        progress=progress,
    )


def retarget_gemx_hpe(
    hpe_results_path: str | Path,
    source_video_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    max_frames: int | None = None,
    output_fps: float = 50.0,
    append_safe_stand: bool | None = None,
    prepend_safe_stand: bool | None = None,
    initial_pose_mode: str | None = None,
    progress: ProgressCallback | None = None,
) -> RetargetResult:
    """Create a 50 Hz PyRoki policy directly from saved GEM-X output."""
    from apps.gemx.motion import load_gemx_motion

    output_dir = Path(output_dir).resolve()
    if progress:
        progress("GEM-X 모션을 PyRoki 입력으로 준비하는 중…")
    source_motion = load_gemx_motion(hpe_results_path, source_video_path)
    source_path = save_visualization_motion(source_motion, output_dir / "gemx_extracted_motion.npz")
    return retarget_motion_clip(
        source_motion,
        output_dir,
        config_path=config_path,
        max_frames=max_frames,
        output_fps=output_fps,
        source_motion_path=source_path,
        source_kind="GEM-X",
        append_safe_stand=append_safe_stand,
        prepend_safe_stand=prepend_safe_stand,
        initial_pose_mode=initial_pose_mode,
        progress=progress,
    )
