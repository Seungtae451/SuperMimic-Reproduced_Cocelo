"""Humanoid V10 mjlab asset using the V9-derived simulation control profile."""

from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


ROOT = Path(__file__).resolve().parents[4]
XML = ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"
PROFILE = ROOT / "configs/robots/humanoid_v10/control_profile.json"
ROOT_BODY = "pelvis_link"
ANCHOR_BODY = "torso_roll_link"
FOOT_GEOM_REGEX = r"^(left|right)_ankle_roll_link_collision$"
TRACKED_BODIES = (
    "pelvis_link", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_roll_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_link",
)


def control_profile() -> dict:
    return json.loads(PROFILE.read_text())


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(XML))
    # SceneCfg supplies the shared terrain; the standalone XML's plane is only
    # for CPU retargeting and physics preflights.
    for geom in tuple(spec.worldbody.geoms):
        if geom.name == "ground":
            spec.delete(geom)
    return spec


def get_v10_robot_cfg() -> EntityCfg:
    profile = control_profile()
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(f"^{re.escape(name)}$",),
            stiffness=float(profile["joints"][name]["kp"]),
            damping=float(profile["joints"][name]["kd"]),
            effort_limit=float(profile["joints"][name]["effort_limit_nm"]),
            armature=float(profile["joints"][name]["armature"]),
        )
        for name in profile["joint_order"]
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.7089731)),
        spec_fn=get_spec,
        sort_actuators=True,
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.9),
    )


_PROFILE = control_profile()
# Keep the V9-style fine control near zero, but let a bounded policy reach the
# full V10 travel at |a|=1. The action term combines the linear and cubic
# components and clamps the asymmetric side to the actual joint limit.
ACTION_LINEAR_SCALE = {
    name: 0.25 * float(_PROFILE["joints"][name]["effort_limit_nm"])
    / float(_PROFILE["joints"][name]["kp"])
    for name in _PROFILE["joint_order"]
}
ACTION_SCALE = {
    name: max(
        abs(float(_PROFILE["joints"][name]["urdf_lower_rad"])),
        abs(float(_PROFILE["joints"][name]["urdf_upper_rad"])),
    )
    for name in _PROFILE["joint_order"]
}
ACTION_CUBIC_SCALE = {
    name: ACTION_SCALE[name] - ACTION_LINEAR_SCALE[name]
    for name in _PROFILE["joint_order"]
}
# Hard clamping already enforces the V10 joint limits. BONES-SEED retargeting
# often legitimately reaches those limits, so the G1 soft-limit penalty fights
# the reference. Reduce the rate cost for V10's normalized bounded actions.
ACTION_RATE_WEIGHT = -0.01
SOFT_JOINT_LIMIT_WEIGHT = 0.0
ACTION_LIMITS = {
    name: (
        float(_PROFILE["joints"][name]["urdf_lower_rad"]),
        float(_PROFILE["joints"][name]["urdf_upper_rad"]),
    )
    for name in _PROFILE["joint_order"]
}
