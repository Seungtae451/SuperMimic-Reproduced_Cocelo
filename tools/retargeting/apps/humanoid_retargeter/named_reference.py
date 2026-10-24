"""Template-exact named reference export using MuJoCo FK and USD fixed frames.

Only the nine simulation keys are written. Names, order and dtypes come from
an NPZ template; robot joint/body counts are discovered from names and MJCF.
USD supplies fixed sensor transforms, never articulated motion or dynamics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


KEYS = ("fps", "joint_names", "body_names", "joint_pos", "joint_vel", "body_pos_w",
        "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def rotations(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]])


def continuous_quaternions(values):
    values = np.array(values, dtype=np.float64, copy=True)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-8) or not np.isfinite(values).all():
        raise ValueError("Expected finite, nonzero WXYZ quaternions.")
    values /= norms
    for frame in range(1, len(values)):
        flip = np.sum(values[frame - 1] * values[frame], axis=-1) < 0
        values[frame] *= np.where(flip, -1., 1.)[..., None]
    return values


@dataclass(frozen=True)
class FixedFrame:
    parent: str
    pos: np.ndarray
    quat: np.ndarray  # WXYZ, child orientation in parent coordinates

    def as_dict(self):
        return dict(parent=self.parent, pos=np.asarray(self.pos).tolist(),
                    quat_wxyz=np.asarray(self.quat).tolist())


def fixed_frames_from_usd(path, names):
    """Extract exact parent-to-child transforms, including both joint anchors.

    Import USD only for asset extraction. The generated JSON can be used by
    the GUI/exporters without an Isaac Lab or USD runtime dependency.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if not stage or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Expected a valid Z-up USD stage.")
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.Joint(prim)
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(children) != 1 or children[0].name not in names:
            continue
        if len(parents) != 1 or children[0].name in result:
            raise ValueError(f"Ambiguous fixed frame: {prim.GetName()}.")
        def rotation(value):
            return Rotation.from_quat([*value.GetImaginary(), value.GetReal()])
        r0, r1 = rotation(joint.GetLocalRot0Attr().Get()), rotation(joint.GetLocalRot1Attr().Get())
        rot = r0 * r1.inv()
        pos = np.asarray(joint.GetLocalPos0Attr().Get()) * scale - rot.apply(
            np.asarray(joint.GetLocalPos1Attr().Get()) * scale)
        result[children[0].name] = FixedFrame(parents[0].name, pos, rot.as_quat()[[3, 0, 1, 2]])
    if set(result) != set(names):
        raise ValueError(f"Missing USD fixed frames: {set(names) - set(result)}.")
    return result


class MujocoNamedRobot:
    def __init__(self, model, fixed_frames=None):
        self.model = model
        self.fixed_frames = dict(fixed_frames or {})
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or model.body_parentid[model.jnt_bodyid[free[0]]] != 0:
            raise ValueError("Expected one floating root body.")
        self.free_id = int(free[0])
        self.root = model.body(int(model.jnt_bodyid[self.free_id])).name
        self.joints = [j for j in range(model.njnt) if j != self.free_id]
        if any(int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
               for j in self.joints):
            raise ValueError("Named joint arrays require scalar hinge/slide joints.")
        self.active_names = [model.joint(j).name for j in self.joints]
        if any(not name for name in self.active_names) or len(set(self.active_names)) != len(self.joints):
            raise ValueError("Each active joint must have a unique name.")
        self.root_qpos = int(model.jnt_qposadr[self.free_id])
        self.root_dof = int(model.jnt_dofadr[self.free_id])

    @classmethod
    def from_config(cls, model, config):
        return cls(model, {name: FixedFrame(item["parent"], np.asarray(item["pos"]),
                                           np.asarray(item["quat_wxyz"]))
                           for name, item in config.get("fixed_frames", {}).items()})

    def _frame(self, name, trail=()):
        if name in trail:
            raise ValueError(f"Fixed frame cycle: {name}.")
        if name in self.fixed_frames:
            frame = self.fixed_frames[name]
            body, parent_pos, parent_rot = self._frame(frame.parent, (*trail, name))
            return body, parent_pos + parent_rot.apply(frame.pos), parent_rot * rotations(frame.quat)
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body <= 0:
            raise ValueError(f"No MuJoCo body or USD fixed frame named {name}.")
        return body, np.zeros(3), Rotation.identity()

    def qpos_from_poses(self, root_pos, root_quat, joint_pos, joint_names):
        if len(joint_names) != len(self.active_names) or set(joint_names) != set(self.active_names):
            raise ValueError("joint_names must map every active robot joint exactly once.")
        root_pos, joint_pos = np.asarray(root_pos), np.asarray(joint_pos)
        n = len(root_pos)
        if n < 1 or root_pos.shape != (n, 3) or np.shape(root_quat) != (n, 4) or joint_pos.shape != (n, len(joint_names)):
            raise ValueError("Inconsistent root/joint state dimensions.")
        if not all(np.isfinite(v).all() for v in (root_pos, root_quat, joint_pos)):
            raise ValueError("Motion contains non-finite states.")
        qpos = np.tile(self.model.qpos0, (n, 1))
        address = self.root_qpos
        qpos[:, address:address + 3] = root_pos
        qpos[:, address + 3:address + 7] = continuous_quaternions(root_quat)
        indices = {name: i for i, name in enumerate(joint_names)}
        for joint in self.joints:
            qpos[:, self.model.jnt_qposadr[joint]] = joint_pos[:, indices[self.model.joint(joint).name]]
        return qpos

    def differentiate(self, qpos, fps):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite.")
        velocities = np.zeros((len(qpos), self.model.nv))
        a, d = self.root_qpos, self.root_dof
        for t in range(len(qpos)):
            lo, hi = max(0, t - 1), min(len(qpos) - 1, t + 1)
            if hi == lo:
                continue
            mujoco.mj_differentiatePos(self.model, velocities[t], (hi - lo) / fps, qpos[lo], qpos[hi])
            # MuJoCo's quaternion difference is local to lo. Express it in
            # this frame's floating-base basis before applying Jacobians.
            velocities[t, d + 3:d + 6] = rotations(qpos[t, a + 3:a + 7]).inv().apply(
                rotations(qpos[lo, a + 3:a + 7]).apply(velocities[t, d + 3:d + 6]))
        return velocities

    def states(self, root_pos, root_quat, joint_pos, joint_names, body_names, fps):
        qpos = self.qpos_from_poses(root_pos, root_quat, joint_pos, joint_names)
        if len(set(body_names)) != len(body_names):
            raise ValueError("Body names must be unique.")
        frames = [self._frame(name) for name in body_names]
        qvel = self.differentiate(qpos, fps)
        n, b = len(qpos), len(body_names)
        pos, lin, ang = (np.empty((n, b, 3)) for _ in range(3))
        quat = np.empty((n, b, 4))
        data = mujoco.MjData(self.model)
        jacp, jacr = np.empty((3, self.model.nv)), np.empty((3, self.model.nv))
        for t, pose in enumerate(qpos):
            data.qpos[:] = pose
            # Kinematics and COM transforms prepare Jacobians without a
            # collision/dynamics rollout; velocities refer to link origins.
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            for index, (body, offset, local_rot) in enumerate(frames):
                rotation = rotations(data.xquat[body])
                pos[t, index] = data.xpos[body] + rotation.apply(offset)
                quat[t, index] = (rotation * local_rot).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_jac(self.model, data, jacp, jacr, pos[t, index], body)
                lin[t, index], ang[t, index] = jacp @ qvel[t], jacr @ qvel[t]
        dofs = [int(self.model.joint(name).dofadr[0]) for name in joint_names]
        return dict(joint_pos=np.asarray(joint_pos), joint_vel=qvel[:, dofs], body_pos_w=pos,
                    body_quat_w=continuous_quaternions(quat), body_lin_vel_w=lin, body_ang_vel_w=ang)


def read_legacy_motion(path, output_joint_names, root_body=None):
    """Read retargeted source poses, honoring angle-name order before controller names."""
    with np.load(path, allow_pickle=False) as archive:
        if {"base_frame_pos", "base_frame_wxyz", "joint_angles"}.issubset(archive.files):
            pos, quat, joints = (np.asarray(archive[k], dtype=np.float64)
                                 for k in ("base_frame_pos", "base_frame_wxyz", "joint_angles"))
            key = next((k for k in ("joint_angle_names", "mujoco_qpos_joint_names", "joint_names") if k in archive), None)
        elif {"joint_pos", "joint_names", "body_names", "body_pos_w", "body_quat_w"}.issubset(archive.files):
            bodies = archive["body_names"].tolist()
            root = bodies.index(root_body) if root_body else 0
            pos = np.asarray(archive["body_pos_w"][:, root], dtype=np.float64)
            quat = np.asarray(archive["body_quat_w"][:, root], dtype=np.float64)
            joints = np.asarray(archive["joint_pos"], dtype=np.float64)
            key = "joint_names"
        elif "qpos" in archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            pos, quat, joints = qpos[:, :3], qpos[:, 3:7], qpos[:, 7:]
            key = next((k for k in ("mujoco_qpos_joint_names", "joint_angle_names") if k in archive), None)
        else:
            raise ValueError("Input requires retargeted base poses and joint angles, or qpos.")
        source_names = archive[key].tolist() if key else list(output_joint_names)
        if len(set(source_names)) != len(source_names) or set(source_names) != set(output_joint_names):
            raise ValueError("Source joint names do not match output names.")
        joints = joints[:, [source_names.index(name) for name in output_joint_names]]
        fps = float(archive["fps"].item())
    n = len(pos)
    if pos.shape != (n, 3) or quat.shape != (n, 4) or joints.shape != (n, len(output_joint_names)) or n < 1:
        raise ValueError("Inconsistent input pose shapes.")
    if not np.isfinite(fps) or fps <= 0 or not all(np.isfinite(v).all() for v in (pos, quat, joints)):
        raise ValueError("Input contains non-finite poses or invalid fps.")
    quat = rotations(quat).as_quat()[:, [3, 0, 1, 2]]
    return pos, quat, joints, fps, key is None


def load_template(template=None):
    if template is not None:
        with np.load(template, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    from apps.paths import CONFIG_ROOT
    config = json.loads((CONFIG_ROOT / "reference_schema.json").read_text())
    return {key: np.asarray(config[key], dtype=config["dtypes"][key])
            if key in {"fps", "joint_names", "body_names"} else
            np.empty((1, *config["shapes"][key][1:]), dtype=config["dtypes"][key])
            for key in config["keys"]}


def arrays_like_template(robot, root_pos, root_quat, angles, fps, template=None):
    """Compute the exact schema; only the time dimension may vary."""
    spec = load_template(template)
    if tuple(spec) != KEYS:
        raise ValueError(f"Template keys/order must be {KEYS}.")
    joints, bodies = spec["joint_names"].tolist(), spec["body_names"].tolist()
    if not np.isclose(fps, float(spec["fps"].item()), rtol=1e-6, atol=1e-8):
        raise ValueError("Source fps differs from template; explicit resampling is required.")
    fps = float(spec["fps"].item())
    values = dict(fps=np.asarray(fps), joint_names=spec["joint_names"], body_names=spec["body_names"])
    values.update(robot.states(root_pos, root_quat, angles, joints, bodies, fps))
    values = {key: np.asarray(values[key], dtype=spec[key].dtype) for key in spec}
    for key in spec:
        expected = spec[key].shape
        if key not in {"fps", "joint_names", "body_names"}:
            expected = (len(root_pos), *expected[1:])
        if values[key].shape != expected:
            raise ValueError(f"Output {key} shape differs from template axes: {values[key].shape} vs {expected}.")
        if values[key].dtype.kind == "f" and not np.isfinite(values[key]).all():
            raise ValueError(f"Output {key} contains non-finite values.")
    return values


def default_robot(model):
    from apps.paths import CONFIG_ROOT
    return MujocoNamedRobot.from_config(model, json.loads((CONFIG_ROOT / "reference_fixed_frames.json").read_text()))


def qpos_from_named_archive(model, archive):
    """Reconstruct MuJoCo playback by names, without adding qpos to the NPZ."""
    robot = MujocoNamedRobot(model)
    required = {"joint_names", "joint_pos", "body_names", "body_pos_w", "body_quat_w"}
    if not required.issubset(archive.files):
        raise ValueError("Expected named joint and body pose arrays.")
    bodies = archive["body_names"].tolist()
    if len(set(bodies)) != len(bodies) or robot.root not in bodies:
        raise ValueError("body_names must contain the unique floating-root link.")
    root = bodies.index(robot.root)
    return robot.qpos_from_poses(archive["body_pos_w"][:, root], archive["body_quat_w"][:, root],
                                archive["joint_pos"], archive["joint_names"].tolist())


def arrays_from_qpos(model, qpos, fps, *, template=None, robot=None):
    robot = default_robot(model) if robot is None else robot
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError("qpos must have shape (T, model.nq).")
    names = load_template(template)["joint_names"].tolist()
    addresses = [int(model.joint(name).qposadr[0]) for name in names]
    a = robot.root_qpos
    return arrays_like_template(robot, qpos[:, a:a+3], qpos[:, a+3:a+7], qpos[:, addresses], fps, template)


def export_like_template(source, output, robot, template):
    """Write exactly the template's nine keys, dtype and name-array order."""
    spec = load_template(template)
    joints, bodies = spec["joint_names"].tolist(), spec["body_names"].tolist()
    pos, quat, angles, source_fps, assumed = read_legacy_motion(source, joints, robot.root)
    values = arrays_like_template(robot, pos, quat, angles, source_fps, template)
    fps = float(values["fps"].item())
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("Use a different output path to preserve the source.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)
    return dict(input=str(Path(source).resolve()), output=str(Path(output).resolve()), frames=len(pos),
        fps=fps, source_fps=source_fps, unnamed_joint_order_assumed_from_template=assumed,
        joint_count=len(joints), body_count=len(bodies), keys=list(values),
        dtypes={key: str(value.dtype) for key, value in values.items()},
        shapes={key: list(value.shape) for key, value in values.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--robot", type=Path, required=True, help="Current MuJoCo MJCF")
    parser.add_argument("--fixed-frames", type=Path, help="USD-derived fixed frame JSON")
    parser.add_argument("--sensor-usd", type=Path, help="Extract camera/IMU frames from this USD instead")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="Verification JSON outside the NPZ")
    args = parser.parse_args()
    if args.sensor_usd and args.fixed_frames:
        parser.error("Choose --sensor-usd or --fixed-frames.")
    config = json.loads(args.fixed_frames.read_text()) if args.fixed_frames else {}
    model = mujoco.MjModel.from_xml_path(str(args.robot.resolve()))
    robot = MujocoNamedRobot.from_config(model, config)
    if args.sensor_usd:
        with np.load(args.template, allow_pickle=False) as template:
            names = [name for name in template["body_names"].tolist()
                     if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0]
        robot = MujocoNamedRobot(model, fixed_frames_from_usd(args.sensor_usd, names))
        config = dict(usd_source=str(args.sensor_usd.resolve()),
                      usd_sha256=hashlib.sha256(args.sensor_usd.read_bytes()).hexdigest(),
                      fixed_frames={name: frame.as_dict() for name, frame in robot.fixed_frames.items()})
    report = dict(template=str(args.template.resolve()), robot=str(args.robot.resolve()),
                  robot_sha256=hashlib.sha256(args.robot.read_bytes()).hexdigest(),
                  fixed_frames=config, runs=[])
    for source in args.inputs:
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        result = export_like_template(source, args.output_dir / source.name, robot, args.template)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        result.update(input_sha256=before, output_sha256=hashlib.sha256(Path(result["output"]).read_bytes()).hexdigest())
        report["runs"].append(result)
        print(result["output"], result["frames"], result["shapes"]["body_pos_w"], flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
