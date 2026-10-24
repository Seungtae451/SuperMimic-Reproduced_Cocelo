"""Measure a cheap G1-joint-to-V10 mapping without touching the direct library.

This is a diagnostic pilot, not a certified training-data converter. It reads
only eligible G1 CSVs from the BONES-SEED archive and compares available pairs
against independently retargeted V10 references.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import tarfile
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/retargeting"))

from supermimic.data.motion_convert import (  # noqa: E402
    G1_JOINT_NAMES, ResampledMotion, forward_kinematics, resample_motion,
)
from apps.humanoid_retargeter.quality import evaluate_reference  # noqa: E402

G1_XML = (Path.home() / ".cache/uv/git-v1/checkouts/0e43ab9fd1cf397c/8bb2bc9/"
          "src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml")
V10_XML = ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"
RENAMED = {
    "torso_yaw_joint": "waist_yaw_joint",
    "torso_pitch_joint": "waist_pitch_joint",
    "torso_roll_joint": "waist_roll_joint",
    "left_wrist_joint": "left_wrist_yaw_joint",
    "right_wrist_joint": "right_wrist_yaw_joint",
}


def load_g1_csv(payload: bytes) -> np.ndarray:
    stream = io.BytesIO(payload)
    header = stream.readline().decode("utf-8-sig").strip().split(",")
    expected = [f"{name}_dof" for name in G1_JOINT_NAMES]
    joint_columns = [column for column in header if column.endswith("_joint_dof")]
    if joint_columns != expected:
        raise ValueError("G1 joint columns do not match the known joint order")
    required = ["root_translateX", "root_translateY", "root_translateZ",
                "root_rotateX", "root_rotateY", "root_rotateZ", *expected]
    if not all(column in header for column in required):
        raise ValueError("G1 CSV is missing a required root/joint column")
    values = np.loadtxt(stream, delimiter=",", dtype=np.float64, ndmin=2)
    if values.shape[1] != len(header) or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("G1 CSV data are incomplete or nonfinite")
    col = {name: index for index, name in enumerate(header)}
    pos = values[:, [col[name] for name in required[:3]]] / 100.0
    euler = values[:, [col[name] for name in required[3:6]]]
    quat = Rotation.from_euler("xyz", euler, degrees=True).as_quat()
    joints = np.deg2rad(values[:, [col[name] for name in expected]])
    return np.concatenate((pos, quat, joints), axis=1)


def joint_mapping(g1: mujoco.MjModel, v10: mujoco.MjModel) -> list[tuple[int, int, float]]:
    g1_names = [g1.joint(i).name for i in range(1, g1.njnt)]
    v10_names = [v10.joint(i).name for i in range(1, v10.njnt)]
    if g1_names != list(G1_JOINT_NAMES):
        raise ValueError("G1 XML has an unexpected joint order")
    g1_data, v10_data = mujoco.MjData(g1), mujoco.MjData(v10)
    mujoco.mj_forward(g1, g1_data)
    mujoco.mj_forward(v10, v10_data)
    mapping = []
    for target_index, name in enumerate(v10_names):
        if name == "head_joint":
            continue
        source_index = g1_names.index(RENAMED.get(name, name))
        dot = float(np.dot(g1_data.xaxis[source_index + 1],
                           v10_data.xaxis[target_index + 1]))
        if abs(dot) < 0.65:
            raise ValueError(f"Ambiguous neutral joint axis: {name}: {dot:.3f}")
        mapping.append((target_index, source_index, math.copysign(1.0, dot)))
    return mapping


def map_motion(raw: np.ndarray, g1: mujoco.MjModel, v10: mujoco.MjModel,
               mapping: list[tuple[int, int, float]],
               calibration: dict[str, dict] | None = None) -> tuple[np.ndarray, ResampledMotion, dict]:
    source = resample_motion(raw, 120.0, 50.0)
    qpos = np.tile(v10.qpos0, (source.num_frames, 1))
    qpos[:, :3] = source.root_pos + (v10.qpos0[:3] - g1.qpos0[:3])
    qpos[:, 3:7] = source.root_quat
    mapped = np.zeros((source.num_frames, v10.njnt - 1))
    for target, original, sign in mapping:
        joint = v10.joint(target + 1).name
        coeff = (calibration or {}).get(joint, {})
        mapped[:, target] = (sign * source.joint_pos[:, original]
                             * coeff.get("scale", 1.0) + coeff.get("offset", 0.0))
    bounds = v10.jnt_range[1:]
    excess = np.maximum(bounds[:, 0] - mapped, mapped - bounds[:, 1]).clip(min=0)
    qpos[:, 7:] = np.clip(mapped, bounds[:, 0], bounds[:, 1])
    ground = mujoco.mj_name2id(v10, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    collision_geoms = [i for i in range(v10.ngeom)
                       if "_collision" in (v10.geom(i).name or "")
                       and "_mesh_visual" not in (v10.geom(i).name or "")]
    data = mujoco.MjData(v10)
    lift = np.zeros(len(qpos))
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(v10, data)
        floor = min(mujoco.mj_geomDistance(v10, data, geom, ground, 2.0, np.zeros(6))
                    for geom in collision_geoms)
        lift[frame] = max(0.0, 0.001 - floor)
    qpos[:, 2] += lift
    velocities = np.gradient(qpos[:, 7:], 1 / 50.0, axis=0)
    root_velocities = np.gradient(qpos[:, :3], 1 / 50.0, axis=0)
    mapped_motion = ResampledMotion(
        fps=50.0, root_pos=qpos[:, :3], root_quat=qpos[:, 3:7],
        joint_pos=qpos[:, 7:], root_lin_vel_w=root_velocities,
        root_ang_vel_w=source.root_ang_vel_w, joint_vel=velocities,
    )
    diagnostics = {
        "frames": len(qpos),
        "clipped_frame_fraction": float(np.mean(np.any(excess > 1e-6, axis=1))),
        "maximum_joint_limit_excess_rad": float(np.max(excess)),
        "maximum_joint_speed_rad_s": float(np.max(np.abs(velocities))),
        "maximum_root_lift_m": float(np.max(lift)),
        "root_lifted_frame_fraction": float(np.mean(lift > 1e-6)),
    }
    return qpos, mapped_motion, diagnostics


def pose_error_to_direct(bodies, direct: Path) -> tuple[float | None, float | None]:
    if not direct.is_file():
        return None, None
    with np.load(direct, allow_pickle=False) as data:
        reference = np.asarray(data["body_pos_w"], dtype=np.float64)
        ref_root = np.asarray(data["body_quat_w"], dtype=np.float64)
        names = [str(name) for name in data["body_names"]]
    if names != bodies.body_names or len(reference) != len(bodies.pos_w):
        return None, None
    root = names.index("pelvis_link")
    def local(pos, quat):
        relative = pos - pos[:, root:root + 1]
        rotation = Rotation.from_quat(quat[:, root][:, [1, 2, 3, 0]])
        return np.einsum("tij,tbj->tbi", rotation.inv().as_matrix(), relative)
    fast_local = local(bodies.pos_w, bodies.quat_w)
    direct_local = local(reference, ref_root)
    pose_rms = float(np.sqrt(np.mean((fast_local - direct_local) ** 2)))
    root_height_rms = float(np.sqrt(np.mean((bodies.pos_w[:, root, 2]
                                             - reference[:, root, 2]) ** 2)))
    return pose_rms, root_height_rms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/v10/downloads/g1.tar.gz")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/v10/manifests/bones_seed_prefilter.csv")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data/v10/accepted")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/v10/fast_g1/pilot_results.json")
    parser.add_argument("--g1-xml", type=Path, default=G1_XML)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--matched-only", action="store_true")
    parser.add_argument("--selection-file", type=Path,
                        help="Newline-separated G1 archive paths to sample across categories")
    parser.add_argument("--save-csv-dir", type=Path,
                        help="Keep selected source CSVs for repeatable local experiments")
    parser.add_argument("--calibration", type=Path,
                        help="Optional held-out-tested G1 to V10 joint calibration JSON")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    with args.manifest.open(newline="") as stream:
        candidates = {
            row["soma_uniform_path"].replace("soma_uniform/bvh/", "g1/csv/")
            .replace(".bvh", ".csv")
            for row in csv.DictReader(stream) if row["status"] == "candidate"
        }
    if args.selection_file:
        requested = set(args.selection_file.read_text().splitlines())
        candidates.intersection_update(requested)
        if not candidates:
            raise SystemExit("Selection file contains no eligible G1 paths")
    g1 = mujoco.MjModel.from_xml_path(str(args.g1_xml))
    v10 = mujoco.MjModel.from_xml_path(str(V10_XML))
    mapping = joint_mapping(g1, v10)
    calibration = ({row["v10"]: row for row in json.loads(args.calibration.read_text())["joints"]}
                   if args.calibration else None)
    rows = []
    archive_start = time.monotonic()
    with tarfile.open(args.archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.removeprefix("./")
            if name not in candidates:
                continue
            parts = Path(name).parts
            if len(parts) < 4 or parts[:2] != ("g1", "csv") or ".." in parts:
                raise ValueError(f"Unsafe archive member: {name}")
            stem = f"{Path(name).parent.name}__{Path(name).stem}"
            direct = args.reference_dir / f"{stem}.npz"
            if args.matched_only and not direct.is_file():
                continue
            payload = tar.extractfile(member)
            if payload is None:
                continue
            csv_bytes = payload.read()
            if args.save_csv_dir:
                target = args.save_csv_dir / f"{stem}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(csv_bytes)
            started = time.monotonic()
            try:
                raw = load_g1_csv(csv_bytes)
                qpos, motion, diagnostics = map_motion(raw, g1, v10, mapping, calibration)
                bodies = forward_kinematics(v10, motion)
                # Geometry without stance labels is a preliminary lower bar.
                quality = evaluate_reference(v10, qpos, 50.0,
                                             np.zeros(len(qpos)), np.zeros(len(qpos)))
                pose_rms, height_rms = pose_error_to_direct(bodies, direct)
                diagnostics.update({
                    "ground_penetration_frames": quality["ground_penetration_frames"],
                    "minimum_ground_distance_m": quality["minimum_ground_distance_m"],
                    "self_collision_frames": quality["self_collision_frames"],
                    "minimum_self_distance_m": quality["minimum_self_distance_m"],
                    "pose_error_to_direct_rms_m": pose_rms,
                    "root_height_error_to_direct_rms_m": height_rms,
                })
                status = "ok"
            except Exception as exc:
                status, diagnostics = "error", {"error": f"{type(exc).__name__}: {exc}"}
            row = {"name": name, "status": status,
                   "elapsed_s": round(time.monotonic() - started, 3), **diagnostics}
            rows.append(row)
            print(f"{len(rows)}/{args.limit} {name}: {row}", flush=True)
            if len(rows) >= args.limit:
                break
    report = {"method": "G1 joint mapping, no IK or contact projection",
              "calibration": str(args.calibration) if args.calibration else None,
              "archive_elapsed_s": round(time.monotonic() - archive_start, 3),
              "mapping": mapping, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
