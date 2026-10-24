"""Inspect the starting contact geometry of V10 retargeted motions.

This is a static diagnostic, not a dynamic feasibility test. In particular,
walking motions can temporarily put their COM outside a support polygon.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError


ROOT = Path(__file__).resolve().parents[2]
FOOT_GEOMS = ("left_ankle_roll_link_collision", "right_ankle_roll_link_collision")
ANKLE_JOINTS = ("left_ankle_pitch_joint", "right_ankle_pitch_joint")


def _margin(points: np.ndarray, com_xy: np.ndarray) -> float:
    """Signed Euclidean distance to the nearest convex-hull edge."""
    if len(points) < 3:
        return float("nan")
    try:
        equations = ConvexHull(points).equations
    except QhullError:
        return float("nan")
    distances = -(equations[:, :2] @ com_xy + equations[:, 2])
    distances /= np.linalg.norm(equations[:, :2], axis=1)
    return float(np.min(distances))


def inspect(path: Path, model: mujoco.MjModel) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as clip:
        names = [str(x) for x in clip["body_names"]]
        pelvis = names.index("pelvis_link")
        root_pos = np.asarray(clip["body_pos_w"][0, pelvis], dtype=np.float64)
        root_quat = np.asarray(clip["body_quat_w"][0, pelvis], dtype=np.float64)
        joints = np.asarray(clip["joint_pos"][0], dtype=np.float64)
        frames = len(clip["joint_pos"])
    data = mujoco.MjData(model)
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qpos[7:] = joints
    mujoco.mj_forward(model, data)

    foot_vertices = []
    for name in FOOT_GEOMS:
        geom = model.geom(name).id
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        local = model.mesh_vert[start : start + count]
        world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        foot_vertices.append(world)
    # Use one ground level for both feet so an airborne foot cannot enlarge
    # the apparent support polygon.
    lowest = min(float(vertices[:, 2].min()) for vertices in foot_vertices)
    near_contact = np.concatenate([
        vertices[vertices[:, 2] <= lowest + 0.005, :2]
        for vertices in foot_vertices
    ])
    reachable_sole = np.concatenate([
        vertices[vertices[:, 2] <= lowest + 0.030, :2]
        for vertices in foot_vertices
    ])
    com = data.subtree_com[0]
    ankle_lower = []
    for name in ANKLE_JOINTS:
        joint = model.joint(name).id
        qpos = data.qpos[model.jnt_qposadr[joint]]
        ankle_lower.append(float(qpos - model.jnt_range[joint, 0]))
    return {
        "clip": path.stem,
        "frames": frames,
        "com_x_m": float(com[0]),
        "com_y_m": float(com[1]),
        "near_contact_margin_m": _margin(near_contact, com[:2]),
        "reachable_sole_margin_m": _margin(reachable_sole, com[:2]),
        "left_sole_min_z_m": float(foot_vertices[0][:, 2].min()),
        "right_sole_min_z_m": float(foot_vertices[1][:, 2].min()),
        "left_ankle_pitch_above_lower_rad": ankle_lower[0],
        "right_ankle_pitch_above_lower_rad": ankle_lower[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-dir", type=Path, default=ROOT / "data/v10/accepted")
    parser.add_argument("--output", type=Path, default=ROOT / "data/v10/manifests/v10_initial_support.csv")
    args = parser.parse_args()
    files = sorted(args.accepted_dir.glob("*.npz"))
    if not files:
        raise SystemExit("No accepted V10 NPZ clips found")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"))
    rows = [inspect(path, model) for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Inspected {len(rows)} clips; wrote {args.output}")


if __name__ == "__main__":
    main()
