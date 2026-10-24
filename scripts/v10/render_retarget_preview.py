"""Render a side-by-side BONES-SEED SOMA skeleton and kinematic V10 reference.

This visualizes retargeted poses. It does not simulate motor control, balance,
or a trained SuperMimic policy. Run with MUJOCO_GL=egl on a headless machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/retargeting"))

from apps.external.formats import load_bvh  # noqa: E402
from apps.humanoid_retargeter.named_reference import qpos_from_named_archive  # noqa: E402
from apps.humanoid_retargeter.pyroki_retargeter import _resample_clip  # noqa: E402
from apps.humanoid_retargeter.shared import _mujoco_from_source_positions  # noqa: E402


MODEL_PATH = ROOT / "assets/robots/humanoid_v10/mjcf/humanoid_v10.xml"
BONES = (
    ("Hips", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Chest"),
    ("Chest", "Neck1"), ("Neck1", "Head"),
    ("Chest", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Chest", "RightShoulder"), ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftLeg"), ("LeftLeg", "LeftShin"),
    ("LeftShin", "LeftFoot"), ("LeftFoot", "LeftToeEnd"),
    ("Hips", "RightLeg"), ("RightLeg", "RightShin"),
    ("RightShin", "RightFoot"), ("RightFoot", "RightToeEnd"),
)


def make_visual_model(width: int, height: int) -> mujoco.MjModel:
    tree = ET.parse(MODEL_PATH)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise ValueError("V10 model has no asset section")
    for mesh in asset.findall("mesh"):
        mesh.set("file", str((MODEL_PATH.parent / mesh.get("file")).resolve()))
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient", "rgb1": "0.92 0.96 1",
        "rgb2": "0.98 0.99 1", "width": "512", "height": "512",
    })
    ET.SubElement(asset, "texture", {
        "name": "preview_grid", "type": "2d", "builtin": "checker",
        "rgb1": "0.9 0.93 0.96", "rgb2": "0.84 0.88 0.92",
        "width": "512", "height": "512",
    })
    ET.SubElement(asset, "material", {
        "name": "preview_floor", "texture": "preview_grid", "texrepeat": "6 6",
        "texuniform": "true", "emission": "0.45", "rgba": "1 1 1 1",
    })
    ground = root.find(".//geom[@name='ground']")
    if ground is None:
        raise ValueError("V10 model has no ground geom")
    ground.set("material", "preview_floor")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(width), "offheight": str(height)})
    ET.SubElement(visual, "headlight", {
        "ambient": "0.24 0.24 0.24", "diffuse": "0.55 0.55 0.55",
        "specular": "0.02 0.02 0.02",
    })
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def add_skeleton(scene: mujoco.MjvScene, points: np.ndarray, names: list[str]) -> None:
    joints = {name: points[i] for i, name in enumerate(names)}
    cyan = np.array([0.03, 0.72, 0.92, 1], dtype=np.float32)
    yellow = np.array([1.0, 0.72, 0.28, 1], dtype=np.float32)
    identity = np.eye(3).ravel()
    for parent, child in BONES:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.full(3, 0.018), np.zeros(3), identity, cyan)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.018,
                            joints[parent], joints[child])
        scene.ngeom += 1
    for name in sorted({joint for pair in BONES for joint in pair}):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.full(3, 0.030), joints[name], identity, yellow)
        scene.ngeom += 1


def motion_category(source: Path, manifest: Path) -> str:
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if source.as_posix().endswith("/" + row["soma_uniform_path"]):
                return row["category"]
    return "BONES-SEED motion"


def overlay(image: np.ndarray, category: str, index: int, total: int,
            frame: int, frames: int, fps: float, width: int, height: int) -> np.ndarray:
    picture = Image.fromarray(image)
    ink = ImageDraw.Draw(picture, "RGBA")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    ink.rounded_rectangle((28, 24, width - 28, 100), radius=15,
                          fill=(245, 250, 253, 225))
    ink.text((65, 43), "SOMA BVH SOURCE", font=font, fill=(8, 45, 62, 255))
    ink.text((width // 2 + 95, 43), "HUMANOID V10 TARGET", font=font,
             fill=(8, 45, 62, 255))
    ink.rounded_rectangle((28, height - 77, width - 28, height - 18),
                          radius=13, fill=(19, 36, 48, 220))
    ink.text((54, height - 67), f"{index}/{total}  {category}  |  {frame / fps:.1f} s",
             font=small, fill=(255, 255, 255, 255))
    ink.text((width - 520, height - 67), "Kinematic preview  ·  no trained policy",
             font=small, fill=(222, 236, 243, 255))
    progress = 0 if frames <= 1 else frame / (frames - 1)
    ink.rounded_rectangle((54, height - 35, width - 54, height - 29),
                          radius=3, fill=(105, 137, 154, 255))
    ink.rounded_rectangle((54, height - 35,
                           54 + max(7, int((width - 108) * progress)), height - 29),
                          radius=3, fill=(15, 199, 221, 255))
    return np.asarray(picture)


def render_clip(renderer: mujoco.Renderer, model: mujoco.MjModel,
                npz_path: Path, manifest: Path, output_fps: int, max_frames: int | None,
                index: int, total: int, width: int, height: int):
    artifact = npz_path.resolve()
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    if metadata.get("robot") != "humanoid_v10":
        raise ValueError(f"Not a V10 reference: {artifact}")
    source = Path(metadata["source_motion"])
    with np.load(artifact, allow_pickle=False) as archive:
        qpos = qpos_from_named_archive(model, archive)
        reference_fps = float(archive["fps"].item())
    if not np.isclose(reference_fps, 50.0) or 50 % output_fps:
        raise ValueError("Input must be 50 Hz and output fps must divide 50")
    clip = _resample_clip(load_bvh(source, length_scale=0.01), 50.0)
    source_positions = _mujoco_from_source_positions(clip.global_transforms()[0])
    if len(source_positions) != len(qpos):
        raise ValueError(f"Source and retargeted frame counts differ: {source}")
    names = clip.names
    lateral = (source_positions[0, names.index("RightLeg")]
               - source_positions[0, names.index("LeftLeg")])
    target_right = Rotation.from_quat(qpos[0, 3:7][[1, 2, 3, 0]]).apply([0, -1, 0])
    yaw = (np.arctan2(target_right[1], target_right[0])
           - np.arctan2(lateral[1], lateral[0]))
    align = Rotation.from_euler("z", yaw)
    category = motion_category(source, manifest)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.azimuth = 220
    camera.elevation = -13
    root_id = model.body("pelvis_link").id
    frame_indices = list(range(0, len(qpos), 50 // output_fps))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    for output_index, frame in enumerate(frame_indices):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        root_position = data.xpos[root_id].copy()
        relative = (source_positions[frame]
                    - source_positions[frame, names.index("Hips")]) * 0.70
        human = align.apply(relative) + root_position + np.array([0, -1.15, 0])
        camera.lookat[:] = root_position + np.array([0, -0.575, -0.08])
        renderer.update_scene(data, camera=camera)
        add_skeleton(renderer.scene, human, names)
        yield overlay(renderer.render(), category, index, total, frame,
                      len(qpos), reference_fps, width, height)
        if output_index % 50 == 0:
            print(f"{npz_path.name}: {output_index + 1}/{len(frame_indices)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="*", type=Path,
                        help="Accepted V10 NPZ paths; defaults to all accepted clips")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/v10/videos/bones_seed_v10_retarget_preview.mp4")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/v10/manifests/bones_seed_prefilter.csv")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames-per-clip", type=int,
                        help="Preview only: limit encoded frames per clip")
    args = parser.parse_args()
    if args.fps < 1 or 50 % args.fps or args.width % 2 or args.height % 2:
        parser.error("fps must divide 50 and video dimensions must be even")
    clips = args.clips or sorted((ROOT / "data/v10/accepted").glob("*.npz"))
    if not clips:
        parser.error("No accepted V10 clips were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = make_visual_model(args.width, args.height)
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    writer = imageio_ffmpeg.write_frames(str(args.output), (args.width, args.height),
                                         fps=args.fps, codec="libx264", quality=7,
                                         pix_fmt_in="rgb24", pix_fmt_out="yuv420p")
    writer.send(None)
    first_frame = None
    try:
        for index, clip_path in enumerate(clips, start=1):
            for frame in render_clip(renderer, model, clip_path, args.manifest,
                                     args.fps, args.max_frames_per_clip, index,
                                     len(clips), args.width, args.height):
                if first_frame is None:
                    first_frame = frame
                writer.send(frame.tobytes())
    finally:
        writer.close()
        renderer.close()
    if first_frame is not None:
        Image.fromarray(first_frame).save(args.output.with_suffix(".png"))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
