"""Watch a trained tracking policy replay reference clips in an interactive viewer.

  uv run python scripts/play.py --ckpt logs/e5_bones_seed/ckpt_latest.pt --clips dance1_subject1
  uv run python scripts/play.py --ckpt logs/e3_lafan1_40_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 4
  uv run python scripts/play.py --ckpt logs/v10_seed_fast_full_2x5080_20261024/ckpt_0030000.pt --library data/v10/demo/v10_demo.npy --device cpu --viewer viser

The grey/blue translucent robot is the reference pose; the solid robot is the policy.
Deterministic evaluation settings (no noise, no DR); every reset starts the clip from its first frame.
Native MuJoCo viewer when a display is available, otherwise viser (browser, http://localhost:8080).
Keys (native): space pause, right-arrow single step, +/- speed, backspace reset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


class _DictPolicy:
    """Adapts policy(obs_policy, obs_goal) to the viewer's policy(obs) protocol."""

    def __init__(self, policy):
        self.policy = policy
        self.enabled = True
        self.noise_std = 0.0

    @torch.no_grad()
    def __call__(self, obs):
        action = self.policy(obs["policy"], obs["goal"])
        if not self.enabled:
            return torch.zeros_like(action)
        if self.noise_std:
            action = action + torch.randn_like(action) * self.noise_std
        return action


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    sampling: Literal["start", "uniform"] = "start",
    library: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    demo_ui: bool = False,
    demo_port: int = 8081,
) -> None:
    if demo_ui and (library is None or viewer == "native"):
        raise ValueError("--demo-ui requires --library and --viewer viser")
    if library is not None and clips is not None:
        raise ValueError("--clips applies to --npz-dir; use the browser picker for --library")
    files = [str(library)] if library is not None else [str(f) for f in discover_npz(npz_dir, names=clips)]
    if not files:
        raise ValueError("No motions found")
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    labels = None
    demo_motions = None
    first_demo_index = 0
    if library is not None:
        meta = json.loads(library.with_suffix(".json").read_text())
        if meta.get("robot") and meta["robot"] != cfg.get("robot", "g1"):
            raise ValueError("Motion library robot does not match checkpoint robot")
        if "demo_motions" in meta:
            labels = tuple(f"{item['category']} · {item['label']}" for item in meta["demo_motions"])
            demo_motions = meta["demo_motions"]
        elif demo_ui:
            # Reuse the small curated catalogue as shortcuts, while the viewer
            # runs against the complete training library without copying it.
            curated_path = Path("data/v10/demo/v10_demo.json")
            curated = (json.loads(curated_path.read_text()).get("demo_motions", [])
                       if curated_path.exists() else [])
            featured = {item["source_name"]: item for item in curated}
            demo_motions = []
            for index, (name, frames) in enumerate(zip(meta["names"], meta["lengths"])):
                item = featured.get(name)
                demo_motions.append({"category": item["category"] if item else "All motions",
                                     "label": item["label"] if item else name.split("__")[1].replace("_", " ") if "__" in name else name,
                                     "source_name": name, "frames": int(frames),
                                     "featured": item is not None})
                if item is not None and first_demo_index == 0:
                    first_demo_index = index
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        robot=cfg.get("robot", "g1"),
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode=sampling,
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        root_pos_weight=cfg["env"].get("root_pos_weight", 0.5),
        root_ori_weight=cfg["env"].get("root_ori_weight", 0.5),
        compact_library=library is not None,
        terminate_on_tracking_error=not demo_ui,
        extra_command_kwargs=({"demo_labels": labels} if labels is not None else
                              {"browser_picker_only": True} if demo_ui else {}),
        debug_vis=ghost,
    )
    env = RslRlVecEnvWrapper(make_env(args, device=device))
    if library is not None:
        command = env.unwrapped.command_manager.get_term("motion")
        command._viewer_clip_index = first_demo_index
        env.reset()
    pol = _DictPolicy(policy)
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    if demo_ui:
        viewer = "viser"
    print(f"[INFO] {command.library.num_clips if library is not None else len(files)} clip(s), "
          f"{num_envs} env(s), viewer={viewer}")
    try:
        if viewer == "native":
            NativeMujocoViewer(env, pol).run()
        else:
            import viser

            server = viser.ViserServer(host=host, port=port,
                                      label="V10 SuperMimic Demo" if library is not None else "SuperMimic")
            demo_server = None
            try:
                if demo_ui:
                    # mjlab's default negative elevation starts Viser below the floor.
                    # A low, above-ground angle keeps the whole robot visible on reload.
                    env.unwrapped.cfg.viewer.distance = 3.0
                    env.unwrapped.cfg.viewer.elevation = 15.0
                browser_viewer = ViserPlayViewer(env, pol, viser_server=server)
                if library is not None:
                    browser_viewer.pause()  # avoid spending CPU while nobody is watching
                    browser_viewer.request_single_step()  # initialize camera and robot pose
                if demo_ui:
                    from supermimic.eval.demo_ui import start_demo_ui

                    demo_server = start_demo_ui(host, demo_port, port, browser_viewer,
                                                command, pol, demo_motions)
                    print(f"[INFO] Official-style V10 demo: http://{host}:{demo_port}", flush=True)
                browser_viewer.run()
            finally:
                if demo_server is not None:
                    demo_server.shutdown()
                    demo_server.server_close()
                server.stop()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
