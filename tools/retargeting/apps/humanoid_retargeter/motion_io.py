"""Shared NumPy motion archive format for GEM-X and Kimodo previews."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .motion_clip import MotionClip


def save_visualization_motion(clip: MotionClip, path: str | Path) -> Path:
    """Save an articulated human clip for synchronized reference playback."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        names=np.asarray(clip.names),
        parents=clip.parents,
        offsets_m=clip.offsets_m,
        local_positions_m=clip.local_positions_m,
        local_rotations_xyzw=clip.local_rotations,
        fps=np.asarray(clip.fps, dtype=np.float64),
    )
    return path


def load_visualization_motion(path: str | Path) -> MotionClip:
    """Load a shared human clip saved beside a reference export."""
    with np.load(path, allow_pickle=False) as archive:
        return MotionClip(
            [str(name) for name in archive["names"]],
            np.asarray(archive["parents"], dtype=np.int32),
            np.asarray(archive["offsets_m"], dtype=np.float64),
            [tuple() for _ in archive["names"]],
            np.asarray(archive["local_positions_m"], dtype=np.float64),
            np.asarray(archive["local_rotations_xyzw"], dtype=np.float64),
            float(np.asarray(archive["fps"]).item()),
        )
