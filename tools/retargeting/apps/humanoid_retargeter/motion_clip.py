"""In-memory kinematic clip used between GEM-X and the PyRoki solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MotionClip:
    """Joint hierarchy with local positions and XYZW local rotations."""

    names: list[str]
    parents: np.ndarray
    offsets_m: np.ndarray
    channels: list[list[str]]
    local_positions_m: np.ndarray
    local_rotations: np.ndarray
    fps: float

    @property
    def frame_count(self) -> int:
        return self.local_positions_m.shape[0]

    def global_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Return global joint positions and XYZW rotations for every frame."""
        frames, joints = self.local_positions_m.shape[:2]
        positions = np.empty((frames, joints, 3), dtype=np.float64)
        rotations = np.empty((frames, joints, 4), dtype=np.float64)
        for joint_index, parent_index in enumerate(self.parents):
            local_rotation = Rotation.from_quat(self.local_rotations[:, joint_index])
            if parent_index < 0:
                positions[:, joint_index] = self.local_positions_m[:, joint_index]
                rotations[:, joint_index] = local_rotation.as_quat()
                continue
            parent_rotation = Rotation.from_quat(rotations[:, parent_index])
            positions[:, joint_index] = positions[:, parent_index] + parent_rotation.apply(
                self.local_positions_m[:, joint_index]
            )
            rotations[:, joint_index] = (parent_rotation * local_rotation).as_quat()
        return positions, rotations
