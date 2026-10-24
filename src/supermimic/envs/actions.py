"""Joint-position action with execution noise (plan D9).

The *clean* policy action stays in ``raw_action`` (so ``last_action`` observations and
the action-rate penalty see it); Gaussian noise ``eps ~ N(0, std^2)`` is added to the
normalized action only when computing the executed set-point. ``executed_raw`` exposes
``a + eps`` for the replay buffer (the world model is fit on executed actions).
V10 optionally uses a linear-plus-cubic map and physical position limits to
retain small-signal control without sacrificing the joint's full travel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        self.noise_std = float(cfg.noise_std)
        self._cubic_scale = None
        if cfg.cubic_scale is not None:
            missing = set(self._target_names) - set(cfg.cubic_scale)
            if missing:
                raise ValueError(f"Missing cubic action scale for {sorted(missing)}")
            self._cubic_scale = torch.tensor(
                [cfg.cubic_scale[name] for name in self._target_names],
                device=self.device,
                dtype=self._raw_actions.dtype,
            ).unsqueeze(0)
        self._position_lower = self._position_upper = None
        if cfg.joint_position_limits is not None:
            missing = set(self._target_names) - set(cfg.joint_position_limits)
            if missing:
                raise ValueError(f"Missing joint-position limits for {sorted(missing)}")
            bounds = torch.tensor(
                [cfg.joint_position_limits[name] for name in self._target_names],
                device=self.device,
                dtype=self._raw_actions.dtype,
            )
            self._position_lower = bounds[:, 0].unsqueeze(0)
            self._position_upper = bounds[:, 1].unsqueeze(0)

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.noise_std > 0:
            self._executed_raw = self._raw_actions + self.noise_std * torch.randn_like(self._raw_actions)
        else:
            self._executed_raw = self._raw_actions.clone()
        self._processed_actions = self._executed_raw * self._scale + self._offset
        if self._cubic_scale is not None:
            self._processed_actions = self._processed_actions + self._executed_raw.pow(3) * self._cubic_scale
        if self._position_lower is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._position_lower, max=self._position_upper
            )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    noise_std: float = 0.1
    cubic_scale: dict[str, float] | None = None
    joint_position_limits: dict[str, tuple[float, float]] | None = None

    def build(self, env: ManagerBasedRlEnv) -> NoisyJointPositionAction:
        return NoisyJointPositionAction(self, env)
