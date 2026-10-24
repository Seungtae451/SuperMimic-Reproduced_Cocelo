"""Keep the critic's configured learning-rate ratio during cosine decay."""

import torch

from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg


def test_set_lr_preserves_critic_to_actor_ratio():
    actor = torch.nn.Parameter(torch.zeros(()))
    critic = torch.nn.Parameter(torch.zeros(()))
    trainer = object.__new__(PolicyTrainer)
    trainer.cfg = PolicyTrainerCfg(lr=5e-5, critic_lr=2e-4)
    trainer.opt = torch.optim.Adam([actor], lr=trainer.cfg.lr)
    trainer.critic_opt = torch.optim.Adam([critic], lr=trainer.cfg.critic_lr)

    trainer.set_lr(1e-5)

    assert trainer.opt.param_groups[0]["lr"] == 1e-5
    assert trainer.critic_opt.param_groups[0]["lr"] == 4e-5
