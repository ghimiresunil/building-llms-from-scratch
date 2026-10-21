"""Reward-model training objective: the Bradley-Terry pairwise loss used by InstructGPT."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bradley_terry_loss(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    """Mean ``-log sigmoid(chosen - rejected)`` over a batch of preference pairs."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def preference_accuracy(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    """Fraction of pairs where the model scores the chosen response higher."""
    return (chosen_rewards > rejected_rewards).float().mean()


def reward_margin(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    """Mean reward gap (chosen - rejected); a useful training diagnostic."""
    return (chosen_rewards - rejected_rewards).mean()
