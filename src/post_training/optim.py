"""Optimizer and learning-rate-schedule helpers shared across stages."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.optim import adamw_param_groups, build_optimizer, lr_at, set_lr
from src.optim.schedules import cosine_lr

__all__ = ["build_optimizer", "configure_optimizer", "cosine_lr", "lr_at", "set_lr"]


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """AdamW with weight decay applied only to >=2D parameters (matrices), not to biases / norms / 1D params."""
    return torch.optim.AdamW(
        adamw_param_groups(model, weight_decay), lr=lr, betas=betas
    )
