"""Reward model: a scalar reward head on top of a :class:`Transformer` backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.checkpoint import load_checkpoint, model_state_from_checkpoint
from src.models.factory import LanguageModel
from src.post_training.utils import build_model_from_config, gather_last


def load_reward_model(cfg, ckpt_path: str, device: str) -> RewardModel:
    """Rebuild a trained reward model (backbone + head) from its checkpoint."""
    backbone = build_model_from_config(cfg)
    rm = RewardModel(backbone)
    state = model_state_from_checkpoint(load_checkpoint(ckpt_path, map_location="cpu"))
    rm.load_state_dict(state, strict=True)
    rm.to(device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    return rm


class RewardModel(nn.Module):
    """Wrap a :class:`Transformer` and add a scalar reward head (no ``lm_head`` used)."""

    def __init__(self, transformer: LanguageModel) -> None:
        super().__init__()
        self.transformer = transformer
        n_embed = transformer.lm_head.in_features
        self.reward_head = nn.Linear(n_embed, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)  # start near-zero rewards

    @property
    def context_length(self) -> int:
        return self.transformer.context_length

    def token_rewards(self, idx: torch.Tensor) -> torch.Tensor:
        """Per-token scalar reward (B, T) -- useful for diagnostics / dense shaping."""
        hidden = self.transformer.forward_hidden(idx)
        return self.reward_head(hidden).squeeze(-1)

    def forward(
        self, idx: torch.Tensor, seq_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Scalar reward per sequence (B,)."""
        rewards = self.token_rewards(idx)  # (B, T)
        if seq_lengths is None:
            return rewards[:, -1]
        return gather_last(rewards, seq_lengths)
