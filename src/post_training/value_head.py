"""Actor-critic wrapper for PPO."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.factory import LanguageModel


class TransformerWithValueHead(nn.Module):
    """Wrap a :class:`Transformer` and add a scalar value head for PPO."""

    def __init__(self, transformer: LanguageModel) -> None:
        super().__init__()
        self.transformer = transformer
        n_embed = transformer.lm_head.in_features
        hidden, out = nn.Linear(n_embed, n_embed), nn.Linear(n_embed, 1)
        # Start with values ~0 so the critic does not destabilize the policy early on.
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.value_head = nn.Sequential(hidden, nn.ReLU(), out)

    # Expose the backbone's context length so rollout's cap detection works.
    @property
    def context_length(self) -> int:
        return self.transformer.context_length

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.transformer.forward_hidden(idx)  # (B, T, n_embed)
        logits = self.transformer.lm_head(hidden)  # (B, T, vocab)
        values = self.value_head(hidden).squeeze(-1)  # (B, T)
        return logits, values

    @torch.no_grad()
    def value_only(self, idx: torch.Tensor) -> torch.Tensor:
        """Per-token values without computing the vocabulary logits."""
        hidden = self.transformer.forward_hidden(idx)
        return self.value_head(hidden).squeeze(-1)
