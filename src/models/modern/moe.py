"""Mixture of Experts (MoE), the layer behind Mixtral, DeepSeek-V3, Qwen 3 MoE and GPT-OSS."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.models.modern.mlp import SwiGLU


class MoE(nn.Module):
    def __init__(self, n_embed: int, hidden: int, n_experts: int, top_k: int, n_shared: int = 0) -> None:
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(n_embed, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(n_embed, hidden) for _ in range(n_experts)])
        self.shared = SwiGLU(n_embed, hidden * n_shared) if n_shared > 0 else None
        # Filled by every forward pass, for the loss and for diagnostics.
        self.aux_loss: Tensor | None = None
        self.tokens_per_expert: Tensor | None = None

    def forward(self, x: Float[Tensor, "batch seq embed"]) -> Float[Tensor, "batch seq embed"]:
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        probs = F.softmax(self.router(flat).float(), dim=-1)  # (N, E)
        weights, chosen = probs.topk(self.top_k, dim=-1)  # (N, k)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token_idx, slot = torch.where(chosen == e)  # which tokens picked expert e, and in which slot
            if token_idx.numel() == 0:
                continue
            y = expert(flat[token_idx]) * weights[token_idx, slot, None]
            out.index_add_(0, token_idx, y)
        if self.shared is not None:
            out = out + self.shared(flat)

        counts = F.one_hot(chosen, self.n_experts).sum(dim=(0, 1)).float()  # routing slots per expert
        fraction = counts / counts.sum()
        self.aux_loss = self.n_experts * (fraction * probs.mean(dim=0)).sum()
        self.tokens_per_expert = counts.detach()
        return out.view(B, T, C)
