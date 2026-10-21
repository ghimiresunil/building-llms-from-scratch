"""RMSNorm (Zhang and Sennrich, 2019), the normalization used by Llama, Qwen, Gemma and DeepSeek."""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Float[Tensor, "*batch dim"]) -> Float[Tensor, "*batch dim"]:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight
