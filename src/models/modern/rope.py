"""Rotary position embeddings (RoPE, Su et al., 2021)."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


def rope_frequencies(dim: int, theta: float = 10_000.0) -> Float[Tensor, " half"]:
    """The rotation speed ``theta_i`` of each of the ``dim / 2`` pairs."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def rope_cache(
    dim: int, max_len: int, theta: float = 10_000.0
) -> tuple[Float[Tensor, "seq half"], Float[Tensor, "seq half"]]:
    """Precompute ``cos`` and ``sin`` of every angle ``position * theta_i``."""
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, rope_frequencies(dim, theta))
    return angles.cos(), angles.sin()


def apply_rope(
    x: Float[Tensor, "*batch seq dim"], cos: Float[Tensor, "seq half"], sin: Float[Tensor, "seq half"]
) -> Float[Tensor, "*batch seq dim"]:
    """Rotate every pair ``(x1, x2)`` of ``x`` by its angle: a 2D rotation per pair."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
