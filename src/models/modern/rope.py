"""Rotary position embeddings (RoPE, Su et al., 2021), plus the ways to stretch them.

A model trained on ``L`` tokens has only ever seen rotation angles up to ``L * theta_i``.
Run it on longer text and the slow pairs see angles they never met in training, and the
loss blows up. The three ``rope_scaling`` modes below change the rotation speeds so a
longer context maps back onto angles the model knows:

* ``linear`` (position interpolation, Chen et al. 2023): slow every pair down by ``factor``.
  Simple, but it also squeezes the fast pairs that encode local word order.
* ``ntk`` (NTK-aware scaling, bloc97 2023): raise the base ``theta`` instead, which barely
  touches the fast pairs and slows the slow ones by close to ``factor``.
* ``yarn`` (Peng et al. 2023): leave pairs that rotate many times inside the original
  context alone, interpolate the ones that rotate less than once, ramp between the two, and
  sharpen attention a little (``0.1 * ln(factor) + 1``) to undo the entropy increase.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

RopeScaling = Literal["none", "linear", "ntk", "yarn"]
ROPE_SCALINGS: tuple[str, ...] = ("none", "linear", "ntk", "yarn")


def rope_frequencies(dim: int, theta: float = 10_000.0) -> Float[Tensor, " half"]:
    """The rotation speed ``theta_i`` of each of the ``dim / 2`` pairs."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def _yarn_correction_dim(
    n_rotations: float, dim: int, theta: float, original_context: int
) -> float:
    """The pair index that makes ``n_rotations`` full turns over ``original_context`` tokens."""
    return (dim * math.log(original_context / (n_rotations * 2 * math.pi))) / (
        2 * math.log(theta)
    )


def yarn_ramp(
    dim: int,
    theta: float,
    original_context: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Float[Tensor, " half"]:
    """Per pair: 0 = keep the original speed (fast pairs), 1 = fully interpolate (slow pairs)."""
    low = float(
        max(
            math.floor(_yarn_correction_dim(beta_fast, dim, theta, original_context)), 0
        )
    )
    high = float(
        min(
            math.ceil(_yarn_correction_dim(beta_slow, dim, theta, original_context)),
            dim - 1,
        )
    )
    if high <= low:
        high = low + 1e-3  # avoid dividing by zero on tiny heads
    idx = torch.arange(dim // 2, dtype=torch.float32)
    return ((idx - low) / (high - low)).clamp(0.0, 1.0)


def yarn_attention_factor(factor: float) -> float:
    """YaRN's ``sqrt(1/t)``: cos and sin are both scaled by it, so the logits get ``1/t``."""
    return 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0


def scaled_rope_frequencies(
    dim: int,
    theta: float = 10_000.0,
    scaling: RopeScaling = "none",
    factor: float = 1.0,
    original_context: int | None = None,
) -> tuple[Float[Tensor, " half"], float]:
    """Rotation speeds after ``scaling``, and the factor to multiply ``cos``/``sin`` by."""
    if scaling == "none" or factor == 1.0:
        return rope_frequencies(dim, theta), 1.0
    if factor < 1.0:
        raise ValueError(
            f"rope_factor must be >= 1 to extend the context, got {factor}"
        )
    if scaling == "linear":
        return rope_frequencies(dim, theta) / factor, 1.0
    if scaling == "ntk":
        return rope_frequencies(dim, theta * factor ** (dim / (dim - 2))), 1.0
    if scaling == "yarn":
        if original_context is None:
            raise ValueError(
                "yarn scaling needs the original (training) context length"
            )
        base = rope_frequencies(dim, theta)
        ramp = yarn_ramp(dim, theta, original_context)
        freqs = base * (1.0 - ramp) + (base / factor) * ramp
        return freqs, yarn_attention_factor(factor)
    raise ValueError(f"unknown rope_scaling {scaling!r}; choose one of {ROPE_SCALINGS}")


def rope_cache(
    dim: int,
    max_len: int,
    theta: float = 10_000.0,
    scaling: RopeScaling = "none",
    factor: float = 1.0,
    original_context: int | None = None,
) -> tuple[Float[Tensor, "seq half"], Float[Tensor, "seq half"]]:
    """Precompute ``cos`` and ``sin`` of every angle ``position * theta_i``."""
    freqs, attn_factor = scaled_rope_frequencies(
        dim, theta, scaling, factor, original_context
    )
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return angles.cos() * attn_factor, angles.sin() * attn_factor


def apply_rope(
    x: Float[Tensor, "*batch seq dim"],
    cos: Float[Tensor, "*cbatch seq half"],
    sin: Float[Tensor, "*cbatch seq half"],
) -> Float[Tensor, "*batch seq dim"]:
    """Rotate every pair ``(x1, x2)`` of ``x`` by its angle: a 2D rotation per pair."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
