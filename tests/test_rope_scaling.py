"""RoPE context extension: linear interpolation, NTK-aware scaling and YaRN."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from config.settings import ConfigError, load_stage
from src.models.factory import build_model
from src.models.modern import ModernConfig, ModernTransformer
from src.models.modern.rope import (
    rope_cache,
    rope_frequencies,
    scaled_rope_frequencies,
    yarn_attention_factor,
    yarn_ramp,
)

DIM, THETA = 64, 10_000.0


@pytest.mark.parametrize("scaling", ["none", "linear", "ntk", "yarn"])
def test_factor_one_changes_nothing(scaling: str) -> None:
    freqs, attn = scaled_rope_frequencies(DIM, THETA, scaling, 1.0, 128)  # type: ignore[arg-type]
    assert torch.allclose(freqs, rope_frequencies(DIM, THETA)) and attn == 1.0


def test_linear_maps_long_positions_onto_trained_angles() -> None:
    cos_long, sin_long = rope_cache(DIM, 512, THETA, "linear", 4.0, 128)
    cos_short, sin_short = rope_cache(DIM, 128, THETA)
    assert torch.allclose(cos_long[::4], cos_short, atol=1e-4)
    assert torch.allclose(sin_long[::4], sin_short, atol=1e-4)


def test_ntk_keeps_the_fastest_pair_and_slows_the_slowest_by_the_factor() -> None:
    base = rope_frequencies(DIM, THETA)
    ntk, _ = scaled_rope_frequencies(DIM, THETA, "ntk", 4.0)
    assert ntk[0] == base[0]
    assert torch.isclose(ntk[-1], base[-1] / 4.0, rtol=1e-4)
    assert torch.all(ntk <= base + 1e-12)


def test_yarn_interpolates_only_the_slow_pairs() -> None:
    base = rope_frequencies(DIM, THETA)
    yarn, attn = scaled_rope_frequencies(DIM, THETA, "yarn", 4.0, original_context=256)
    ramp = yarn_ramp(DIM, THETA, 256)
    assert torch.all(ramp[1:] >= ramp[:-1]) and ramp.min() == 0.0 and ramp.max() == 1.0
    assert yarn[0] == base[0]  # many turns within 256 tokens: untouched
    assert torch.isclose(yarn[-1], base[-1] / 4.0)  # under one turn: fully interpolated
    assert (
        attn == pytest.approx(0.1 * math.log(4.0) + 1.0) == yarn_attention_factor(4.0)
    )
    cos, _ = rope_cache(DIM, 8, THETA, "yarn", 4.0, 256)
    assert cos[0].max().item() == pytest.approx(
        attn
    )  # the attention factor rides on cos/sin


def test_bad_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        scaled_rope_frequencies(DIM, THETA, "linear", 0.5)
    with pytest.raises(ValueError, match="original"):
        scaled_rope_frequencies(DIM, THETA, "yarn", 2.0)
    # the shape checker (when on) or __post_init__ (when off) refuses an unknown mode
    with pytest.raises((ValueError, TypeError)):
        ModernConfig(rope_scaling="longrope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rope_factor"):
        ModernConfig(rope_factor=0.5)


def test_config_derives_the_original_context() -> None:
    cfg = ModernConfig(context_length=1024, rope_scaling="yarn", rope_factor=4.0)
    assert cfg.original_context == 256
    assert replace(cfg, rope_original_context=300).original_context == 300


@pytest.mark.parametrize("attention", ["gqa", "mla"])
def test_scaled_model_runs_and_differs_from_unscaled(attention: str) -> None:
    cfg = ModernConfig(
        vocab_size=50,
        context_length=64,
        n_embed=32,
        n_head=4,
        n_blocks=1,
        attention=attention,
        kv_latent_dim=8,
        rope_dim=4,
    )  # type: ignore[arg-type]
    torch.manual_seed(0)
    plain = ModernTransformer(cfg).eval()
    scaled = ModernTransformer(
        replace(cfg, rope_scaling="yarn", rope_factor=2.0)
    ).eval()
    scaled.load_state_dict(plain.state_dict())
    x = torch.randint(0, 50, (2, 40))
    a, b = plain(x)[0], scaled(x)[0]
    assert torch.isfinite(b).all() and not torch.allclose(a, b)
    # cached generation uses the same scaled tables
    assert scaled.generate(x[:, :5], 6, top_k=1).shape == (2, 11)


def test_settings_carry_rope_scaling_to_the_model() -> None:
    cfg = load_stage("sft", env="test").replace(
        arch="modern", rope_scaling="yarn", rope_factor=2.0
    )
    model = build_model(cfg)
    assert isinstance(model, ModernTransformer)
    assert model.config.rope_scaling == "yarn" and model.config.original_context == 32
    assert load_stage("sft", env="test").rope_scaling == "none"
    with pytest.raises(ConfigError, match="rope_scaling"):
        load_stage("sft", env="test", overrides={"rope_scaling": "bogus"})
    with pytest.raises(ConfigError, match="rope_factor"):
        load_stage("sft", env="test", overrides={"rope_factor": "0.5"})
