"""Knowledge-distillation losses (src/post_training/distill.py) and the distill stage settings."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from config.settings import ConfigError, load_stage
from src.post_training.distill import (
    distillation_loss,
    shift_for_next_token,
    token_divergence,
)

KINDS = ["forward_kl", "reverse_kl", "jsd"]


def _logits(seed: int, shape=(2, 5, 11)) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 2


@pytest.mark.parametrize("kind", KINDS)
def test_divergence_is_zero_only_for_identical_distributions(kind: str) -> None:
    a, b = _logits(0), _logits(1)
    assert torch.allclose(token_divergence(a, a, kind), torch.zeros(2, 5), atol=1e-5)  # type: ignore[arg-type]
    assert (token_divergence(a, b, kind) > 0).all()  # type: ignore[arg-type]


def test_forward_and_reverse_kl_match_torch() -> None:
    s, t = _logits(0), _logits(1)
    log_s, log_t = s.log_softmax(-1), t.log_softmax(-1)
    fwd = F.kl_div(log_s, log_t, log_target=True, reduction="none").sum(
        -1
    )  # KL(t || s)
    rev = F.kl_div(log_t, log_s, log_target=True, reduction="none").sum(
        -1
    )  # KL(s || t)
    assert torch.allclose(token_divergence(s, t, "forward_kl"), fwd, atol=1e-5)
    assert torch.allclose(token_divergence(s, t, "reverse_kl"), rev, atol=1e-5)
    assert not torch.allclose(fwd, rev)  # KL is not symmetric


def test_jsd_is_bounded_and_interpolates() -> None:
    s, t = _logits(0) * 10, _logits(1) * 10  # very confident, very different
    jsd = token_divergence(s, t, "jsd", beta=0.5)
    assert (jsd <= math.log(2) + 1e-5).all()
    assert torch.allclose(
        token_divergence(s, t, "jsd", beta=0.0), token_divergence(s, t, "forward_kl")
    )
    assert torch.allclose(
        token_divergence(s, t, "jsd", beta=1.0), token_divergence(s, t, "reverse_kl")
    )


def test_temperature_scales_by_tau_squared() -> None:
    s, t = _logits(0), _logits(1)
    tau = 3.0
    direct = token_divergence(s / tau, t / tau, "forward_kl") * tau**2
    assert torch.allclose(
        token_divergence(s, t, "forward_kl", temperature=tau), direct, atol=1e-5
    )


def test_vocab_mismatch_is_an_error() -> None:
    # the shape checker (when on) or the function itself (when off) refuses it
    with pytest.raises((ValueError, TypeError)):
        token_divergence(_logits(0), _logits(1, (2, 5, 12)))


def test_alpha_blends_divergence_and_masked_cross_entropy() -> None:
    s = _logits(0).requires_grad_(True)
    t = _logits(1).requires_grad_(True)
    targets = torch.randint(0, 11, (2, 5))
    mask = torch.tensor([[1, 1, 0, 0, 1], [0, 1, 1, 1, 0]])
    total, div, ce = distillation_loss(s, t, targets, mask, alpha=0.0)
    keep = mask.bool()
    expected_ce = F.cross_entropy(s[keep], targets[keep])
    assert torch.allclose(total, expected_ce) and torch.allclose(ce, expected_ce)
    total, div, ce = distillation_loss(s, t, targets, mask, alpha=0.7, temperature=2.0)
    assert torch.allclose(total, 0.7 * div + 0.3 * ce)
    total.backward()
    assert s.grad is not None and t.grad is None  # the teacher is never trained


def test_shift_lines_up_next_token_targets() -> None:
    logits, tokens, mask = (
        _logits(0, (1, 4, 11)),
        torch.tensor([[5, 6, 7, 8]]),
        torch.tensor([[0, 0, 1, 1]]),
    )
    lg, tg, mk = shift_for_next_token(logits, tokens, mask)
    assert (
        lg.shape == (1, 3, 11)
        and tg.tolist() == [[6, 7, 8]]
        and mk.tolist() == [[0, 1, 1]]
    )


def test_distill_stage_settings() -> None:
    cfg = load_stage("distill", env="smoke")
    assert cfg.distill_loss == "forward_kl" and cfg.max_steps == 10
    assert (
        load_stage("distill", overrides={"student_ckpt": "none"}).student_ckpt is None
    )
    with pytest.raises(ConfigError, match="distill_alpha"):
        load_stage("distill", overrides={"distill_alpha": "1.5"})
    with pytest.raises(ConfigError, match="distill_loss"):
        load_stage("distill", overrides={"distill_loss": "mse"})
    with pytest.raises(ConfigError, match="distill_temperature"):
        load_stage("distill", overrides={"distill_temperature": "0"})
