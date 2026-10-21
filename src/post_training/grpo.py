"""GRPO, the DeepSeek-R1 RL algorithm, plus switches for Dr. GRPO, DAPO and GSPO."""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from src.post_training.utils import masked_mean, masked_mean_per_row

AdvantageNorm = Literal["std", "none"]
LossAggregation = Literal[
    "token-mean", "seq-mean-token-mean", "seq-mean-token-sum-norm"
]
RatioLevel = Literal["token", "sequence"]


def group_advantages(
    rewards: Float[Tensor, " n"],
    group_size: int,
    eps: float = 1e-4,
    scale: AdvantageNorm = "std",
) -> Float[Tensor, " n"]:
    """Group-relative advantage: ``(r - group_mean) / (group_std + eps)``."""
    r = rewards.view(-1, group_size)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale == "std":
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.reshape(-1)


def k3_kl(
    new_logp: Float[Tensor, "*shape"], ref_logp: Float[Tensor, "*shape"]
) -> Float[Tensor, "*shape"]:
    """Per-token unbiased, non-negative KL estimator (Schulman's k3) for KL(policy||ref)."""
    diff = ref_logp - new_logp
    return torch.exp(diff) - diff - 1.0


def _mean_over_answers(
    per_answer: Float[Tensor, " batch"], mask: Bool[Tensor, "batch steps"]
) -> Float[Tensor, ""]:
    """Average over the answers that have at least one trained token."""
    active = mask.any(dim=-1).to(per_answer.dtype)
    return (per_answer * active).sum() / active.sum().clamp(min=1.0)


def aggregate_token_loss(
    per_token: Float[Tensor, "batch steps"],
    mask: Bool[Tensor, "batch steps"],
    mode: LossAggregation = "token-mean",
    max_len: int | None = None,
) -> Float[Tensor, ""]:
    """Reduce a per-token loss to one number."""
    m = mask.to(per_token.dtype)
    if mode == "token-mean":
        return masked_mean(per_token, m)
    if mode == "seq-mean-token-mean":
        return _mean_over_answers(masked_mean_per_row(per_token, m), mask)
    if mode == "seq-mean-token-sum-norm":
        return _mean_over_answers(
            (per_token * m).sum(dim=-1) / float(max_len or per_token.size(-1)), mask
        )
    raise ValueError(f"unknown loss aggregation {mode!r}")


def grpo_loss(
    new_logp: Float[Tensor, "batch steps"],
    old_logp: Float[Tensor, "batch steps"],
    ref_logp: Float[Tensor, "batch steps"],
    advantages: Float[Tensor, " batch"],
    resp_mask: Bool[Tensor, "batch steps"],
    clip: float = 0.2,
    kl_coef: float = 0.04,
    *,
    clip_high: float | None = None,
    loss_agg: LossAggregation = "token-mean",
    ratio_level: RatioLevel = "token",
    max_len: int | None = None,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Clipped surrogate plus KL penalty, with the Dr. GRPO, DAPO and GSPO switches."""
    high = clip if clip_high is None else clip_high
    adv = advantages[:, None]
    kl = k3_kl(new_logp, ref_logp)

    if ratio_level == "sequence":
        # GSPO: s_i = exp(mean_t log(pi/pi_old)), one ratio for the whole answer.
        log_ratio = masked_mean_per_row(new_logp - old_logp, resp_mask)
        ratio = torch.exp(log_ratio)
        surrogate = torch.min(
            ratio * advantages, torch.clamp(ratio, 1.0 - clip, 1.0 + high) * advantages
        )
        loss = -_mean_over_answers(
            surrogate - kl_coef * masked_mean_per_row(kl, resp_mask), resp_mask
        )
        clipped = _mean_over_answers(
            ((ratio < 1.0 - clip) | (ratio > 1.0 + high)).float(), resp_mask
        )
    else:
        ratio = torch.exp(new_logp - old_logp)
        surrogate = torch.min(
            ratio * adv, torch.clamp(ratio, 1.0 - clip, 1.0 + high) * adv
        )
        loss = -aggregate_token_loss(
            surrogate - kl_coef * kl, resp_mask, loss_agg, max_len
        )
        clipped = masked_mean(
            ((ratio < 1.0 - clip) | (ratio > 1.0 + high)).float(), resp_mask
        )

    stats = {"kl": masked_mean(kl, resp_mask).item(), "clipfrac": clipped.item()}
    return loss, stats
