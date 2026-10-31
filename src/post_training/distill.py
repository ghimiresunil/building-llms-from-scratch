"""Knowledge distillation: train a small student to match a bigger teacher's predictions.

A one-hot label says "the next token is *cat*". The teacher's distribution says "probably
*cat*, maybe *kitten*, certainly not *carburetor*", and that extra information (Hinton's
"dark knowledge") is why a distilled student usually beats the same student trained on
labels alone. It is how most small open models are made today.

Three ways to compare the two distributions, per token:

* ``forward_kl`` -- ``KL(teacher || student)``, the classic (Hinton et al. 2015). The student
  must cover every token the teacher likes (mode covering). Softened with a temperature
  ``tau`` and multiplied by ``tau**2`` so gradients keep their size.
* ``reverse_kl`` -- ``KL(student || teacher)`` (MiniLLM, Gu et al. 2023). The student is
  punished for putting mass where the teacher has none, so it picks the teacher's main
  modes instead of smearing itself thin (mode seeking). Often better for small students.
* ``jsd`` -- the generalized Jensen-Shannon divergence of GKD (Agarwal et al. 2024), which
  moves between the two with ``beta``: near 0 it behaves like forward KL, near 1 like
  reverse KL, and it is bounded, so a confidently wrong student can't blow up the loss.

Sequence-level distillation (Kim & Rush 2016), where the student is fine-tuned on text the
teacher *generated*, needs no new loss: generate with ``scripts/rejection_sampling.py
--keep all`` and run SFT on the result.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

DistillLoss = Literal["forward_kl", "reverse_kl", "jsd"]


def _kl(
    log_p: Float[Tensor, "*batch vocab"], log_q: Float[Tensor, "*batch vocab"]
) -> Float[Tensor, "*batch"]:
    """``KL(p || q)`` per position, from log-probabilities."""
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def token_divergence(
    student_logits: Float[Tensor, "*batch vocab"],
    teacher_logits: Float[Tensor, "*batch vocab"],
    kind: DistillLoss = "forward_kl",
    temperature: float = 1.0,
    beta: float = 0.5,
) -> Float[Tensor, "*batch"]:
    """Divergence between the teacher's and the student's next-token distributions."""
    if student_logits.size(-1) != teacher_logits.size(-1):
        raise ValueError(
            f"student vocab ({student_logits.size(-1)}) != teacher vocab "
            f"({teacher_logits.size(-1)}); distillation needs the same tokenizer"
        )
    log_s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    log_t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    if kind == "forward_kl":
        div = _kl(log_t, log_s)
    elif kind == "reverse_kl":
        div = _kl(log_s, log_t)
    elif kind == "jsd":
        if beta <= 0.0:
            div = _kl(log_t, log_s)
        elif beta >= 1.0:
            div = _kl(log_s, log_t)
        else:
            # log of the mixture beta * teacher + (1 - beta) * student, computed stably
            log_m = torch.logsumexp(
                torch.stack([log_t + math.log(beta), log_s + math.log(1.0 - beta)]),
                dim=0,
            )
            div = beta * _kl(log_t, log_m) + (1.0 - beta) * _kl(log_s, log_m)
    else:
        raise ValueError(f"unknown distillation loss {kind!r}")
    return div * temperature**2  # keep gradient size independent of the temperature


def distillation_loss(
    student_logits: Float[Tensor, "batch seq vocab"],
    teacher_logits: Float[Tensor, "batch seq vocab"],
    targets: Int[Tensor, "batch seq"],
    mask: Float[Tensor, "batch seq"] | Int[Tensor, "batch seq"],
    *,
    kind: DistillLoss = "forward_kl",
    temperature: float = 2.0,
    alpha: float = 0.9,
    beta: float = 0.5,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
    """``alpha * divergence + (1 - alpha) * cross-entropy``, averaged over masked tokens.

    The logits at position ``t`` must already be aligned with ``targets[:, t]``. Returns
    ``(total, divergence, cross_entropy)`` so the parts can be logged separately.
    """
    m = mask.to(torch.float32)
    denom = m.sum().clamp(min=1.0)
    div = token_divergence(
        student_logits, teacher_logits.detach(), kind, temperature, beta
    )
    div = (div * m).sum() / denom
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)).float(),
        targets.reshape(-1).long(),
        reduction="none",
    ).view(targets.shape)
    ce = (ce * m).sum() / denom
    return alpha * div + (1.0 - alpha) * ce, div, ce


def shift_for_next_token(
    logits: Float[Tensor, "batch seq vocab"],
    tokens: Int[Tensor, "batch seq"],
    loss_mask: Tensor,
) -> tuple[Float[Tensor, "batch steps vocab"], Int[Tensor, "batch steps"], Tensor]:
    """Line up packed SFT rows: position ``t`` predicts token ``t + 1``."""
    return logits[:, :-1, :], tokens[:, 1:], loss_mask[:, 1:]
