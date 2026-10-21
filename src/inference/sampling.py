"""Turning next-token logits into a choice."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


def filter_logits(
    logits: Float[Tensor, "batch vocab"],
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
) -> Float[Tensor, "batch vocab"]:
    """Apply temperature, then top-k, top-p and min-p filtering. Returns new logits."""
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumprobs > top_p
        # Keep at least the top token; shift so the token that crosses top_p stays.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))

    if min_p is not None and 0.0 < min_p < 1.0:
        probs = logits.softmax(dim=-1)
        threshold = min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < threshold, float("-inf"))

    return logits


def sample_next_token(
    logits: Float[Tensor, "batch vocab"],
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> Int[Tensor, "batch 1"]:
    """Pick the next token id for every row of ``logits``."""
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(
        filter_logits(logits.float(), temperature, top_k, top_p, min_p), dim=-1
    )
    return torch.multinomial(probs, num_samples=1, generator=generator)
