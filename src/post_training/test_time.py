"""Spending compute at inference time: pass@k, majority voting and best-of-N.

Sampling one answer per question measures what a model does on its first try. Sampling
``n`` answers and choosing well often gains more accuracy than a lot of extra training:

* ``pass@k`` -- is *any* of ``k`` samples right? The ceiling for every method below, and
  the number RL tries to move into pass@1.
* ``maj@k`` (self-consistency, Wang et al. 2022) -- the most common final answer wins. Needs
  no verifier, so it works where the gold answer is unknown.
* best-of-N -- score every sample (with a verifier or a reward model) and keep the best.

Rejection-sampling fine-tuning (RFT, Yuan et al. 2023; STaR, Zelikman et al. 2022) turns
this into training data: keep the samples a verifier accepts and run SFT on them. It is the
simplest baseline GRPO has to beat.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

ANSWER_TOL = 1e-4


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from ``n`` samples of which ``c`` are correct (Chen et al. 2021)."""
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if k > n:
        raise ValueError(f"k ({k}) can't exceed the number of samples n ({n})")
    if n - c < k:
        return 1.0
    # 1 - C(n - c, k) / C(n, k), as a product so it never overflows
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def _key(answer: float) -> float:
    """Round so 3.0 and 3.00001 vote together."""
    return round(answer / ANSWER_TOL) * ANSWER_TOL


def majority_vote(answers: Sequence[float | None]) -> float | None:
    """The most common parsed answer; ties go to the one seen first. ``None`` never votes."""
    keys = [_key(a) for a in answers if a is not None]
    if not keys:
        return None
    counts = Counter(keys)
    best = max(counts.values())
    return next(k for k in keys if counts[k] == best)


def best_of_n(scores: Sequence[float]) -> int:
    """Index of the highest-scoring sample (the first one on a tie)."""
    if not scores:
        raise ValueError("best_of_n needs at least one score")
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def answers_match(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and math.isclose(a, b, abs_tol=ANSWER_TOL)
