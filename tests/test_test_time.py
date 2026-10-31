"""pass@k, majority voting and best-of-N (src/post_training/test_time.py)."""

from __future__ import annotations

import math
from itertools import combinations

import pytest

from src.post_training.test_time import (
    answers_match,
    best_of_n,
    majority_vote,
    pass_at_k,
)


def _pass_at_k_by_counting(n: int, c: int, k: int) -> float:
    """Literally enumerate every k-subset of the n samples."""
    samples = [True] * c + [False] * (n - c)
    subsets = list(combinations(samples, k))
    return sum(any(s) for s in subsets) / len(subsets)


@pytest.mark.parametrize(
    "n,c,k", [(10, 3, 1), (10, 3, 4), (8, 0, 3), (8, 8, 2), (6, 2, 5), (5, 1, 5)]
)
def test_pass_at_k_matches_enumeration(n: int, c: int, k: int) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(_pass_at_k_by_counting(n, c, k))


def test_pass_at_one_is_the_plain_accuracy() -> None:
    assert pass_at_k(16, 4, 1) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        pass_at_k(4, 5, 1)
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 5)


def test_majority_vote() -> None:
    assert majority_vote([3.0, 5.0, 3.00001, None, 5.0, 3.0]) == pytest.approx(3.0)
    assert majority_vote([7.0, 2.0, 2.0, 7.0]) == pytest.approx(
        7.0
    )  # tie: first seen wins
    assert majority_vote([None, None]) is None
    assert majority_vote([]) is None


def test_best_of_n_and_answer_matching() -> None:
    assert best_of_n([0.1, 0.9, 0.9, -1.0]) == 1
    with pytest.raises(ValueError):
        best_of_n([])
    assert answers_match(18.0, 18.00001) and not answers_match(18.0, 19.0)
    assert not answers_match(None, 18.0) and not answers_match(math.nan, math.nan)
