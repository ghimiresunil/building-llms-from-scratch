"""Verifiable reward functions (RLVR)."""

from __future__ import annotations

import math

from src.post_training.rewards.parsing import extract_answer, has_well_formed_answer

CORRECT_BONUS = 1.0
FORMAT_BONUS = 0.2
REWARD_CLIP = 1.2
FLOAT_TOL = 1e-4


def _answers_match(pred: float | None, gold: float | None) -> bool:
    if pred is None or gold is None:
        return False
    return math.isclose(pred, gold, rel_tol=0.0, abs_tol=FLOAT_TOL)


def is_correct(text: str, gold: float | None) -> bool:
    """Whether the model's parsed answer matches the gold answer (used for accuracy eval)."""
    return _answers_match(extract_answer(text), gold)


def reward_gsm8k(text: str, gold: float | None) -> float:
    """Reward for a GSM8K response: correctness (dominant) + small format bonus, clipped."""
    r = 0.0
    if _answers_match(extract_answer(text), gold):
        r += CORRECT_BONUS
    if has_well_formed_answer(text):
        r += FORMAT_BONUS
    return min(r, REWARD_CLIP)


# Arithmetic warm-up uses the same logic (gold is a single number).
reward_arithmetic = reward_gsm8k


def reward_format(text: str) -> float:
    """Pure format reward (no gold): 1.0 if exactly one well-formed answer block else 0."""
    return 1.0 if has_well_formed_answer(text) else 0.0
