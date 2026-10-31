"""Quality filters and MinHash deduplication (src/curation)."""

from __future__ import annotations

import random

import pytest

from src.curation import (
    Deduplicator,
    MinHasher,
    QualityConfig,
    jaccard,
    quality_issues,
    shingles,
)
from src.curation.minhash import estimated_jaccard

WORDS = (
    "the river carried small boats past the old mill while children watched from the bridge "
    "and their parents talked about the harvest the weather and the price of bread in town"
).split()


def _doc(seed: int, n: int = 120) -> str:
    rng = random.Random(seed)
    sentences = []
    for _ in range(n // 12):
        sentences.append(
            " ".join(rng.choice(WORDS) for _ in range(12)).capitalize() + "."
        )
    return "\n".join(
        " ".join(sentences[i : i + 2]) for i in range(0, len(sentences), 2)
    )


def test_shingles_ignore_case_and_punctuation() -> None:
    assert shingles("The cat, the HAT!", n=2) == {"the cat", "cat the", "the hat"}
    assert shingles("one two", n=5) == {"one two"}
    assert shingles("", n=5) == set()


def test_minhash_estimates_jaccard() -> None:
    a = {f"w{i}" for i in range(300)}
    b = {f"w{i}" for i in range(100, 400)}  # true Jaccard = 200 / 400 = 0.5
    hasher = MinHasher(num_perm=256, seed=1)
    est = estimated_jaccard(hasher.signature(a), hasher.signature(b))
    assert jaccard(a, b) == 0.5
    assert est == pytest.approx(0.5, abs=0.1)
    assert estimated_jaccard(hasher.signature(a), hasher.signature(a)) == 1.0


def test_deduplicator_catches_exact_and_near_copies() -> None:
    dedup = Deduplicator(threshold=0.7)
    original = _doc(0, 240)
    near = original.replace("bridge", "bridges", 1) + " A short extra line."
    assert not dedup.is_duplicate(original)
    assert dedup.is_duplicate(
        original.upper()
    )  # same words: exact duplicate after normalizing
    assert dedup.is_duplicate(near)
    assert not dedup.is_duplicate(_doc(1, 240))
    assert (dedup.n_exact, dedup.n_near) == (1, 1)
    assert 0.5 < dedup.candidate_threshold() < 0.9
    with pytest.raises(ValueError):
        Deduplicator(num_perm=100, bands=16)


def test_quality_filters_keep_prose_and_flag_junk() -> None:
    assert quality_issues(_doc(3)) == []
    assert quality_issues("too short") == ["too_short"]
    menu = "\n".join(f"• Home • About • Contact {i}" for i in range(30))
    assert "bullet_lines" in quality_issues(menu)
    spam = "buy now " * 80
    assert "repetition" in quality_issues(spam)
    boiler = "\n".join(["All rights reserved by the company."] * 20 + [_doc(4, 60)])
    assert "duplicate_lines" in quality_issues(boiler)
    numbers = " ".join(str(i * 7919) for i in range(100))
    assert {"non_alpha"} <= set(quality_issues(numbers))
    teasers = "\n".join(
        f"Read the story about the river and the mill number {i} ..." for i in range(20)
    )
    assert "ellipsis_lines" in quality_issues(teasers)
    assert "banned_phrase" in quality_issues(_doc(5) + " lorem ipsum dolor")
    assert quality_issues(_doc(6, 40), QualityConfig(min_words=10)) == []
