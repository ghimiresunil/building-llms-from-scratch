"""Heuristic quality filters, after Gopher (Rae et al. 2021) and C4 (Raffel et al. 2020).

Web text is full of navigation menus, cookie banners, keyword spam, tables of numbers and
pages repeated a thousand times. None of it teaches a model language, and all of it costs
the same compute as good text. A handful of cheap rules removes most of it. Each rule below
looks at one symptom; a document is dropped if any rule fires.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

BULLETS = ("•", "-", "*", "‣", "◦", "·")


@dataclass(frozen=True)
class QualityConfig:
    min_words: int = 50  # too short to be a real document
    max_words: int = 100_000
    min_mean_word_len: float = 3.0  # shorter: token soup, tables of numbers
    max_mean_word_len: float = 10.0  # longer: URLs, hashes, minified code
    max_symbol_ratio: float = 0.1  # '#' and '...' per word: hashtags, truncated teasers
    max_ellipsis_lines: float = 0.3  # lines ending in '...': lists of article teasers
    max_bullet_lines: float = 0.9  # nothing but bullet points: menus
    min_alpha_words: float = 0.8  # words with at least one letter
    max_dup_line_frac: float = 0.3  # the same line again and again: boilerplate
    max_top_2gram_frac: float = 0.2  # one word pair dominating: keyword spam
    banned: tuple[str, ...] = ("lorem ipsum",)


def _top_ngram_fraction(words: list[str], n: int = 2) -> float:
    """Characters covered by the most common word n-gram, as a fraction of all characters."""
    if len(words) < n + 1:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    if count < 2:
        return 0.0
    total = sum(len(w) for w in words)
    return count * sum(len(w) for w in gram) / max(1, total)


DEFAULT_QUALITY = QualityConfig()


def quality_issues(text: str, cfg: QualityConfig = DEFAULT_QUALITY) -> list[str]:
    """Names of every rule ``text`` breaks (empty = keep it)."""
    words = text.split()
    n = len(words)
    if n < cfg.min_words:
        return ["too_short"]
    issues: list[str] = []
    if n > cfg.max_words:
        issues.append("too_long")
    mean_len = sum(len(w) for w in words) / n
    if not cfg.min_mean_word_len <= mean_len <= cfg.max_mean_word_len:
        issues.append("word_length")
    symbols = text.count("#") + text.count("...") + text.count("…")
    if symbols / n > cfg.max_symbol_ratio:
        issues.append("symbols")
    if sum(any(c.isalpha() for c in w) for w in words) / n < cfg.min_alpha_words:
        issues.append("non_alpha")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        ellipsis = sum(ln.endswith(("...", "…")) for ln in lines) / len(lines)
        if ellipsis > cfg.max_ellipsis_lines:
            issues.append("ellipsis_lines")
        bullets = sum(ln.startswith(BULLETS) for ln in lines) / len(lines)
        if bullets > cfg.max_bullet_lines:
            issues.append("bullet_lines")
        dup = 1.0 - len(set(lines)) / len(lines)
        if dup > cfg.max_dup_line_frac:
            issues.append("duplicate_lines")
    if _top_ngram_fraction([w.lower() for w in words]) > cfg.max_top_2gram_frac:
        issues.append("repetition")
    lower = text.lower()
    if any(b in lower for b in cfg.banned):
        issues.append("banned_phrase")
    return issues
