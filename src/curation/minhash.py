"""Exact and near-duplicate removal with MinHash and locality-sensitive hashing.

The web repeats itself: the same news story on fifty sites, the same license on every
repository, the same template with a different city name. Training on duplicates wastes
compute and teaches the model to memorize (Lee et al. 2022, "Deduplicating Training Data
Makes Language Models Better").

Exact duplicates are easy: hash the normalized text. *Near* duplicates need a similarity,
and the usual one is the Jaccard similarity of the documents' sets of word 5-grams
("shingles"). Comparing every pair is quadratic, so we use two tricks:

1. **MinHash** (Broder 1997). Hash every shingle with ``k`` different hash functions and
   keep the minimum of each. The chance that two documents share a minimum equals their
   Jaccard similarity, so the fraction of equal entries in two signatures estimates it.
2. **LSH banding.** Cut the signature into ``b`` bands of ``r`` rows. Documents that match on
   *all* rows of *any* band become candidates. Similar pairs almost always collide in some
   band, dissimilar pairs almost never do, and the threshold sits near ``(1/b) ** (1/r)``.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict

import numpy as np

_WORD_RE = re.compile(r"\w+")
_PRIME = np.uint64(4_294_967_311)  # a prime just above 2**32
_MAX = np.uint64(2**32 - 1)


def normalize(text: str) -> str:
    """Lowercase and keep only words, so formatting differences don't hide a duplicate."""
    return " ".join(_WORD_RE.findall(text.lower()))


def shingles(text: str, n: int = 5) -> set[str]:
    """The set of word ``n``-grams of ``text`` (one shingle for very short texts)."""
    words = normalize(text).split()
    if len(words) <= n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """``|a & b| / |a | b|``: 1 for identical sets, 0 for disjoint ones."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    """``num_perm`` hash functions ``(a * x + b) mod p``, applied to 32-bit shingle hashes."""

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, 2**31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 2**31, size=num_perm, dtype=np.uint64)

    def signature(self, items: set[str]) -> np.ndarray:
        """The MinHash signature: per hash function, the smallest hash of any item."""
        if not items:
            return np.full(self.num_perm, _MAX, dtype=np.uint64)
        x = np.fromiter(
            (zlib.crc32(s.encode("utf-8")) for s in items),
            dtype=np.uint64,
            count=len(items),
        )
        hashed = (x[:, None] * self.a[None, :] + self.b[None, :]) % _PRIME
        return np.minimum(hashed, _MAX).min(axis=0)


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of equal signature entries: an unbiased estimate of the Jaccard similarity."""
    return float((sig_a == sig_b).mean())


class Deduplicator:
    """Streaming filter: ``is_duplicate(text)`` is True if a kept document is (nearly) the same."""

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        bands: int = 16,
        shingle_size: int = 5,
        seed: int = 0,
    ) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold = threshold
        self.bands, self.rows = bands, num_perm // bands
        self.shingle_size = shingle_size
        self.hasher = MinHasher(num_perm, seed)
        self._exact: set[bytes] = set()
        self._buckets: list[dict[bytes, list[int]]] = [
            defaultdict(list) for _ in range(bands)
        ]
        self._signatures: list[np.ndarray] = []
        self.n_exact = 0
        self.n_near = 0

    def candidate_threshold(self) -> float:
        """Similarity at which a pair becomes a candidate with probability about 1/2."""
        return (1.0 / self.bands) ** (1.0 / self.rows)

    def is_duplicate(self, text: str) -> bool:
        """Check ``text`` against everything kept so far; remember it if it is new."""
        digest = hashlib.sha1(normalize(text).encode("utf-8")).digest()
        if digest in self._exact:
            self.n_exact += 1
            return True
        sig = self.hasher.signature(shingles(text, self.shingle_size))
        keys = [
            sig[i * self.rows : (i + 1) * self.rows].tobytes()
            for i in range(self.bands)
        ]
        seen: set[int] = set()
        for band, key in enumerate(keys):
            for doc in self._buckets[band].get(key, ()):
                if doc in seen:
                    continue
                seen.add(doc)
                if estimated_jaccard(sig, self._signatures[doc]) >= self.threshold:
                    self.n_near += 1
                    return True
        doc_id = len(self._signatures)
        self._signatures.append(sig)
        self._exact.add(digest)
        for band, key in enumerate(keys):
            self._buckets[band][key].append(doc_id)
        return False
