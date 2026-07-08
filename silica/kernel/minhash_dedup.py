"""Embedder-free near-duplicate detection — the STABLE dedup leg.

MinHash over character k-shingles, deterministic and dependency-free (stdlib
hashlib only). The twin of the co-occurrence relatedness leg: when the embedding
index is empty or the embedder is down, this still catches near-duplicate
concepts so COLLISION does not silently let duplicates land in the vault.

MinHash idea ported from Graphify (github.com/safishamsi/graphify, MIT,
Copyright (c) 2026 Safi Shamsi). Their `_minhash.py` is a vectorised
datasketch-compatible drop-in with band-LSH for codebase-scale all-pairs dedup;
this is the pure-stdlib slice Silica needs for one-query-vs-vault lookups.

# ponytail: O(n) scan over the corpus. Corpus signatures are memoized per run
# via near_duplicates(sig_cache=...); what stays deferred is LSH banding to skip
# the full scan — add it only past ~10^4 notes or if this lands on a hot path.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from random import Random

_MERSENNE = (1 << 61) - 1   # prime modulus for the (a·h + b) hash family
_MASK32 = (1 << 32) - 1
_K = 3                      # char-shingle width (short labels survive; mirrors Graphify)

Signature = tuple[int, ...]


@lru_cache(maxsize=None)
def _coeffs(num_perm: int) -> tuple[tuple[int, int], ...]:
    """Fixed-seed (a, b) permutation coefficients — same across calls/processes."""
    rng = Random(1)
    return tuple((rng.randint(1, _MERSENNE), rng.randint(0, _MERSENNE)) for _ in range(num_perm))


def _shingles(text: str, k: int = _K) -> set[str]:
    """Character k-grams of the normalised text."""
    s = " ".join(text.lower().split())
    if not s:
        return set()
    if len(s) < k:
        return {s}
    return {s[i : i + k] for i in range(len(s) - k + 1)}


def minhash_signature(text: str, *, num_perm: int = 64) -> Signature:
    """Return the MinHash signature of text. Empty text → empty signature."""
    shingles = _shingles(text)
    if not shingles:
        return ()
    hashed = [
        int.from_bytes(hashlib.sha1(sh.encode("utf-8")).digest()[:4], "little")
        for sh in shingles
    ]
    coeffs = _coeffs(num_perm)
    return tuple(
        min(((a * h + b) % _MERSENNE) & _MASK32 for h in hashed)
        for a, b in coeffs
    )


def estimate_jaccard(sig_a: Signature, sig_b: Signature) -> float:
    """Estimated Jaccard similarity = fraction of matching signature slots.

    An empty signature (empty text) is similar to nothing, so → 0.0.
    """
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


def near_duplicates(
    query: str,
    corpus: dict[str, str],
    *,
    threshold: float = 0.7,
    num_perm: int = 64,
    sig_cache: dict[str, tuple[int, Signature]] | None = None,
) -> list[tuple[str, float]]:
    """Keys in corpus whose text is a near-duplicate of query, best first.

    Args:
        query:     the incoming concept text (name + excerpt).
        corpus:    {key: text} of existing notes to compare against.
        threshold: minimum estimated Jaccard to count as a near-duplicate.
        sig_cache: optional {key: (content_fingerprint, signature)} memo so the
                   caller can reuse corpus signatures across calls. Entries whose
                   fingerprint no longer matches the text are recomputed, so a
                   changed note is never matched against its stale signature.

    Returns [(key, score)] sorted by score descending; empty query → [].
    """
    q_sig = minhash_signature(query, num_perm=num_perm)
    if not q_sig:
        return []

    def corpus_sig(key: str, text: str) -> Signature:
        if sig_cache is None:
            return minhash_signature(text, num_perm=num_perm)
        fp = hash(text)
        hit = sig_cache.get(key)
        if hit is not None and hit[0] == fp:
            return hit[1]
        sig = minhash_signature(text, num_perm=num_perm)
        sig_cache[key] = (fp, sig)
        return sig

    hits = [
        (key, score)
        for key, text in corpus.items()
        if (score := estimate_jaccard(q_sig, corpus_sig(key, text))) >= threshold
    ]
    hits.sort(key=lambda kv: kv[1], reverse=True)
    return hits
