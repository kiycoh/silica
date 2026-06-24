"""Embedder-free near-duplicate detection (MinHash) — the STABLE dedup leg.

Twin of the co-occurrence leg: deterministic, no network, no model. Used when
the embedding index is empty or the embedder is down so COLLISION still catches
near-duplicate concepts instead of silently letting them land in the vault.
"""
from __future__ import annotations

from silica.kernel.minhash_dedup import (
    estimate_jaccard,
    minhash_signature,
    near_duplicates,
)


def test_signature_is_deterministic_across_calls() -> None:
    # Fixed-seed permutations: the same text must hash to the same signature
    # every call (and every process), or a persisted index would be useless.
    a = minhash_signature("the quick brown fox jumps over the lazy dog")
    b = minhash_signature("the quick brown fox jumps over the lazy dog")
    assert a == b


def test_identical_text_estimates_full_jaccard() -> None:
    sig = minhash_signature("transformers use self-attention over tokens")
    assert estimate_jaccard(sig, sig) == 1.0


def test_disjoint_text_estimates_near_zero_jaccard() -> None:
    a = minhash_signature("photosynthesis converts light into chemical energy")
    b = minhash_signature("the federal reserve sets short-term interest rates")
    assert estimate_jaccard(a, b) < 0.1


def test_near_duplicate_is_returned_above_threshold() -> None:
    corpus = {
        "notes/attention.md": "Transformers use self-attention over input tokens.",
        "notes/cooking.md": "Sourdough needs a long cold ferment for flavour.",
    }
    hits = near_duplicates(
        "Transformers apply self-attention across the input tokens.",
        corpus,
        threshold=0.3,
    )
    assert hits, "expected the near-duplicate note to be flagged"
    assert hits[0][0] == "notes/attention.md"


def test_unrelated_query_returns_nothing_above_threshold() -> None:
    corpus = {
        "notes/cooking.md": "Sourdough needs a long cold ferment for flavour.",
    }
    hits = near_duplicates(
        "Quantum error correction protects logical qubits from decoherence.",
        corpus,
        threshold=0.3,
    )
    assert hits == []


def test_empty_text_is_never_a_duplicate() -> None:
    assert near_duplicates("", {"a.md": "anything"}, threshold=0.0) == []
    assert estimate_jaccard(minhash_signature(""), minhash_signature("x")) == 0.0


# ── COLLISION wiring: the embedder-free pass that runs when the embed index is
#    empty or the embedder is down (the hole this leg closes).

def test_embedder_free_pass_defers_only_the_near_dup_concept() -> None:
    from silica.router.states.collision import _embedder_free_near_dups

    chunk = {
        "batches": [
            {
                "inbox_file": "inbox/src.md",
                "concepts": [
                    {
                        "name": "Self-attention",
                        "excerpt": "Transformers apply self-attention across the input tokens.",
                    },
                    {"name": "Sourdough", "excerpt": "Long cold ferment for flavour."},
                ],
            }
        ]
    }
    corpus = {"notes/attention.md": "Transformers use self-attention over input tokens."}

    out = _embedder_free_near_dups(chunk, corpus, threshold=0.3)

    assert [d["concept"]["name"] for d in out] == ["Self-attention"]
    assert out[0]["top_match"]["path"] == "notes/attention.md"
    assert out[0]["inbox_file"] == "inbox/src.md"


def test_embedder_free_pass_returns_nothing_when_no_near_dup() -> None:
    from silica.router.states.collision import _embedder_free_near_dups

    chunk = {
        "batches": [
            {"inbox_file": "inbox/src.md", "concepts": [{"name": "Sourdough", "excerpt": "bread"}]}
        ]
    }
    corpus = {"notes/attention.md": "Transformers use self-attention over input tokens."}

    assert _embedder_free_near_dups(chunk, corpus, threshold=0.3) == []
