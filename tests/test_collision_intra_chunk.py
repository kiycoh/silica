"""Intra-chunk dedup (#4): near-duplicate sibling concepts within one chunk must
collapse to a single concept before COLLISION routes against the vault.

`_collapse_near_dup_concepts` takes an injected near-dup predicate, so the union-find
/ drop logic is tested here with a deterministic fake (no embedder, no MinHash).
"""
from __future__ import annotations

from silica.router.states.collision import _collapse_near_dup_concepts


def _names(chunk):
    return [c["name"] for b in chunk["batches"] for c in b["concepts"]]


def test_collapse_drops_sibling_near_dups_keeps_richest_and_distinct():
    a = {"name": "Neurone artificiale", "excerpt": "x" * 40}          # richest twin
    a2 = {"name": "Neurone Artificiale (ANN)", "excerpt": "x" * 10}   # near-dup, shorter
    b = {"name": "Discesa del gradiente", "excerpt": "y" * 30}        # distinct
    chunk = {"schema_version": 1, "batches": [{"inbox_file": "s.md", "concepts": [a, a2, b]}]}

    near = lambda ta, tb: ("Neurone" in ta) and ("Neurone" in tb)  # the two twins only
    out = _collapse_near_dup_concepts(chunk, is_near_dup=near)

    names = _names(out)
    assert "Discesa del gradiente" in names          # distinct survives
    assert names.count("Neurone artificiale") == 1   # richest twin kept
    assert "Neurone Artificiale (ANN)" not in names  # shorter twin dropped
    assert len(names) == 2


def test_collapse_noop_when_no_near_dups():
    chunk = {"schema_version": 1, "batches": [{"inbox_file": "s.md", "concepts": [
        {"name": "SVM", "excerpt": "margine"},
        {"name": "PCA", "excerpt": "varianza"},
    ]}]}
    out = _collapse_near_dup_concepts(chunk, is_near_dup=lambda a, b: False)
    assert len(_names(out)) == 2
