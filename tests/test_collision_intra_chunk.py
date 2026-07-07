"""Intra-chunk dedup (#4): near-duplicate sibling concepts within one chunk must
collapse to a single concept before COLLISION routes against the vault.

`_collapse_near_dup_concepts` takes an injected near-dup predicate over (name,
excerpt) pairs, so the union-find / drop logic is tested with a deterministic fake,
and the embedder-free cold predicate is tested on its own.
"""
from __future__ import annotations

from silica.router.states.collision import (
    _collapse_near_dup_concepts,
    _cold_intra_chunk_near_dup,
)


def _names(chunk):
    return [c["name"] for b in chunk["batches"] for c in b["concepts"]]


def test_collapse_drops_sibling_near_dups_keeps_richest_and_distinct():
    a = {"name": "Neurone artificiale", "excerpt": "x" * 40}          # richest twin
    a2 = {"name": "Neurone Artificiale (ANN)", "excerpt": "x" * 10}   # near-dup, shorter
    b = {"name": "Discesa del gradiente", "excerpt": "y" * 30}        # distinct
    chunk = {"schema_version": 1, "batches": [{"inbox_file": "s.md", "concepts": [a, a2, b]}]}

    near = lambda a, b: ("Neurone" in a[0]) and ("Neurone" in b[0])  # the two twins only
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


def test_cold_predicate_matches_title_variants():
    """title_key identity collapses title variants of the same note (the motivating
    'Neurone Artificiale' family) — no embedder, no bodies needed."""
    assert _cold_intra_chunk_near_dup(("Neurone artificiale", ""), ("Neurone Artificiale (ANN)", ""))
    assert _cold_intra_chunk_near_dup(("Neurone artificiale", "a"), ("Neurone Artificiale 1", "b"))
    assert not _cold_intra_chunk_near_dup(("SVM", "margine massimo"), ("PCA", "componenti principali"))


def test_cold_predicate_matches_near_verbatim_bodies():
    """Different titles but near-verbatim bodies (Description/Descriptor twins) match
    via MinHash even when title_key disagrees."""
    body = "One-class SVDD fits the smallest hypersphere enclosing the target class." * 2
    assert _cold_intra_chunk_near_dup(("Data Description", body), ("Data Descriptor", body))


def test_cold_predicate_collapses_chunk():
    twins = [
        {"name": "Neurone artificiale", "excerpt": "definizione completa " * 5},
        {"name": "Neurone Artificiale (ANN)", "excerpt": "breve"},
        {"name": "Discesa del gradiente", "excerpt": "ottimizzazione dei pesi"},
    ]
    chunk = {"schema_version": 1, "batches": [{"inbox_file": "s.md", "concepts": twins}]}
    out = _collapse_near_dup_concepts(chunk, is_near_dup=_cold_intra_chunk_near_dup)
    names = _names(out)
    assert names.count("Neurone artificiale") == 1
    assert "Neurone Artificiale (ANN)" not in names
    assert "Discesa del gradiente" in names
