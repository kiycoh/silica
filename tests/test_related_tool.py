"""Tests for the silica_related tool (path-native relatedness facade).

The tool is a thin wrapper over kernel.relatedness.related_notes: it resolves a
note name-or-path to the canonical vault path and fuses embeddings + co-occurrence
(+ note-edges). The facade itself is covered by test_relatedness; here we pin the
tool-level contract — name/path resolution, evidence pass-through, empty-index hint.
"""
from __future__ import annotations

import types

from silica.kernel.embed import EmbedStore
from silica.kernel.cooccurrence import CooccurStore, build_contribution


def _embed_store(tmp_path) -> EmbedStore:
    es = EmbedStore(path=tmp_path / "e.json")
    es.upsert("A", "A note", [1.0, 0.0])
    es.upsert("B", "B note", [0.9, 0.1])   # close to A
    es.upsert("C", "C note", [0.0, 1.0])   # orthogonal
    return es


def _cooc_store(tmp_path) -> CooccurStore:
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    st.upsert_note("B", build_contribution("B", "beta gamma delta"))  # shares beta, gamma
    st.upsert_note("C", build_contribution("C", "zeta eta theta"))    # disjoint
    return st


def _fake_driver(names: dict[str, str]):
    """DRIVER stub: read_note resolves a wikilink name -> path, else raises."""
    def read_note(note: str):
        if note in names:
            return types.SimpleNamespace(ref=types.SimpleNamespace(path=names[note]))
        raise KeyError(note)  # unresolved -> tool falls back to treating input as a path
    return types.SimpleNamespace(read_note=read_note)


def _wire(monkeypatch, tmp_path, *, names, embed=True, cooc=True):
    es = _embed_store(tmp_path) if embed else EmbedStore(path=tmp_path / "empty.json")
    monkeypatch.setattr("silica.kernel.embed.get_store", lambda: es)
    if cooc:
        st = _cooc_store(tmp_path)
        monkeypatch.setattr("silica.kernel.cooccurrence.get_cooccur_store", lambda **_: st)
    else:
        monkeypatch.setattr(
            "silica.kernel.cooccurrence.get_cooccur_store",
            lambda **_: CooccurStore(path=tmp_path / "empty_c.json", lang="english"),
        )
    monkeypatch.setattr("silica.driver.DRIVER", _fake_driver(names))


def test_resolves_wikilink_name_and_fuses_with_evidence(tmp_path, monkeypatch):
    from silica.tools.graph import silica_related
    _wire(monkeypatch, tmp_path, names={"Alpha": "A"})  # name -> store key "A"

    out = silica_related("Alpha", k=5)
    assert out["note"] == "Alpha"
    by_path = {r["path"]: r for r in out["results"]}
    assert "B" in by_path                       # nearest embed + strongest cooccur overlap
    ev = by_path["B"]["evidence"]
    assert any(e.startswith("embed:") for e in ev)
    assert any(e.startswith("cooccur:") for e in ev)
    assert "A" not in by_path                    # never returns the query itself


def test_excludes_query_when_resolved_path_carries_md(tmp_path, monkeypatch):
    # Real backends return ref.path WITH .md ("A.md") while the store keys are
    # .md-stripped ("A"). The tool must reduce to the store keyspace so the query
    # note is excluded from its own results (self-exclusion) and still resolves.
    from silica.tools.graph import silica_related
    _wire(monkeypatch, tmp_path, names={"Alpha": "A.md"})  # resolves to a .md path

    out = silica_related("Alpha", k=5)
    paths = {r["path"] for r in out["results"]}
    assert "A" not in paths          # query must not resurface among its own results
    assert "B" in paths


def test_accepts_raw_path_when_name_unresolvable(tmp_path, monkeypatch):
    from silica.tools.graph import silica_related
    _wire(monkeypatch, tmp_path, names={})       # read_note raises -> input used as path

    out = silica_related("A", k=5)               # "A" is a store key, not a wikilink name
    assert "B" in {r["path"] for r in out["results"]}


def test_empty_index_returns_refresh_hint(tmp_path, monkeypatch):
    from silica.tools.graph import silica_related
    _wire(monkeypatch, tmp_path, names={"Alpha": "A"}, embed=False, cooc=False)

    out = silica_related("Alpha", k=5)
    assert "error" in out and "refresh" in out["error"].lower()


def test_unresolved_note_with_no_neighbors_hints_resolution(tmp_path, monkeypatch):
    # Name doesn't resolve AND the raw string isn't a store key -> empty results.
    # The hint must point at resolution, not the index.
    from silica.tools.graph import silica_related
    _wire(monkeypatch, tmp_path, names={})       # nothing resolves

    out = silica_related("Zzz", k=5)
    assert out["results"] == []
    assert "did not resolve" in out["hint"]


def test_empty_embed_but_isolated_note_hints_embed_refresh(tmp_path, monkeypatch):
    # Embed index empty (co-occurrence only); the resolved note has no cooccur
    # neighbors -> empty results with a hint to build the embedding index.
    from silica.tools.graph import silica_related
    _wire(monkeypatch, tmp_path, names={"Gamma": "C"}, embed=False)  # C is cooccur-disjoint

    out = silica_related("Gamma", k=5)
    assert out["results"] == []
    assert "embedding index empty" in out["hint"]


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
