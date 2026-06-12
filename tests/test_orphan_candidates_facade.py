"""Migration test: Coordinator._orphan_candidates routes through the relatedness
facade, fusing embeddings + co-occurrence instead of embeddings alone.

This is a pure candidate-generation site (no cosine thresholding), so swapping
cosine_top_k for the RRF facade is a clean, zero-regression upgrade: it gains
the co-occurrence leg and keeps working when the embedder index is empty.
"""
from __future__ import annotations

import pytest

from silica.router.coordinator import Coordinator
from silica.kernel.embed import EmbedStore
from silica.kernel.cooccurrence import CooccurStore, build_contribution


def _bare_coordinator() -> Coordinator:
    # bypass the heavy __init__ (FSM construction); _orphan_candidates uses no self state
    return object.__new__(Coordinator)


@pytest.fixture
def isolated_indexes(tmp_path, monkeypatch):
    """Redirect BOTH on-disk indexes to tmp (cooccur is already isolated by the
    autouse conftest fixture; embed needs explicit redirection here)."""
    import silica.kernel.embed as embed_mod
    import silica.kernel.cooccurrence as cooc_mod
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    return embed_mod, cooc_mod


def test_orphan_candidates_fuses_both_legs(isolated_indexes):
    es = EmbedStore()
    es.upsert("orphan", "Orphan", [1.0, 0.0])
    es.upsert("near", "Near", [0.95, 0.05])   # embed-close to orphan
    es.upsert("far", "Far", [0.0, 1.0])
    es.save()

    cs = CooccurStore(lang="english")          # default path -> isolated tmp
    cs.upsert_note("orphan", build_contribution("Orphan", "neural network model"))
    cs.upsert_note("near", build_contribution("Near", "neural network training"))
    cs.save()

    out = _bare_coordinator()._orphan_candidates("Orphan", k=3)
    paths = [c["path"] for c in out]
    assert "near" in paths
    assert "orphan" not in paths                # never the query itself
    assert all("name" in c and "path" in c for c in out)


def test_orphan_candidates_routes_on_cooccurrence_when_embed_empty(isolated_indexes):
    # Embed index left empty -> embed leg abstains -> co-occurrence alone routes.
    cs = CooccurStore(lang="english")
    cs.upsert_note("orphan", build_contribution("Orphan", "neural network model"))
    cs.upsert_note("near", build_contribution("Near", "neural network training"))
    cs.upsert_note("far", build_contribution("Far", "sailing boat harbour"))
    cs.save()

    out = _bare_coordinator()._orphan_candidates("orphan", k=3)
    paths = [c["path"] for c in out]
    assert "near" in paths                      # shared concepts -> related
    assert "far" not in paths                   # disjoint concepts


def test_orphan_candidates_empty_when_no_signal(isolated_indexes):
    # Nothing indexed anywhere -> both legs abstain -> [] (best-effort, no raise)
    out = _bare_coordinator()._orphan_candidates("orphan", k=3)
    assert out == []
