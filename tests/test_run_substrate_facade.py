"""Migration test: build_substrate routes through the relatedness facade.

The per-chunk '## Related Notes (candidates)' section now fuses embeddings +
co-occurrence (fresh-query facade), so it surfaces co-occurrence-related notes
the embedding leg alone would miss, and still works when the embedder is down.
"""
from __future__ import annotations

import silica.agent.providers as providers
import silica.kernel.embed as embed_mod
from silica.kernel.embed import EmbedStore
from silica.kernel.cooccurrence import CooccurStore, build_contribution
from silica.kernel.run_substrate import build_substrate


class _FakeEmbedder:
    def __init__(self, vec):
        self._vec = vec

    def embed(self, texts):
        return [list(self._vec) for _ in texts]


def _chunk(name="neural network"):
    return {"batches": [{"concepts": [{"name": name, "inbox_excerpt": ""}]}]}


def test_substrate_fuses_cooccurrence_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    es = EmbedStore()
    es.upsert("Concepts/Near", "Near", [1.0, 0.0])
    es.upsert("Concepts/Far", "Far", [0.0, 1.0])
    es.save()

    cs = CooccurStore(lang="english")  # default path isolated by conftest
    cs.upsert_note("Concepts/Cooc", build_contribution("Cooc", "neural network model"))
    cs.save()

    monkeypatch.setattr(providers, "get_embedder", lambda *a, **k: _FakeEmbedder([1.0, 0.0]))

    out = build_substrate(_chunk(), manifest_titles=[])
    assert out is not None
    assert "Near" in out   # embedding leg (cosine 1.0)
    assert "Cooc" in out   # co-occurrence-only candidate — the fusion win


def test_substrate_routes_on_cooccurrence_when_embed_index_empty(tmp_path, monkeypatch):
    # No embedding index -> build_substrate returned None before; now the
    # co-occurrence leg alone can still propose candidates.
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")  # empty
    cs = CooccurStore(lang="english")
    cs.upsert_note("Concepts/Cooc", build_contribution("Cooc", "neural network model"))
    cs.save()
    monkeypatch.setattr(providers, "get_embedder", lambda *a, **k: _FakeEmbedder([1.0, 0.0]))

    out = build_substrate(_chunk(), manifest_titles=[])
    assert out is not None
    assert "Cooc" in out


def test_substrate_excludes_manifest_titles(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    es = EmbedStore()
    es.upsert("Concepts/Near", "Near", [1.0, 0.0])
    es.save()
    monkeypatch.setattr(providers, "get_embedder", lambda *a, **k: _FakeEmbedder([1.0, 0.0]))

    out = build_substrate(_chunk(), manifest_titles=["Near"])
    # "Near" was already injected this run -> must not be re-proposed
    assert out is None or "Near" not in out
