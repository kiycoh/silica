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


def test_substrate_includes_vault_vocabulary(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    es = EmbedStore()
    es.upsert("Concepts/Near", "Near", [1.0, 0.0])
    es.save()

    cs = CooccurStore(lang="english")  # default path isolated by conftest
    cs.upsert_note(
        "Concepts/Near", build_contribution("Near", "neural network model trains fast")
    )
    cs.save()

    monkeypatch.setattr(providers, "get_embedder", lambda *a, **k: _FakeEmbedder([1.0, 0.0]))

    out = build_substrate(_chunk(), manifest_titles=[], hub_names=["Machine Learning"])

    assert out is not None
    assert "## Vault vocabulary" in out
    assert "Hub notes: Machine Learning" in out
    # Related-notes section still present alongside vocabulary.
    assert "[[Near]]" in out


def test_vocabulary_alone_when_no_related(tmp_path, monkeypatch):
    # Empty embed index and a cooccur index that matches nothing in the chunk:
    # related-notes may be empty, but vocabulary should still surface.
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    EmbedStore().save()

    cs = CooccurStore(lang="english")
    cs.upsert_note("Other/Topic", build_contribution("Topic", "astronomy telescopes optics"))
    cs.save()

    monkeypatch.setattr(
        providers, "get_embedder",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
    )

    out = build_substrate(_chunk("completely unrelated"), manifest_titles=[])

    assert out is not None and "## Vault vocabulary" in out


def test_vocabulary_respects_token_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    EmbedStore().save()

    cs = CooccurStore(lang="english")
    cs.upsert_note("a.md", build_contribution("a", "alpha beta gamma"))
    cs.save()
    # Force an oversized stem list through the class, regardless of content.
    monkeypatch.setattr(
        CooccurStore, "top_stems",
        lambda self, n=20: [f"very-long-stem-label-{i:04d}" for i in range(200)],
    )

    monkeypatch.setattr(
        providers, "get_embedder",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
    )

    out = build_substrate(_chunk(), manifest_titles=[])

    assert out is not None and "## Vault vocabulary" in out
    stems_line = next(
        l for l in out.splitlines() if l.startswith("very-long-stem-label")
    )
    assert len(stems_line) <= 600  # hard cap from the implementation


def test_no_vocabulary_with_empty_cooccur_index(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    es = EmbedStore()
    es.upsert("Concepts/Near", "Near", [1.0, 0.0])
    es.save()
    # conftest isolates the cooccur index; do not populate it.

    monkeypatch.setattr(providers, "get_embedder", lambda *a, **k: _FakeEmbedder([1.0, 0.0]))

    out = build_substrate(_chunk(), manifest_titles=[])

    assert out is not None
    assert "## Vault vocabulary" not in out  # omitted, related-notes intact


def test_substrate_annotates_degree_and_cluster(tmp_path, monkeypatch):
    # Tier A: candidate lines carry wikilink degree (links=N, from the same
    # read as graph-far) and cluster membership (cached ctx, hub-labeled).
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    es = EmbedStore()
    es.upsert("Concepts/Near", "Near", [1.0, 0.0])
    es.save()
    monkeypatch.setattr(providers, "get_embedder", lambda *a, **k: _FakeEmbedder([1.0, 0.0]))

    import silica.kernel.graph_export as ge
    ge.save_cluster_ctx(
        [2, 1], {"Concepts/Near": {"cluster_id": 0, "hub": "Concepts/ML.md", "is_hub": False}}
    )

    from silica.driver.base import NoteRef

    class _FakeDriver:
        def links(self, ref):
            return [NoteRef(name="Other", path="Concepts/Other.md")]

        def backlinks(self, ref):
            return [NoteRef(name="In", path="Concepts/In.md")]

    import silica.driver as driver_mod
    monkeypatch.setattr(driver_mod, "DRIVER", _FakeDriver())

    out = build_substrate(_chunk(), manifest_titles=[])
    assert out is not None
    assert "links=2" in out      # 1 outgoing + 1 backlink
    assert "cluster=ML" in out   # hub label, .md stripped


def test_substrate_omits_annotations_when_cold(tmp_path, monkeypatch):
    # No ctx cache, driver links unavailable -> plain line, no links=/cluster=.
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    es = EmbedStore()
    es.upsert("Concepts/Near", "Near", [1.0, 0.0])
    es.save()
    monkeypatch.setattr(providers, "get_embedder", lambda *a, **k: _FakeEmbedder([1.0, 0.0]))

    out = build_substrate(_chunk(), manifest_titles=[])
    assert out is not None and "[[Near]]" in out
    assert "cluster=" not in out
