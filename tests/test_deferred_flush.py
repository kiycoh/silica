"""Scaling Fix A — defer the embed/cooccur flush to once per run.

The write path upserts into the shared in-memory singleton during the run but no
longer rewrites the whole index file per note (1.17s/note at 10k). A single flush
at end-of-run persists everything. A startup reconcile (set-diff vault vs index)
re-embeds anything a hard crash stranded, so deferring never permanently desyncs.
"""
from __future__ import annotations

import orjson

from silica.kernel.embed import EmbedStore, refresh_note
from silica.kernel import cooccurrence as cooc


class _Emb:
    def embed(self, texts):
        return [[float(len(t) % 5), 1.0, 0.0] for t in texts]


def test_embed_refresh_note_defers_save(tmp_path):
    idx = tmp_path / "embeddings.json"
    store = EmbedStore(path=idx)
    refresh_note(_Emb(), "a", "A", "body one", store=store, save=False)
    assert store.has("a")        # upserted in memory (readers see it)
    assert not idx.exists()      # but NOT persisted yet

    store.save()                 # the single end-of-run flush
    reloaded = EmbedStore(path=idx)
    assert reloaded.has("a")


def test_embed_refresh_note_saves_by_default(tmp_path):
    idx = tmp_path / "embeddings.json"
    store = EmbedStore(path=idx)
    refresh_note(_Emb(), "a", "A", "body", store=store)  # default save=True
    assert idx.exists()


def test_cooccur_build_index_defers_save(tmp_path):
    idx = tmp_path / "cooccurrence.json"
    store = cooc.CooccurStore(path=idx, lang="english")
    notes = [("a", "A", "neural network training gradient descent loss function")]
    cooc.build_index(notes, store=store, lang="english", force=True, save=False)
    assert store.paths()         # upserted in memory
    assert not idx.exists()      # not persisted

    store.save()
    reloaded = cooc.CooccurStore(path=idx, lang="english")
    assert reloaded.paths()


# --- end-of-run flush (Fix A) ---------------------------------------------

def test_flush_indexes_persists_both_singletons(tmp_path, monkeypatch):
    """_flush_indexes saves the deferred embed + cooccur upserts once."""
    import silica.kernel.embed as embed
    import silica.kernel.cooccurrence as cooc_mod
    from silica.router.orchestrator import InjectorFSM

    ei = tmp_path / "embeddings.json"
    ci = tmp_path / "cooccurrence.json"
    monkeypatch.setattr(embed, "_index_path", lambda: ei)
    monkeypatch.setattr(cooc_mod, "_index_path", lambda: ci)
    embed.clear(); cooc_mod.clear()

    embed.get_store().upsert("a", "A", [1.0, 0.0])  # deferred (no save)
    cooc_mod.build_index([("a", "A", "neural network training")],
                         lang="english", force=True, save=False)
    assert not ei.exists() and not ci.exists()

    fsm = object.__new__(InjectorFSM)
    fsm.context = {"_embed_dirty": True, "_cooccur_dirty": True}
    fsm._flush_indexes()  # the end-of-run flush
    assert ei.exists() and ci.exists()
    assert embed.EmbedStore(path=ei).has("a")


def test_flush_skips_when_not_dirty(tmp_path, monkeypatch):
    """No writes this run (or embedder down) → no index rewrite at all."""
    import silica.kernel.embed as embed
    from silica.router.orchestrator import InjectorFSM

    ei = tmp_path / "embeddings.json"
    monkeypatch.setattr(embed, "_index_path", lambda: ei)
    embed.clear()
    embed.get_store().upsert("a", "A", [1.0, 0.0])  # in memory, but run not "dirty"

    fsm = object.__new__(InjectorFSM)
    fsm.context = {}  # nothing deferred this run
    fsm._flush_indexes()
    assert not ei.exists()  # untouched


# --- startup reconcile (Fix A safety net) ---------------------------------

def _seed_index(embed, ei, entries):
    embed.clear()
    s = embed.get_store()
    for p in entries:
        s.upsert(p, p.upper(), [1.0, 0.0])
    s.save()


def test_reconcile_embeds_only_missing(tmp_path, monkeypatch):
    import silica.kernel.embed as embed
    import silica.router.orchestrator as orch
    from silica.driver.base import NoteRef
    from types import SimpleNamespace

    ei = tmp_path / "embeddings.json"
    monkeypatch.setattr(embed, "_index_path", lambda: ei)
    _seed_index(embed, ei, ["a", "b"])  # index has a, b (non-empty)

    monkeypatch.setattr(orch, "DRIVER", SimpleNamespace(
        list_files=lambda folder="": [NoteRef("A", "a.md"), NoteRef("B", "b.md"), NoteRef("C", "c.md")],
        read_note=lambda p: SimpleNamespace(content="a body about c"),
    ))
    monkeypatch.setattr("silica.agent.providers.get_embedder", lambda cfg: _Emb())

    assert orch._reconcile_embed_index() == 1   # only c was missing
    assert embed.get_store().has("c")


def test_reconcile_skips_cold_index(tmp_path, monkeypatch):
    """An empty index is a cold build — reconcile must not implicitly embed all."""
    import silica.kernel.embed as embed
    import silica.router.orchestrator as orch
    from types import SimpleNamespace

    monkeypatch.setattr(embed, "_index_path", lambda: tmp_path / "embeddings.json")
    embed.clear()

    def _boom(*a, **k):
        raise AssertionError("reconcile must not enumerate/embed on a cold index")

    monkeypatch.setattr(orch, "DRIVER", SimpleNamespace(list_files=_boom))
    assert orch._reconcile_embed_index() == 0


def test_reconcile_skips_when_drift_exceeds_cap(tmp_path, monkeypatch):
    import silica.kernel.embed as embed
    import silica.router.orchestrator as orch
    from silica.driver.base import NoteRef
    from types import SimpleNamespace

    ei = tmp_path / "embeddings.json"
    monkeypatch.setattr(embed, "_index_path", lambda: ei)
    _seed_index(embed, ei, ["a"])
    monkeypatch.setattr(orch, "_RECONCILE_CAP", 2)

    many = [NoteRef("A", "a.md")] + [NoteRef(f"N{i}", f"N{i}.md") for i in range(5)]
    monkeypatch.setattr(orch, "DRIVER", SimpleNamespace(
        list_files=lambda folder="": many,
        read_note=lambda p: SimpleNamespace(content="body"),
    ))
    monkeypatch.setattr("silica.agent.providers.get_embedder",
                        lambda cfg: (_ for _ in ()).throw(AssertionError("must not embed past cap")))

    assert orch._reconcile_embed_index() == 0  # 5 missing > cap 2 → skip
