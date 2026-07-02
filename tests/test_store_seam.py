"""Fix 3 — one loaded index behind a seam.

The cached accessors (`get_store` / `get_cooccur_store`) return a process-lifetime
singleton per resolved index path, so readers stop re-deserialising the fat JSON
blob and the write path mutates the same instance every reader sees.

Concurrency invariant (verified, load-bearing): worker threads never touch the
store — all embed/cooccur store access is single-threaded (the producer/main
thread). The singleton is therefore safe without a lock. See
``test_workers_never_touch_store``.
"""
from __future__ import annotations

import silica.kernel.embed as embed
import silica.kernel.cooccurrence as cooc


def test_get_store_returns_same_instance(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "_index_path", lambda: tmp_path / "embeddings.json")
    embed.clear()
    a = embed.get_store()
    b = embed.get_store()
    assert a is b


def test_get_store_distinct_per_vault(tmp_path, monkeypatch):
    embed.clear()
    monkeypatch.setattr(embed, "_index_path", lambda: tmp_path / "a" / "embeddings.json")
    a = embed.get_store()
    monkeypatch.setattr(embed, "_index_path", lambda: tmp_path / "b" / "embeddings.json")
    b = embed.get_store()
    assert a is not b


def test_clear_forces_fresh_instance(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "_index_path", lambda: tmp_path / "embeddings.json")
    embed.clear()
    a = embed.get_store()
    embed.clear()
    b = embed.get_store()
    assert a is not b


# --- co-occurrence twin ---------------------------------------------------

def test_get_cooccur_store_returns_same_instance(tmp_path, monkeypatch):
    monkeypatch.setattr(cooc, "_index_path", lambda: tmp_path / "cooc.json")
    cooc.clear()
    a = cooc.get_cooccur_store()
    b = cooc.get_cooccur_store()
    assert a is b


def test_get_cooccur_store_clear_forces_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(cooc, "_index_path", lambda: tmp_path / "cooc.json")
    cooc.clear()
    a = cooc.get_cooccur_store()
    cooc.clear()
    b = cooc.get_cooccur_store()
    assert a is not b


# --- consistency: the reason the seam exists -----------------------------

def test_write_mutation_visible_to_reader_without_reload(tmp_path, monkeypatch):
    """A mutation through the singleton is visible to a later reader with no
    disk round-trip — the in-memory instance is authoritative."""
    monkeypatch.setattr(embed, "_index_path", lambda: tmp_path / "embeddings.json")
    embed.clear()
    embed.get_store().upsert("notes/x", "X", [1.0, 0.0, 0.0])
    assert embed.get_store().get_vec("notes/x") == [1.0, 0.0, 0.0]


# --- concurrency invariant (load-bearing) --------------------------------

def test_workers_never_touch_store():
    """Worker threads run ONLY capabilities + the commit gate. The lockless
    singleton is safe iff none of that worker-reachable surface touches the
    embed/cooccur store. Encode that invariant: a capability that read the
    store on a worker thread would race the main thread's upsert.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "silica"
    worker_reachable = sorted((root / "capabilities").glob("*.py"))
    worker_reachable += [root / "agent" / "commit.py", root / "agent" / "subagent.py"]

    forbidden = (
        "EmbedStore", "CooccurStore", "get_store", "get_cooccur_store",
        "refresh_note", "cosine_top_k", "related_notes",
    )
    offenders = []
    for f in worker_reachable:
        src = f.read_text(encoding="utf-8")
        for sym in forbidden:
            if sym in src:
                offenders.append(f"{f.relative_to(root)}: {sym}")
    assert not offenders, (
        "Worker-reachable code touches the embed/cooccur store — the lockless "
        f"singleton would race. Offenders: {offenders}"
    )
