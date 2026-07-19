# SPDX-License-Identifier: AGPL-3.0-or-later
"""knn_edges: store-key↔node-id mapping, self-exclusion, undirected dedup."""
from __future__ import annotations

from silica.kernel import graph_export


class _FakeStore:
    """Minimal EmbedStore stand-in: vectors keyed in the stripped-.md keyspace."""

    def __init__(self, vecs: dict[str, list[float]]):
        self._vecs = vecs

    def __len__(self):
        return len(self._vecs)

    def get_vec(self, key):
        return self._vecs.get(key)

    def cosine_top_k(self, vec, k, exclude):
        # Rank every other note by cosine; return store-keyspace paths.
        import math

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        ranked = sorted(
            ((cos(vec, v), p) for p, v in self._vecs.items() if p not in exclude),
            reverse=True,
        )
        return [{"path": p, "name": p, "score": s} for s, p in ranked[:k]]


def _nodes(*ids):
    return [{"id": i, "type": "note"} for i in ids]


def test_knn_edges_maps_keys_and_dedups(monkeypatch):
    # Node ids carry .md; store keys are stripped — knn_edges must bridge them.
    store = _FakeStore({
        "a": [1.0, 0.0],
        "b": [0.9, 0.1],   # closest to a
        "c": [0.0, 1.0],   # orthogonal
    })
    monkeypatch.setattr(graph_export, "get_store", lambda: store, raising=False)
    # get_store is imported inside the function; patch the source module too.
    from silica.kernel import embed
    monkeypatch.setattr(embed, "get_store", lambda: store)

    edges = graph_export.knn_edges(_nodes("a.md", "b.md", "c.md"), k=2)

    # Every edge references real node ids (with .md), never a store key.
    ids = {"a.md", "b.md", "c.md"}
    for e in edges:
        assert e["from"] in ids and e["to"] in ids
        assert e["from"] != e["to"]            # no self-loops
        assert e["type"] == "SIMILAR"
        assert e["from"] < e["to"]             # canonical undirected order

    # Undirected dedup: a-b is mutual (a's top, b's top) but appears once.
    pairs = [(e["from"], e["to"]) for e in edges]
    assert pairs.count(("a.md", "b.md")) == 1
    assert len(pairs) == len(set(pairs))


def test_knn_edges_empty_store(monkeypatch):
    from silica.kernel import embed
    monkeypatch.setattr(embed, "get_store", lambda: _FakeStore({}))
    assert graph_export.knn_edges(_nodes("a.md"), k=6) == []
