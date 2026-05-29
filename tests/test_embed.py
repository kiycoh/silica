"""Phase 3 tests — embedding store, cosine similarity, and semantic search tools."""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from silica.kernel.embed import (
    EmbedStore,
    _cosine,
    _note_text,
    build_index,
    refresh_note,
)


# ---------------------------------------------------------------------------
# _cosine — pure maths
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine(a, b) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine([1.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_length_mismatch_returns_zero():
    assert _cosine([1.0], [1.0, 2.0]) == 0.0


def test_cosine_normalised_vectors():
    # normalised: dot == cosine
    a = [3.0 / 5, 4.0 / 5]
    b = [4.0 / 5, 3.0 / 5]
    expected = (3 * 4 + 4 * 3) / 25
    assert _cosine(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _note_text — truncation
# ---------------------------------------------------------------------------

def test_note_text_truncates_at_1200():
    long_body = "x" * 2000
    result = _note_text("Title", long_body)
    assert len(result) <= 1200


def test_note_text_includes_title():
    result = _note_text("Neural Networks", "Deep learning intro")
    assert "Neural Networks" in result


# ---------------------------------------------------------------------------
# EmbedStore — in-memory and persistence
# ---------------------------------------------------------------------------

def test_embed_store_empty_on_missing_file(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    assert len(store) == 0
    assert store.paths() == []


def test_embed_store_upsert_and_get(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("Concepts/NeuralNet", "Neural Net", [1.0, 0.0, 0.0])
    assert store.has("Concepts/NeuralNet")
    assert store.get_vec("Concepts/NeuralNet") == [1.0, 0.0, 0.0]


def test_embed_store_roundtrip(tmp_path):
    idx = tmp_path / "embeddings.json"
    store = EmbedStore(path=idx)
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.upsert("Concepts/B", "B", [0.0, 1.0])
    store.save()

    store2 = EmbedStore(path=idx)
    assert store2.has("Concepts/A")
    assert store2.has("Concepts/B")
    assert store2.get_vec("Concepts/A") == [1.0, 0.0]


def test_embed_store_delete(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.delete("Concepts/A")
    assert not store.has("Concepts/A")


def test_embed_store_len(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("a", "A", [1.0])
    store.upsert("b", "B", [0.5])
    assert len(store) == 2


# ---------------------------------------------------------------------------
# EmbedStore.cosine_top_k
# ---------------------------------------------------------------------------

def test_cosine_top_k_ordering(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("best",   "Best",   [1.0, 0.0])
    store.upsert("middle", "Middle", [0.8, 0.6])
    store.upsert("worst",  "Worst",  [0.0, 1.0])

    results = store.cosine_top_k([1.0, 0.0], k=3)
    assert [r["path"] for r in results] == ["best", "middle", "worst"]


def test_cosine_top_k_respects_k(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    for i in range(10):
        store.upsert(f"note_{i}", f"Note {i}", [float(i), 0.0])

    results = store.cosine_top_k([1.0, 0.0], k=3)
    assert len(results) == 3


def test_cosine_top_k_exclude(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("query_note", "Query", [1.0, 0.0])
    store.upsert("other",      "Other", [0.9, 0.0])

    results = store.cosine_top_k([1.0, 0.0], k=5, exclude={"query_note"})
    paths = [r["path"] for r in results]
    assert "query_note" not in paths
    assert "other" in paths


def test_cosine_top_k_returns_score(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("a", "A", [1.0, 0.0])
    results = store.cosine_top_k([1.0, 0.0], k=1)
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# build_index — incremental and force
# ---------------------------------------------------------------------------

def _make_embedder(dim: int = 2):
    """Return a mock embedder that produces deterministic unit vectors."""
    embedder = MagicMock()
    call_count = [0]

    def fake_embed(texts):
        vecs = []
        for _ in texts:
            angle = call_count[0] * 0.1
            vecs.append([math.cos(angle), math.sin(angle)])
            call_count[0] += 1
        return vecs

    embedder.embed.side_effect = fake_embed
    return embedder


def test_build_index_embeds_new_notes(tmp_path):
    embedder = _make_embedder()
    notes = [
        ("Concepts/A", "A", "body a"),
        ("Concepts/B", "B", "body b"),
    ]
    store = build_index(embedder, notes, store=EmbedStore(tmp_path / "idx.json"))
    assert store.has("Concepts/A")
    assert store.has("Concepts/B")
    assert embedder.embed.call_count == 1  # both in one batch


def test_build_index_skips_existing(tmp_path):
    embedder = _make_embedder()
    idx = tmp_path / "idx.json"
    store = EmbedStore(idx)
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.save()

    notes = [
        ("Concepts/A", "A", "body a"),  # already indexed
        ("Concepts/B", "B", "body b"),  # new
    ]
    store2 = build_index(embedder, notes, store=EmbedStore(idx))
    assert store2.has("Concepts/A")
    assert store2.has("Concepts/B")
    # Only B should have been sent to the embedder
    embedded_texts = [
        t for call in embedder.embed.call_args_list for t in call.args[0]
    ]
    assert all("A" not in t for t in embedded_texts)


def test_build_index_force_reembeds_all(tmp_path):
    embedder = _make_embedder()
    idx = tmp_path / "idx.json"
    store = EmbedStore(idx)
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.save()

    notes = [("Concepts/A", "A", "body a")]
    build_index(embedder, notes, store=EmbedStore(idx), force=True)
    assert embedder.embed.call_count == 1


# ---------------------------------------------------------------------------
# refresh_note
# ---------------------------------------------------------------------------

def test_refresh_note_updates_vec(tmp_path):
    embedder = _make_embedder()
    idx = tmp_path / "idx.json"

    store = EmbedStore(idx)
    store.upsert("Concepts/A", "A", [0.0, 1.0])
    store.save()

    refresh_note(embedder, "Concepts/A", "A", "new body", store=EmbedStore(idx))

    store2 = EmbedStore(idx)
    # The vector should have been updated (not the old one)
    assert store2.get_vec("Concepts/A") != [0.0, 1.0]


# ---------------------------------------------------------------------------
# silica_semantic_search tool — mocked embedder
# ---------------------------------------------------------------------------

def test_silica_semantic_search_empty_index(tmp_path, monkeypatch):
    from silica.kernel.embed import EmbedStore as _ES

    # Ensure the store loaded by the tool sees an empty index
    monkeypatch.setattr("silica.kernel.embed._INDEX_PATH", tmp_path / "empty.json")

    from silica.tools.composed import silica_semantic_search
    result = silica_semantic_search(query="neural networks")
    assert "error" in result


def test_silica_semantic_search_returns_results(tmp_path, monkeypatch):
    from silica.kernel.embed import EmbedStore as _ES, _INDEX_PATH

    idx = tmp_path / "embeddings.json"
    monkeypatch.setattr("silica.kernel.embed._INDEX_PATH", idx)
    # Pre-populate the store
    store = _ES(idx)
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.upsert("Concepts/B", "B", [0.0, 1.0])
    store.save()

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[1.0, 0.0]]

    with patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        from silica.tools.composed import silica_semantic_search
        result = silica_semantic_search(query="test", k=2)

    assert "error" not in result
    assert len(result["results"]) == 2
    assert result["results"][0]["path"] == "Concepts/A"


# ---------------------------------------------------------------------------
# OpenAIEmbedder — unit test the wrapper class
# ---------------------------------------------------------------------------

def test_openai_embedder_returns_sorted_embeddings():
    """The embedder should return vectors in input order regardless of API ordering."""
    from silica.agent.providers import OpenAIEmbedder

    # Build a mock openai client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(index=1, embedding=[0.0, 1.0]),
        MagicMock(index=0, embedding=[1.0, 0.0]),
    ]
    mock_client.embeddings.create.return_value = mock_response

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.client = mock_client
    embedder.model = "test-model"

    result = embedder.embed(["text_a", "text_b"])
    assert result == [[1.0, 0.0], [0.0, 1.0]]


def test_openai_embedder_empty_input():
    from silica.agent.providers import OpenAIEmbedder

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    assert embedder.embed([]) == []
