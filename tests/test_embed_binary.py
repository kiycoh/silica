"""Fix 2A — embedding vectors persist as a binary array, not pretty-printed JSON.

The index is machine-only derived state. Vectors dominate its size; storing them
as float32 binary (npz) instead of text floats is ~4x smaller with a no-parse
load. Metadata (name/ts/lengths) rides along in a small JSON blob inside the
archive. Each save stays self-contained and per-note → crash-safe, no behaviour
change. Old JSON indexes auto-migrate on first load (reformat, never re-embed).
"""
from __future__ import annotations

import orjson
import pytest

from silica.kernel.embed import EmbedStore


def test_binary_roundtrip_preserves_vectors(tmp_path):
    idx = tmp_path / "embeddings.json"
    s = EmbedStore(path=idx)
    s.upsert("a", "A", [0.1, 0.2, 0.3], title_vec=[0.4, 0.5, 0.6])
    s.upsert("b", "B", [0.7, 0.8, 0.9])
    s.save()

    s2 = EmbedStore(path=idx)
    assert s2.paths() == ["a", "b"]
    assert s2.get_vec("a") == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    assert s2.get_title_vec("a") == pytest.approx([0.4, 0.5, 0.6], abs=1e-6)
    assert s2.get_vec("b") == pytest.approx([0.7, 0.8, 0.9], abs=1e-6)
    assert s2.get_title_vec("b") is None  # not all notes carry a title vec
    assert s2._notes["a"]["name"] == "A"


def test_save_writes_binary_not_json(tmp_path):
    idx = tmp_path / "embeddings.json"
    s = EmbedStore(path=idx)
    s.upsert("a", "A", [0.1] * 64)
    s.save()
    head = idx.read_bytes()[:2]
    assert head == b"PK"  # npz (zip) magic — not the legacy JSON '{'


def test_old_json_format_auto_migrates(tmp_path):
    idx = tmp_path / "embeddings.json"
    idx.write_bytes(orjson.dumps({
        "version": 1,
        "notes": {"x": {"vec": [1.0, 2.0], "name": "X", "ts": 0.0,
                        "title_vec": [3.0, 4.0]}},
    }))
    s = EmbedStore(path=idx)
    assert s.get_vec("x") == pytest.approx([1.0, 2.0])
    assert s.get_title_vec("x") == pytest.approx([3.0, 4.0])
    # On save the file reformats to binary.
    s.save()
    assert idx.read_bytes()[:2] == b"PK"


def test_empty_store_roundtrips(tmp_path):
    idx = tmp_path / "embeddings.json"
    EmbedStore(path=idx).save()
    assert len(EmbedStore(path=idx)) == 0


def test_each_save_is_self_contained(tmp_path):
    """Crash-safety: every save writes a complete, fully-loadable index."""
    idx = tmp_path / "embeddings.json"
    s = EmbedStore(path=idx)
    s.upsert("a", "A", [1.0, 0.0])
    s.save()
    s.upsert("b", "B", [0.0, 1.0])
    s.save()
    reloaded = EmbedStore(path=idx)
    assert reloaded.has("a") and reloaded.has("b")
