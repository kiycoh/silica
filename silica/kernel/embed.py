"""Persistent embedding store and cosine-similarity search (Phase 3).

Architecture:
  - EmbedStore  — orjson-backed index at ~/.silica/index/embeddings.json
  - build_index — incremental: skips notes already present, batches new ones
  - cosine_top_k inside EmbedStore — pure Python, no numpy
  - refresh_note — re-embed a single note (call after writes)

Embeddings substrate rule (from the plan):
  "embeddings PROPOSE, graph DISPOSES"
  This module is a CANDIDATE GENERATOR only. It is never authoritative about
  vault structure; that role belongs to graph_diff / the driver.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import orjson

_INDEX_PATH = Path.home() / ".silica" / "index" / "embeddings.json"

# Maximum characters of note content to embed (title + body prefix).
# Keeps embedding calls fast without losing most of the signal.
_MAX_CHARS = 1200


# ---------------------------------------------------------------------------
# Pure maths
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity in [−1, 1] between two vectors.

    Returns 0.0 if either vector is the zero vector (degenerate case).
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(ai * bi for ai, bi in zip(a, b))
    mag_a = sum(ai * ai for ai in a) ** 0.5
    mag_b = sum(bi * bi for bi in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# EmbedStore
# ---------------------------------------------------------------------------

class EmbedStore:
    """orjson-backed flat index mapping note paths to embedding vectors.

    File schema:
        {
          "version": 1,
          "notes": {
            "<vault-relative-path>": {
              "vec":  [float, ...],
              "name": str,          # display name / title
              "ts":   float         # unix timestamp of last embed
            }
          }
        }

    Keys are vault-relative paths WITHOUT the .md extension.
    """

    def __init__(self, path: Path | None = None):
        # Resolve lazily so tests can monkeypatch `_INDEX_PATH` after import
        self._path = path if path is not None else _INDEX_PATH
        self._notes: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = orjson.loads(self._path.read_bytes())
                self._notes = data.get("notes", {})
            except Exception:
                self._notes = {}

    def save(self) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(
            orjson.dumps(
                {"version": 1, "notes": self._notes},
                option=orjson.OPT_INDENT_2,
            )
        )
        return self._path

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def upsert(self, path: str, name: str, vec: list[float]) -> None:
        """Insert or replace a note's embedding."""
        self._notes[path] = {"vec": vec, "name": name, "ts": time.time()}

    def delete(self, path: str) -> None:
        self._notes.pop(path, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_vec(self, path: str) -> list[float] | None:
        entry = self._notes.get(path)
        return entry["vec"] if entry else None

    def has(self, path: str) -> bool:
        return path in self._notes

    def paths(self) -> list[str]:
        return list(self._notes.keys())

    def __len__(self) -> int:
        return len(self._notes)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def cosine_top_k(
        self,
        query_vec: list[float],
        k: int = 5,
        exclude: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k most similar notes as dicts with keys:
            path, name, score
        Optionally exclude a set of paths (e.g. the query note itself).
        """
        exclude = exclude or set()
        results: list[tuple[float, str]] = []
        for path, entry in self._notes.items():
            if path in exclude:
                continue
            score = _cosine(query_vec, entry["vec"])
            results.append((score, path))
        results.sort(reverse=True)
        return [
            {"path": path, "name": self._notes[path]["name"], "score": round(score, 4)}
            for score, path in results[:k]
        ]


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _note_text(title: str, body: str) -> str:
    """Combine title and body prefix for embedding."""
    combined = f"{title}\n\n{body}"
    return combined[:_MAX_CHARS]


def build_index(
    embedder: Any,
    notes: list[tuple[str, str, str]],
    *,
    store: EmbedStore | None = None,
    batch_size: int = 32,
    force: bool = False,
) -> EmbedStore:
    """Build or incrementally refresh the embedding index.

    Args:
        embedder: an object with `embed(texts: list[str]) -> list[list[float]]`
        notes: list of (path, name, body) tuples — vault-relative path (no .md),
               display name (title), and body text.
        store: existing EmbedStore to update (loads from disk if None).
        batch_size: number of texts to embed per API call.
        force: if True, re-embed ALL notes regardless of existing entries.

    Returns:
        The updated EmbedStore (already saved to disk).
    """
    if store is None:
        store = EmbedStore()

    to_embed = [
        (path, name, body)
        for path, name, body in notes
        if force or not store.has(path)
    ]

    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i : i + batch_size]
        texts = [_note_text(name, body) for _, name, body in batch]
        try:
            vecs = embedder.embed(texts)
        except Exception as exc:
            raise RuntimeError(f"Embedding call failed on batch {i//batch_size}: {exc}") from exc
        for (path, name, _), vec in zip(batch, vecs):
            store.upsert(path, name, vec)

    store.save()
    return store


def refresh_note(
    embedder: Any,
    path: str,
    name: str,
    body: str,
    *,
    store: EmbedStore | None = None,
) -> EmbedStore:
    """Re-embed a single note and persist the updated store.

    Designed to be called after a note is written to the vault (freshness hook).
    """
    if store is None:
        store = EmbedStore()
    text = _note_text(name, body)
    vecs = embedder.embed([text])
    store.upsert(path, name, vecs[0])
    store.save()
    return store
