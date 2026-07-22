# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

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

import heapq
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from silica.kernel.paths import atomic_write_bytes

_LEGACY_INDEX_PATH = Path.home() / ".silica" / "index" / "embeddings.json"


def _index_path() -> Path:
    # Function, not constant: resolves per current vault; tests monkeypatch it.
    from silica.kernel import paths

    return paths.index_file("embeddings")

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
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(va @ vb) / denom


def centroid(vectors: list[list[float]]) -> list[float]:
    """Component-wise mean of a list of vectors. Returns [] if empty or ragged."""
    if not vectors:
        return []
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        return []
    return np.mean(np.asarray(vectors, dtype=np.float64), axis=0).tolist()


# Theme vectors are requested twice per inbox file (RECON rerank + SALIENCE
# gate) with an identical cleaned body — cache by content so the second call
# is free. ponytail: crude clear-at-cap bound, fine for per-run lifetimes.
_theme_cache: dict[tuple[str, str, int], list[float]] = {}
_THEME_CACHE_MAX = 64


def document_theme_vector(embedder: Any, body: str, *, segment_chars: int = _MAX_CHARS) -> list[float]:
    """Thematic centroid of a document: embed body segments then average.

    Robust on long notes. Returns [] if embedder fails or body is empty.
    Cached per (model, body-hash, segment_chars) — see _theme_cache above.
    """
    if not body.strip():
        return []
    import hashlib
    key = (
        getattr(embedder, "model", ""),
        hashlib.sha1(body.encode("utf-8", "ignore")).hexdigest(),
        segment_chars,
    )
    cached = _theme_cache.get(key)
    if cached is not None:
        return cached
    segs = [body[i:i + segment_chars] for i in range(0, len(body), segment_chars)] or [body]
    try:
        vecs = embedder.embed(segs)
    except Exception:
        return []
    vec = centroid(vecs)
    if vec:
        if len(_theme_cache) >= _THEME_CACHE_MAX:
            _theme_cache.clear()
        _theme_cache[key] = vec
    return vec


# ---------------------------------------------------------------------------
# Binary persistence (Fix 2A)
# ---------------------------------------------------------------------------
#
# The index is machine-only derived state and the float vectors dominate its
# size. Storing them as float32 binary instead of pretty-printed text floats is
# ~4x smaller (102 MB -> ~25 MB) with a no-parse load. One self-contained npz
# per save (crash-safe, per-note): all `vec`s concatenated into one flat array,
# all `title_vec`s into another, with a small JSON `meta` blob giving each note's
# name/ts and its slice lengths. Flat-concat (not a 2D matrix) so ragged/odd-dim
# vectors survive a reformat untouched.

def _serialize_notes(notes: dict[str, dict[str, Any]]) -> bytes:
    import io

    meta: dict[str, Any] = {"version": 2, "notes": {}}
    vecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    for path, entry in notes.items():
        v = np.asarray(entry.get("vec", []), dtype=np.float32).ravel()
        vecs.append(v)
        m: dict[str, Any] = {
            "name": entry.get("name", ""),
            "ts": entry.get("ts", 0.0),
            "vlen": int(v.size),
        }
        tv = entry.get("title_vec")
        if tv is not None:
            tva = np.asarray(tv, dtype=np.float32).ravel()
            tvecs.append(tva)
            m["tlen"] = int(tva.size)
        ch = entry.get("content_hash")
        if ch:
            m["chash"] = ch
        meta["notes"][path] = m

    flat = np.concatenate(vecs) if vecs else np.zeros(0, dtype=np.float32)
    tflat = np.concatenate(tvecs) if tvecs else np.zeros(0, dtype=np.float32)
    meta_arr = np.frombuffer(orjson.dumps(meta), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez(buf, flat=flat, tflat=tflat, meta=meta_arr)
    return buf.getvalue()


def _deserialize_notes(raw: bytes) -> dict[str, dict[str, Any]]:
    import io

    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            flat = z["flat"]
            tflat = z["tflat"]
            meta = orjson.loads(z["meta"].tobytes())
    except Exception:
        return {}

    notes: dict[str, dict[str, Any]] = {}
    voff = toff = 0
    for path, m in meta.get("notes", {}).items():
        vlen = int(m.get("vlen", 0))
        entry: dict[str, Any] = {
            "vec": flat[voff:voff + vlen].tolist(),
            "name": m.get("name", ""),
            "ts": m.get("ts", 0.0),
        }
        voff += vlen
        tlen = m.get("tlen")
        if tlen is not None:
            tlen = int(tlen)
            entry["title_vec"] = tflat[toff:toff + tlen].tolist()
            toff += tlen
        ch = m.get("chash")
        if ch:
            entry["content_hash"] = ch
        notes[path] = entry
    return notes


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
        # Resolve lazily so tests can monkeypatch `_index_path` after import
        self._path = path if path is not None else _index_path()
        self._notes: dict[str, dict[str, Any]] = {}
        # Lazily-built, unit-normalized search matrix (numpy). Invalidated on any
        # mutation; rebuilt on the next cosine_top_k. Keeps _notes authoritative
        # while making search a single BLAS matrix-vector product.
        self._mat: np.ndarray | None = None
        self._mat_paths: list[str] = []
        self._mat_dim: int | None = None
        self._tmat: np.ndarray | None = None
        self._tmat_paths: list[str] = []
        self._tmat_dim: int | None = None
        self._load()

    def _invalidate_matrix(self) -> None:
        self._mat = None
        self._mat_paths = []
        self._mat_dim = None
        self._tmat = None
        self._tmat_paths = []
        self._tmat_dim = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self._invalidate_matrix()
        src = self._path
        if not src.exists() and src != _LEGACY_INDEX_PATH and _LEGACY_INDEX_PATH.exists():
            src = _LEGACY_INDEX_PATH  # one-time soft migration: copied forward on next save()
        if not src.exists():
            return
        try:
            raw = src.read_bytes()
        except Exception:
            return
        # Sniff the format: npz archives start with the zip magic 'PK'; the
        # legacy index is orjson text starting with '{'. Old files auto-migrate
        # to binary on the next save() — reformat, never re-embed.
        if raw[:2] == b"PK":
            self._notes = _deserialize_notes(raw)
        else:
            try:
                self._notes = orjson.loads(raw).get("notes", {})
            except Exception:
                self._notes = {}

    def save(self) -> Path:
        atomic_write_bytes(self._path, _serialize_notes(self._notes))
        return self._path

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def upsert(self, path: str, name: str, vec: list[float],
                *, title_vec: list[float] | None = None,
                content_hash: str | None = None) -> None:
        """Insert or replace a note's embedding.

        `title_vec` is the secondary title-only vector used for the dedup
        title-similarity gate. Omitting it preserves any existing title_vec
        stored for that path (backward-compatible with old index entries).

        `content_hash` is the signature of the embedded text (see
        `_embed_signature`); build_index uses it to skip unchanged notes and
        re-embed edited ones. Omitting it preserves any existing hash.
        """
        existing = self._notes.get(path, {})
        entry: dict[str, Any] = {"vec": vec, "name": name, "ts": time.time()}
        # Preserve existing title_vec if not explicitly provided
        resolved_tv = title_vec if title_vec is not None else existing.get("title_vec")
        if resolved_tv is not None:
            entry["title_vec"] = resolved_tv
        resolved_ch = content_hash if content_hash is not None else existing.get("content_hash")
        if resolved_ch is not None:
            entry["content_hash"] = resolved_ch
        self._notes[path] = entry
        self._invalidate_matrix()

    def delete(self, path: str) -> None:
        self._notes.pop(path, None)
        self._invalidate_matrix()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_vec(self, path: str) -> list[float] | None:
        entry = self._notes.get(path)
        return entry["vec"] if entry else None

    def get_title_vec(self, path: str) -> list[float] | None:
        """Return the title-only embedding vector, or None if not yet indexed.

        Returns None for old index entries that pre-date the title_vec feature;
        callers must handle the None case (title_score = 0.0 fallback).
        """
        entry = self._notes.get(path)
        return entry.get("title_vec") if entry else None

    def get_content_hash(self, path: str) -> str | None:
        """Return the embedded-text signature, or None for un-hashed entries.

        None for notes indexed before content-change detection existed; such
        entries are treated as stale (re-embedded once to backfill the hash).
        """
        entry = self._notes.get(path)
        return entry.get("content_hash") if entry else None

    def has(self, path: str) -> bool:
        return path in self._notes

    def paths(self) -> list[str]:
        return list(self._notes.keys())

    def __len__(self) -> int:
        return len(self._notes)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_matrix(self, vec_key: str) -> tuple[np.ndarray, list[str], int | None]:
        """Unit-normalized search matrix, path list, and dim for a vector field.

        Only notes carrying the field and sharing its modal dimension are placed
        in the matrix; a note missing the field (e.g. a legacy entry with no
        title_vec) or off-dimension falls through to a 0.0 score, matching the
        old per-pair _cosine length guard. Zero vectors normalize to zero rows.
        """
        vecs = {p: self._notes[p].get(vec_key) for p in self._notes}
        paths = [p for p, v in vecs.items() if v]
        if not paths:
            return np.zeros((0, 0), dtype=np.float32), [], None
        # Modal dimension, not the first note's: in a mixed-dim store (post model
        # swap, A11) the first note may be a minority dim, which would drop the
        # majority to 0.0 and, if the query matches the majority dim, zero the leg.
        dim = Counter(len(vecs[p]) for p in paths).most_common(1)[0][0]
        kept = [p for p in paths if len(vecs[p]) == dim]
        mat = np.asarray([vecs[p] for p in kept], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # zero rows stay zero → 0.0 similarity
        return mat / norms, kept, dim

    def _ensure_matrix(self) -> None:
        """Build the body + title search matrices from _notes (lazy, cached).

        Both are built and invalidated together under one guard, so every
        mutation that resets _mat also refreshes the title matrix.
        """
        if self._mat is not None:
            return
        self._mat, self._mat_paths, self._mat_dim = self._build_matrix("vec")
        self._tmat, self._tmat_paths, self._tmat_dim = self._build_matrix("title_vec")

    def _search(
        self,
        mat: np.ndarray | None,
        mat_paths: list[str],
        mat_dim: int | None,
        query_vec: list[float],
        k: int,
        exclude: set[str] | None,
    ) -> list[dict[str, Any]]:
        exclude = exclude or set()
        q = np.asarray(query_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        matrix: list[tuple[float, str]] = []
        if q_norm != 0.0 and mat is not None and mat.size and mat_dim == q.shape[0]:
            sims = mat @ (q / q_norm)
            matrix = [(float(s), p) for p, s in zip(mat_paths, sims.tolist()) if p not in exclude]
        # heapq.nlargest(k, ...) is documented-equivalent to sorted(reverse=True)[:k]:
        # same top-k, same (score, path) desc tie-break, but O(N log k).
        # Notes outside the matrix (missing field, off-dim, or dim mismatch) score
        # exactly 0.0 — _cosine's degenerate cases. A 0.0/negative note can only enter
        # the top-k when the matrix yields fewer than k strictly-positive hits, since any
        # positive score outranks it. In that common case nlargest over the matrix alone
        # is bit-identical; only otherwise do we pay the full-vault scan to place the
        # off-matrix 0.0 rows correctly.
        top = heapq.nlargest(k, matrix)
        if len(top) < k or not top or top[-1][0] <= 0.0:
            scored = {p: s for s, p in matrix}
            results = [(scored.get(p, 0.0), p) for p in self._notes if p not in exclude]
            top = heapq.nlargest(k, results)
        return [
            {"path": path, "name": self._notes[path]["name"], "score": round(float(score), 4)}
            for score, path in top
        ]

    def cosine_top_k(
        self,
        query_vec: list[float],
        k: int = 5,
        exclude: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k most similar notes as dicts with keys:
            path, name, score
        Optionally exclude a set of paths (e.g. the query note itself).

        Search is a single normalized matrix-vector product (numpy/BLAS); this
        is the hot path for COLLISION and AUTOLINK on large vaults.
        """
        self._ensure_matrix()
        return self._search(self._mat, self._mat_paths, self._mat_dim, query_vec, k, exclude)

    def title_cosine_top_k(
        self,
        query_vec: list[float],
        k: int = 5,
        exclude: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k notes by TITLE-vector cosine (like-vs-like title identity).

        The novelty gate's order parameter: a short concept name embedded as a
        title and scored against stored title vectors, never against full note
        bodies — the body signal was measured not to separate captured from
        novel concepts (their cosine distributions overlap). Notes predating the
        title_vec feature score 0.0.
        """
        self._ensure_matrix()
        return self._search(self._tmat, self._tmat_paths, self._tmat_dim, query_vec, k, exclude)


# ---------------------------------------------------------------------------
# Cached accessor (the seam — Fix 3)
# ---------------------------------------------------------------------------

# Process-lifetime cache keyed by resolved index path. Keying by the *path*
# (not the raw vault) is a superset of per-vault keying: it follows a /vault
# switch automatically and respects tests that monkeypatch `_index_path`.
_STORE_CACHE: dict[str, "EmbedStore"] = {}


def get_store() -> "EmbedStore":
    """Return the shared EmbedStore for the current vault's index.

    A process-lifetime singleton per resolved index path: readers stop
    re-deserialising the index, and the write path mutates the same instance
    every reader sees (no reload needed for consistency). Use `clear()` in tests.
    """
    from silica.kernel.paths import path_keyed_singleton
    return path_keyed_singleton(_STORE_CACHE, str(_index_path()), EmbedStore)


def clear() -> None:
    """Drop all cached stores (test isolation; also frees memory on /vault switch)."""
    _STORE_CACHE.clear()


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _note_text(title: str, body: str, *, folder: str = "") -> str:
    """Combine title and body prefix for embedding.

    If `folder` is provided, it is prepended as a bracketed domain hint
    (e.g. "[Robotica] CAN\n\n...") to anchor domain-ambiguous acronyms
    in their correct semantic neighbourhood. This never alters vault content.

    Images and other media embeds are stripped via kernel.media.strip_images
    before the text is truncated, so they never pollute the embedding space.
    """
    from silica.kernel.media import strip_images
    prefix = f"[{folder}] " if folder else ""
    combined = f"{prefix}{title}\n\n{strip_images(body)}"
    return combined[:_MAX_CHARS]

def _note_title_text(title: str, *, folder: str = "") -> str:
    """Title-only text for the secondary title-similarity embedding vector.

    Used alongside `_note_text` to build a compact, body-free representation
    that captures title-level semantic relationships (e.g. "ROS" ↔ "JSON in
    ROS 2") even when the full-note vectors diverge below the dedup threshold.
    """
    prefix = f"[{folder}] " if folder else ""
    return f"{prefix}{title}"


def _embed_signature(name: str, body: str, *, folder: str = "", model: str = "") -> str:
    """Stable hash of the exact text that determines a note's embedding.

    Signed over the truncated/image-stripped `_note_text` plus `_note_title_text`
    — not the raw body — so edits past the truncation point or inside stripped
    media syntax don't trigger a needless re-embed. build_index compares this
    against the stored hash to detect content changes on incremental refresh.

    `model` (A11): the embedder identity is part of the basis, so a model swap
    marks every content-unchanged note stale and re-embeds it. Without it,
    unchanged notes keep old-dimension vectors while new notes get new-dimension
    ones — a mixed-dimension store that can silently zero the whole embed leg.
    """
    import hashlib
    basis = (_note_text(name, body, folder=folder) + "\x00"
             + _note_title_text(name, folder=folder) + "\x00" + model)
    return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()


def build_index(
    embedder: Any,
    notes: list[tuple[str, str, str]],
    *,
    store: EmbedStore | None = None,
    batch_size: int = 32,
    force: bool = False,
    save: bool = True,
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

    Embedding strategy — interleaved batch:
        For each note we embed two texts in one call:
            [full_0, title_0, full_1, title_1, ...]
        Full vectors (even indices)  → note's primary `vec`.
        Title vectors (odd indices)  → note's secondary `title_vec`.
        This captures title-level relationships for the dedup title-gate
        with zero extra API round-trips.
    """
    if store is None:
        store = get_store()
    _model = getattr(embedder, "model", "")

    def _stale(path: str, name: str, body: str) -> bool:
        # Re-embed when new, forced, or the embedded text changed since last
        # indexing (hand-edits, bridge writes, organize). A present note with no
        # stored hash (pre-feature index) is treated as stale → backfilled once.
        if force or not store.has(path):
            return True
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        return store.get_content_hash(path) != _embed_signature(name, body, folder=folder, model=_model)

    to_embed = [(path, name, body) for path, name, body in notes if _stale(path, name, body)]

    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i : i + batch_size]
        folders = [path.rsplit("/", 1)[0] if "/" in path else "" for path, _, _ in batch]
        full_texts  = [_note_text(name, body, folder=f)  for (_, name, body), f in zip(batch, folders)]
        title_texts = [_note_title_text(name, folder=f)  for (_, name, _),    f in zip(batch, folders)]
        # Interleave: [full_0, title_0, full_1, title_1, ...]
        interleaved = [t for pair in zip(full_texts, title_texts) for t in pair]
        try:
            vecs = embedder.embed(interleaved)
        except Exception as exc:
            raise RuntimeError(f"Embedding call failed on batch {i//batch_size}: {exc}") from exc
        # A backend that dedups/drops/returns an odd count would make the 0::2/1::2
        # de-interleave silently truncate and mispair full-vs-title vectors, poisoning
        # the index with no error. Fail loud like COLLISION's batched_ok guard (A12).
        if len(vecs) != 2 * len(batch):
            raise RuntimeError(
                f"Embedder returned {len(vecs)} vectors for {2 * len(batch)} interleaved "
                f"inputs (batch {i//batch_size}); refusing to de-interleave a mispaired reply"
            )
        full_vecs  = vecs[0::2]  # even positions
        title_vecs = vecs[1::2]  # odd positions
        for (path, name, body), fv, tv, f in zip(batch, full_vecs, title_vecs, folders):
            store.upsert(path, name, fv, title_vec=tv,
                         content_hash=_embed_signature(name, body, folder=f, model=_model))

    if save:
        store.save()
    return store


def refresh_note(
    embedder: Any,
    path: str,
    name: str,
    body: str,
    *,
    store: EmbedStore | None = None,
    save: bool = True,
) -> EmbedStore:
    """Re-embed a single note and (by default) persist the updated store.

    Designed to be called after a note is written to the vault (freshness hook).
    Embeds both the full note text and the title-only text in a single API call.

    ``save=False`` (Fix A) upserts into the in-memory store only — the caller
    flushes once at end-of-run instead of rewriting the whole index per note.
    """
    if store is None:
        store = get_store()
    _folder = path.rsplit("/", 1)[0] if "/" in path else ""
    full_text  = _note_text(name, body, folder=_folder)
    title_text = _note_title_text(name, folder=_folder)
    vecs = embedder.embed([full_text, title_text])
    store.upsert(path, name, vecs[0], title_vec=vecs[1],
                 content_hash=_embed_signature(name, body, folder=_folder, model=getattr(embedder, "model", "")))
    if save:
        store.save()
    return store
