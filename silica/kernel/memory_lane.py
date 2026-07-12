# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Personal-memory recall lane (ADR-0019).

The memory vault (default ~/.silica/vault, override SILICA_MEMORY_VAULT) is a
second, READ-ONLY pair of (embed, cooccur) stores fed to the same RRF fusion
as the active vault's legs. Writes never route here — material enters the
memory vault through its own inbox (UC1, same trust regime as any ingress).

Degenerate case (today's default without repo mode): when the active vault IS
the memory vault, the lane abstains — `memory_vault()` returns None, no store
is loaded twice, and fusion collapses to single-vault behavior bit-identically.
"""
from __future__ import annotations

from pathlib import Path

from silica.config import CONFIG


def memory_vault() -> Path | None:
    """Resolved memory-vault path, or None when the lane must abstain.

    Abstains when the memory vault coincides with the active vault (resolved
    path equality — nested vaults are out of scope, ADR-0019) or does not
    exist on disk yet.
    """
    raw = (getattr(CONFIG, "memory_vault", "") or "").strip()
    # Default mirrors cli.default_user_vault (kernel must not import cli).
    mem = (Path(raw).expanduser() if raw else Path.home() / ".silica" / "vault").resolve()
    active = (getattr(CONFIG, "vault_path", "") or "").strip()
    if active and Path(active).resolve() == mem:
        return None
    if not mem.is_dir():
        return None
    return mem


# Process-lifetime store cache keyed by the memory vault's index dir (twin of
# embed._STORE_CACHE). ponytail: an index built mid-session by another process
# is not picked up until restart; add an mtime check if that ever bites.
_CACHE: dict[str, tuple[object, object]] = {}


def memory_stores():
    """Memory-lane ``(embed_store, cooccur_store)`` for the fusion facade.

    Each leg is None when its index is absent/empty (the facade then abstains
    on that leg); ``(None, None)`` when the whole lane abstains. Never raises.
    """
    mem = memory_vault()
    if mem is None:
        return None, None
    try:
        from silica.kernel import paths
        from silica.kernel.cooccurrence import CooccurStore
        from silica.kernel.embed import EmbedStore

        idx = paths.index_dir_for(str(mem))
        key = str(idx)
        hit = _CACHE.get(key)
        if hit is None:
            # Explicit exists() guard: the store constructors soft-migrate from
            # the LEGACY global index when their file is missing — right for
            # the active vault, wrong for this lane (it would resurrect some
            # old vault's data as "memory"). A missing index ⇒ the leg abstains.
            ep = idx / "embeddings.json"
            cp = idx / "cooccurrence.json"
            hit = (
                EmbedStore(path=ep) if ep.is_file() else None,
                CooccurStore(path=cp) if cp.is_file() else None,
            )
            _CACHE[key] = hit
        es, cs = hit
        return (
            es if es is not None and len(es) else None,
            cs if cs is not None and len(cs) else None,
        )
    except Exception:
        return None, None


def clear() -> None:
    """Drop cached memory stores (test isolation)."""
    _CACHE.clear()
