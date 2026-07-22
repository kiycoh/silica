# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Worklist index of contested note paths.

The run digest surfaces contested notes for a human to resolve; scanning every
note's frontmatter each digest would be a full-vault read, so the flag tool
records the path here and the digest reads this small list. Truth stays in each
note's frontmatter, so this file is a rebuildable index, not a source of record.

ponytail: a JSON list under the per-vault index dir, atomic write, dedup on add.
No schema, no db.
"""
from __future__ import annotations

import json

from silica.kernel.paths import atomic_write_bytes, index_dir


def _register_path():
    return index_dir() / "contested_register.json"


def entries() -> list[str]:
    """Recorded contested paths in insertion order; [] on missing/corrupt file."""
    try:
        data = json.loads(_register_path().read_text(encoding="utf-8"))
    except Exception:
        return []
    return [p for p in data if isinstance(p, str)] if isinstance(data, list) else []


def _save(paths: list[str]) -> None:
    p = _register_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(p, json.dumps(paths).encode("utf-8"))


def add(path: str) -> None:
    """Record a contested note path (idempotent, order-preserving)."""
    cur = entries()
    if path not in cur:
        _save(cur + [path])


def discard(path: str) -> None:
    """Drop a path from the register (no-op when absent)."""
    cur = entries()
    if path in cur:
        _save([p for p in cur if p != path])
