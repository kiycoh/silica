# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Deferred Op Store — persists rejected ops for later autonomous retry.

When the VALIDATE gate rejects some ops in the injector pipeline, the LLM-
generated content for those ops is saved here rather than discarded.  On the
next run of the same source file the orchestrator surfaces the bundle so the
model can retry autonomously (silica_deferred_retry) or the user can inspect
and discard (silica_deferred_list / silica_deferred_flush).

Storage: ~/.silica/deferred/<content_hash>.json
Bundle schema:
  content_hash     — SHA-256 of the source inbox file
  source_path      — vault-relative inbox path
  target_dir       — injection target folder
  hub              — hub note name (or null)
  timestamp        — epoch float
  rejected_ops     — list[Op dict] (no Rejection wrapper — ready to re-parse)
  rejection_reasons — {path_or_heading: reason}
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import orjson


# Pre-C2 location: one global queue for every vault. Read once as a migration
# source by get_deferred_store(), never written to again.
_LEGACY_DEFERRED_DIR = Path.home() / ".silica" / "deferred"


def _store_dir() -> Path:
    # Function, not constant: resolves per current vault; tests monkeypatch it.
    from silica.kernel import paths

    return paths.index_dir() / "deferred"


def _dedup_ops(ops: list[dict]) -> list[dict]:
    """Collapse ops sharing (path, heading) to their latest, first-seen order.

    _defer merges `existing + new` across runs; without this a file that stays
    in the inbox and re-rejects the same op duplicates it every run. dict keeps
    each key's first position and last value → stable order, newest content.
    """
    last: dict[tuple, int] = {}
    for i, o in enumerate(ops):
        key = (o.get("path"), o.get("heading"))
        if key != (None, None):  # unidentifiable ops can't be deduped — keep all
            last[key] = i
    return [
        o for i, o in enumerate(ops)
        if (o.get("path"), o.get("heading")) == (None, None)
        or last[(o.get("path"), o.get("heading"))] == i
    ]


class DeferredStore:
    def __init__(self, path: Path | str | None = None):
        self._dir = Path(path) if path else _store_dir()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _bundle_path(self, content_hash: str) -> Path:
        return self._dir / f"{content_hash}.json"

    def put(
        self,
        content_hash: str,
        source_path: str,
        target_dir: str,
        hub: str | None,
        rejected_ops: list[dict],
        rejection_reasons: dict[str, str] | None = None,
    ) -> None:
        """Persist (or overwrite) a deferred bundle for this content hash."""
        bundle: dict[str, Any] = {
            "content_hash": content_hash,
            "source_path": source_path,
            "target_dir": target_dir,
            "hub": hub,
            "timestamp": time.time(),
            "rejected_ops": _dedup_ops(rejected_ops),
            "rejection_reasons": rejection_reasons or {},
        }
        self._bundle_path(content_hash).write_bytes(
            orjson.dumps(bundle, option=orjson.OPT_INDENT_2)
        )

    def get(self, content_hash: str) -> dict[str, Any] | None:
        p = self._bundle_path(content_hash)
        if not p.exists():
            return None
        return orjson.loads(p.read_bytes())  # type: ignore[return-value]

    def list_all(self) -> list[dict[str, Any]]:
        result = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                bundle = orjson.loads(p.read_bytes())
                result.append({
                    "content_hash": bundle.get("content_hash", p.stem),
                    "source_path": bundle.get("source_path", ""),
                    "target_dir": bundle.get("target_dir", ""),
                    "hub": bundle.get("hub"),
                    "rejected_count": len(bundle.get("rejected_ops", [])),
                    "timestamp": bundle.get("timestamp", 0.0),
                })
            except Exception:
                pass
        return result

    def queue_depth(self) -> int:
        """Return the number of pending bundles in the deferred store."""
        return sum(1 for _ in self._dir.glob("*.json"))

    def remove_op(self, content_hash: str, heading: str) -> bool:
        """Drop the op with `heading` from a bundle (verdict routed elsewhere).

        Removes the whole bundle when its last op is dropped. Returns True iff
        an op was removed.
        """
        bundle = self.get(content_hash)
        if bundle is None:
            return False
        ops = bundle.get("rejected_ops", [])
        kept = [o for o in ops if o.get("heading") != heading]
        if len(kept) == len(ops):
            return False
        if not kept:
            return self.remove(content_hash)
        removed_paths = {o.get("path") for o in ops if o.get("heading") == heading}
        bundle["rejected_ops"] = kept
        bundle["rejection_reasons"] = {
            k: v for k, v in bundle.get("rejection_reasons", {}).items()
            if k != heading and k not in removed_paths
        }
        self._bundle_path(content_hash).write_bytes(
            orjson.dumps(bundle, option=orjson.OPT_INDENT_2)
        )
        return True

    def remove(self, content_hash: str) -> bool:
        p = self._bundle_path(content_hash)
        if p.exists():
            p.unlink()
            return True
        return False


# Keyed by resolved store dir so a /vault switch serves the new vault's queue
# instead of a stale singleton (same pattern as embed/cooccurrence stores).
_stores: dict[str, DeferredStore] = {}


def get_deferred_store(path: Path | str | None = None) -> DeferredStore:
    key = str(Path(path) if path else _store_dir())
    store = _stores.get(key)
    if store is None:
        store = _stores[key] = DeferredStore(key)
        if path is None:
            _adopt_legacy(store)
    return store


# The pre-C2 global store accumulated test pollution: FSM tests deferring
# through the un-isolated default wrote their fake-lint fixture bundles
# (every rejection reason == this string) next to real user bundles.
_FIXTURE_REASON = "lint failed: ['e']"


def _adopt_legacy(store: DeferredStore) -> None:
    """One-shot drain of the legacy global ~/.silica/deferred into `store`.

    Real bundles are adopted into the active vault's queue (they cannot be
    attributed to a specific vault, and the active one is where the user is
    working); fixture-pollution bundles are irrecoverable test artifacts and
    are flushed. Best-effort: unreadable files are left in place.
    """
    legacy = _LEGACY_DEFERRED_DIR
    if not legacy.is_dir() or legacy == store._dir:
        return
    for p in legacy.glob("*.json"):
        try:
            bundle = orjson.loads(p.read_bytes())
            reasons = list((bundle.get("rejection_reasons") or {}).values())
            if not (reasons and all(r == _FIXTURE_REASON for r in reasons)):
                store._bundle_path(bundle.get("content_hash", p.stem)).write_bytes(
                    orjson.dumps(bundle, option=orjson.OPT_INDENT_2)
                )
            p.unlink()
        except Exception:
            continue
