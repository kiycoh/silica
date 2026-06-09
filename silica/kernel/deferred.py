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


_DEFAULT_DEFERRED_DIR = Path.home() / ".silica" / "deferred"


class DeferredStore:
    def __init__(self, path: Path | str | None = None):
        self._dir = Path(path) if path else _DEFAULT_DEFERRED_DIR
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
            "rejected_ops": rejected_ops,
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

    def remove(self, content_hash: str) -> bool:
        p = self._bundle_path(content_hash)
        if p.exists():
            p.unlink()
            return True
        return False


_store: DeferredStore | None = None


def get_deferred_store(path: Path | str | None = None) -> DeferredStore:
    global _store
    if _store is None:
        _store = DeferredStore(path)
    return _store
