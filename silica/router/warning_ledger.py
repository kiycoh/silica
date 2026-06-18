"""WarningLedger — run-scoped memory of non-blocking warnings.

The Injector's blocking gates (lint failure, blocking graph regression) roll the
chunk back, so they have no residue to repair.  But *non-blocking* warnings —
notably orphan notes (in-degree 0) — leave a committed note that may still be
fixable.  Those warnings are transient during a run (a later chunk, AUTOLINK, or
BACKLINK can connect the note), so we do NOT act on them per-chunk.

Instead the FSM records each warning here as it happens.  At the END of the run
the Coordinator recomputes which warnings are *still* unresolved and hands only
that residual to the leashed sub-agents, then re-verifies.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from pathlib import Path

import orjson


@dataclass
class WarningEntry:
    path: str            # vault-rel note path the warning is about
    kind: str            # e.g. "orphan"
    detail: str = ""     # human-readable context (the gate message)


class WarningLedger:
    """Thread-safe accumulator of run warnings, deduplicated by (path, kind)."""

    def __init__(self, run_dir: str | Path | None = None):
        self._entries: dict[tuple[str, str], WarningEntry] = {}
        self._lock = threading.Lock()
        try:
            self._run_dir = Path(run_dir) if isinstance(run_dir, (str, Path)) else None
        except (TypeError, ValueError):
            self._run_dir = None

    def add(self, path: str, kind: str, detail: str = "") -> None:
        if not path:
            return
        with self._lock:
            self._entries[(path, kind)] = WarningEntry(path=path, kind=kind, detail=detail)
        self._persist()

    def paths(self, kind: str | None = None) -> list[str]:
        with self._lock:
            return [
                e.path for e in self._entries.values()
                if kind is None or e.kind == kind
            ]

    def entries(self, kind: str | None = None) -> list[WarningEntry]:
        with self._lock:
            return [e for e in self._entries.values() if kind is None or e.kind == kind]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _persist(self) -> None:
        if not self._run_dir:
            return
        try:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = [asdict(e) for e in self._entries.values()]
                (self._run_dir / "warnings.json").write_bytes(
                    orjson.dumps(payload, option=orjson.OPT_INDENT_2)
                )
        except Exception:
            pass
