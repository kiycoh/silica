# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

# silica/kernel/undo_journal.py
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from silica.kernel.ops import InverseOp, InverseOpKind

_DEFAULT_JOURNAL_PATH = Path.home() / ".silica" / "undo_journal.db"


class UndoJournalStore:
    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else _DEFAULT_JOURNAL_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id      TEXT PRIMARY KEY,
                source      TEXT,
                started_at  REAL NOT NULL,
                reverted_at REAL
            );
            CREATE TABLE IF NOT EXISTS inverses (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id        TEXT NOT NULL,
                path          TEXT NOT NULL,
                kind          TEXT NOT NULL,
                version       INTEGER,
                prior_content TEXT,
                post_hash     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_inverses_run ON inverses(run_id);
            """
        )
        self._conn.commit()

    def start_run(self, source: str | None = None) -> str:
        run_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO runs (run_id, source, started_at) VALUES (?, ?, ?)",
            (run_id, source, time.time()),
        )
        self._conn.commit()
        return run_id

    def record(self, run_id: str, inverse: InverseOp, post_hash: str | None) -> None:
        self._conn.execute(
            "INSERT INTO inverses (run_id, path, kind, version, prior_content, post_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, inverse.path, inverse.kind.value, inverse.version,
             inverse.prior_content, post_hash),
        )
        self._conn.commit()

    def last_active_run(self) -> str | None:
        row = self._conn.execute(
            """
            SELECT r.run_id FROM runs r WHERE r.reverted_at IS NULL
            AND EXISTS (SELECT 1 FROM inverses i WHERE i.run_id = r.run_id)
            ORDER BY r.started_at DESC, r.rowid DESC LIMIT 1
            """
        ).fetchone()
        return row["run_id"] if row else None

    def inverses_for(self, run_id: str) -> list[tuple[InverseOp, str | None]]:
        rows = self._conn.execute(
            "SELECT path, kind, version, prior_content, post_hash "
            "FROM inverses WHERE run_id = ? ORDER BY id DESC",
            (run_id,),
        ).fetchall()
        out: list[tuple[InverseOp, str | None]] = []
        for r in rows:
            inv = InverseOp(
                kind=InverseOpKind(r["kind"]), path=r["path"],
                version=r["version"], prior_content=r["prior_content"],
            )
            out.append((inv, r["post_hash"]))
        return out

    def mark_reverted(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE runs SET reverted_at = ? WHERE run_id = ?", (time.time(), run_id)
        )
        self._conn.commit()


_store: UndoJournalStore | None = None


def get_undo_journal(path: Path | str | None = None) -> UndoJournalStore:
    global _store
    if _store is None:
        _store = UndoJournalStore(path)
    return _store


import hashlib as _hashlib

from silica.driver import DRIVER


def _content_hash(text: str | None) -> str:
    return _hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def revert_run(run_id: str, *, store: UndoJournalStore | None = None) -> dict:
    """Replay a run's inverses LIFO, refusing notes modified since the inject.

    Version guard: re-read each note, hash it. If recorded post_hash exists and
    the current hash differs (note edited since inject), skip it — don't clobber
    newer work. Mark the run reverted when done.
    """
    from silica.tools.wrapped import silica_restore

    store = store or get_undo_journal()
    entries = store.inverses_for(run_id)  # LIFO
    reverted: list[str] = []
    skipped: list[dict] = []
    errors: list[dict] = []

    for inv, post_hash in entries:
        try:
            current = DRIVER.read_note(inv.path).content
            cur_hash: str | None = _content_hash(current)
        except Exception:
            cur_hash = None  # note absent

        if post_hash is not None and cur_hash is not None and cur_hash != post_hash:
            skipped.append({"path": inv.path, "reason": "modified since inject"})
            continue

        try:
            silica_restore(txn_id=run_id, inverses=[inv.model_dump()])
            reverted.append(inv.path)
        except Exception as e:
            errors.append({"path": inv.path, "error": str(e)})

    store.mark_reverted(run_id)
    return {"run_id": run_id, "reverted": reverted, "skipped": skipped, "errors": errors}
