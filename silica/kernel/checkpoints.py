# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Per-note checkpoint stack — durable undo for interactive patches.

Every successful interactive patch (silica_patch_note) pushes the resulting
note content onto a per-path stack. ``/undo`` pops the top and restores the
note to the new top, walking back one patch at a time. The very first push for
a note also seeds the *original* (pre-patch) content as the immovable floor, so
a chain of undos returns the note to exactly how it was before Silica touched it.

This is deliberately separate from the FSM's transactional snapshot/rollback
(silica_snapshot / silica_restore): that protects a whole pipeline run and is
discarded on success, whereas this is a lightweight, user-facing edit history
for single notes that survives across REPL sessions.

Storage: ~/.silica/checkpoints.db (SQLite), one row per restore point, ordered
by autoincrement id. Keyed by vault-relative path.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_DEFAULT_CHECKPOINT_PATH = Path.home() / ".silica" / "checkpoints.db"


class CheckpointStore:
    """A persistent stack of full-note restore points, keyed by vault path."""

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else _DEFAULT_CHECKPOINT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                path       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_checkpoints_path
                ON checkpoints(path);
            """
        )
        self._conn.commit()

    # -- internals ---------------------------------------------------------

    def _insert(self, path: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO checkpoints (path, content, created_at) VALUES (?, ?, ?)",
            (path, content, time.time()),
        )
        self._conn.commit()

    # -- write -------------------------------------------------------------

    def push(self, path: str, prior_content: str, new_content: str) -> int:
        """Record a restore point after a patch.

        On the first push for ``path`` the original ``prior_content`` is seeded
        as the floor entry before ``new_content``, so undo can reach the
        pre-patch state. Subsequent pushes append only ``new_content``
        (``prior_content`` is ignored — it already sits on top of the stack).

        Returns the resulting stack depth.
        """
        if self.depth(path) == 0:
            self._insert(path, prior_content)
        self._insert(path, new_content)
        return self.depth(path)

    # -- read --------------------------------------------------------------

    def depth(self, path: str) -> int:
        """Number of restore points stored for a note (floor included)."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM checkpoints WHERE path = ?", (path,)
        ).fetchone()
        return int(row["n"]) if row else 0

    def most_recent_path(self) -> str | None:
        """Vault path of the most recently pushed checkpoint, or None if empty.

        Backs ``/undo`` with no argument, including after a restart.
        """
        row = self._conn.execute(
            "SELECT path FROM checkpoints ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["path"] if row else None

    # -- undo --------------------------------------------------------------

    def undo(self, path: str) -> str | None:
        """Pop the top restore point and return the content to restore to.

        Removes the most recent entry for ``path`` and returns the content of
        the new top (the previous patch's result, or the original floor). The
        floor entry is never removed: once only it remains there is nothing left
        to undo and this returns None.

        Returns None if there is nothing to undo (no entries, or only the
        floor remains). The caller is responsible for overwriting the note with
        the returned content.
        """
        if self.depth(path) <= 1:
            return None

        top = self._conn.execute(
            "SELECT id FROM checkpoints WHERE path = ? ORDER BY id DESC LIMIT 1",
            (path,),
        ).fetchone()
        if top is None:
            return None
        self._conn.execute("DELETE FROM checkpoints WHERE id = ?", (top["id"],))
        self._conn.commit()

        new_top = self._conn.execute(
            "SELECT content FROM checkpoints WHERE path = ? ORDER BY id DESC LIMIT 1",
            (path,),
        ).fetchone()
        return new_top["content"] if new_top else None

    def clear(self, path: str) -> None:
        """Drop all restore points for a note."""
        self._conn.execute("DELETE FROM checkpoints WHERE path = ?", (path,))
        self._conn.commit()


_store: CheckpointStore | None = None


def get_checkpoint_store(path: Path | str | None = None) -> CheckpointStore:
    global _store
    if _store is None:
        _store = CheckpointStore(path)
    return _store
