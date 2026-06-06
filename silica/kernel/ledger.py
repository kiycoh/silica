"""SQLite ledger for idempotent resume (ADR-011 / Addendum C5).

Tracks per-op outcomes. The orchestrator writes here after commit.
On re-run, the router checks the ledger: a source with all ops 'committed',
a matching content_hash, and all outputs still on disk is skipped.

Schema:
    ops(id, txn_id, source_canonical, path, op, status, content_hash, ts)
    status ∈ {committed, failed, rolled_back}
    UNIQUE constraint on (source_canonical, path) → UPSERT semantics (C2.4)

Migration:
    Existing DBs with column 'source_basename' are renamed to 'source_canonical'.
    Existing rows without 'content_hash' (NULL) are treated as stale (C2.6).
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_DEFAULT_LEDGER_PATH = Path.home() / ".silica" / "ledger.db"


class Ledger:
    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else _DEFAULT_LEDGER_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema management & migration
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        # Run column migration BEFORE creating table (handles renamed columns in existing DBs)
        self._migrate()

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ops (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                txn_id            TEXT NOT NULL,
                source_canonical  TEXT NOT NULL,
                path              TEXT,
                op                TEXT NOT NULL,
                status            TEXT NOT NULL,
                content_hash      TEXT,
                ts                REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON ops(source_canonical)"
        )
        # Unique index enables UPSERT (C2.4)
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_src_path "
            "ON ops(source_canonical, path)"
        )
        self._conn.commit()

    def _migrate(self) -> None:
        """Apply backward-safe migrations for pre-existing ledger.db files."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(ops)")}

        # Table doesn't exist yet — nothing to migrate
        if not cols:
            return

        # Rename source_basename → source_canonical (SQLite ≥ 3.25)
        if "source_basename" in cols and "source_canonical" not in cols:
            self._conn.execute(
                "ALTER TABLE ops RENAME COLUMN source_basename TO source_canonical"
            )
            self._conn.commit()
            cols.add("source_canonical")
            cols.discard("source_basename")

        # Add content_hash column if missing (NULL = stale, C2.6)
        if "content_hash" not in cols:
            self._conn.execute("ALTER TABLE ops ADD COLUMN content_hash TEXT")
            self._conn.commit()

        # Add unique index if not yet present (idempotent)
        existing_idx = {
            row[1]
            for row in self._conn.execute("SELECT * FROM sqlite_master WHERE type='index'")
        }
        if "idx_src_path" not in existing_idx:
            try:
                self._conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_src_path "
                    "ON ops(source_canonical, path)"
                )
                self._conn.commit()
            except Exception:
                pass  # may fail on duplicate data from old schema

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        txn_id: str,
        source_canonical: str,
        path: str | None,
        op: str,
        status: str,
        content_hash: str | None = None,
    ) -> None:
        """Upsert an op record keyed by (source_canonical, path)."""
        self._conn.execute(
            """
            INSERT INTO ops(txn_id, source_canonical, path, op, status, content_hash, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_canonical, path)
            DO UPDATE SET
                txn_id       = excluded.txn_id,
                op           = excluded.op,
                status       = excluded.status,
                content_hash = excluded.content_hash,
                ts           = excluded.ts
            """,
            (txn_id, source_canonical, path, op, status, content_hash, time.time()),
        )
        self._conn.commit()

    def mark_rolled_back(self, txn_id: str) -> None:
        """Mark all ops of a transaction as rolled_back."""
        self._conn.execute(
            "UPDATE ops SET status='rolled_back' WHERE txn_id=?",
            (txn_id,),
        )
        self._conn.commit()

    def mark_failed(self, txn_id: str) -> None:
        """Mark all ops of a transaction as failed (abort before WRITE)."""
        self._conn.execute(
            "UPDATE ops SET status='failed' WHERE txn_id=?",
            (txn_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read — skip check (C2.1–C2.6)
    # ------------------------------------------------------------------

    def is_committed(
        self,
        source_canonical: str,
        content_hash: str | None = None,
    ) -> bool:
        """True iff ALL of the following hold:

        1. There are rows for source_canonical.
        2. All rows have status='committed'.
        3. The stored content_hash matches the provided hash (NULL → stale).
        4. Every registered output path exists on disk.
        """
        rows = self._conn.execute(
            "SELECT status, content_hash, path FROM ops WHERE source_canonical=?",
            (source_canonical,),
        ).fetchall()

        if not rows:
            return False

        for status, stored_hash, path in rows:
            # All must be committed
            if status != "committed":
                return False
            # Hash must match (NULL → stale)
            if stored_hash is None or stored_hash != content_hash:
                return False
            # Output must still exist on disk
            if path and not Path(path).exists():
                return False

        return True

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton — lazily initialised
# ---------------------------------------------------------------------------

_ledger: Ledger | None = None


def get_ledger(path: Path | str | None = None) -> Ledger:
    global _ledger
    if _ledger is None:
        _ledger = Ledger(path)
    return _ledger
