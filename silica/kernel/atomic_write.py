# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Per-note self-atomic write primitive.

commit_note_atomic applies a single Op with a micro-snapshot → write → lint
→ self-revert-on-failure lifecycle. This is the building block for per-note
independent atomicity without requiring a full batch transaction.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from silica.driver import DRIVER
from silica.kernel.bulk import VerifyMismatchError, execute_one
from silica.kernel.ops import Op, OpType, InverseOp


def _sha256(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass
class NoteCommitResult:
    ok: bool
    path: str
    op: str
    inverses: list[InverseOp] = field(default_factory=list)
    post_hash: str | None = None
    error: str | None = None
    reverted: bool = False


def commit_note_atomic(op: Op, *, hub: str | None = None, lint: bool = True) -> NoteCommitResult:
    """Apply one Op atomically: micro-snapshot -> write -> lint -> self-revert on fail.

    Args:
        op:   The single Op to apply (write / patch / overwrite / delete / skip).
        hub:  Optional hub override (falls back to op.hub).
        lint: Whether to run silica_lint after the write. On lint failure the
              write is reverted before returning.

    Returns:
        NoteCommitResult with ok=True on success, ok=False + reverted=True if
        lint failed and the note was restored to its prior state.
    """
    from silica.tools.wrapped import build_txn, silica_restore
    from silica.tools.composed import silica_lint

    path = op.touched_ref() or ""
    op_name = op.op.value if op.op else ""

    # 1. micro-snapshot — captures prior content / inverse strategy BEFORE write
    txn = build_txn([op])
    inverses: list[InverseOp] = list(txn.inverses)

    # 2. execute the single op
    try:
        execute_one(op)
    except VerifyMismatchError as e:
        # The DRIVER call already happened and the post-write read-back
        # proves something (possibly corrupted) landed on disk — unlike every
        # other execute_one failure (missing params, a driver error raised
        # before/without a successful write), "nothing landed" no longer
        # holds here. Revert via the same micro-snapshot inverses the
        # lint-failure branch below uses (built at step 1, before the write).
        silica_restore(
            txn_id=txn.id,
            inverses=[inv.model_dump() for inv in inverses],
        )
        return NoteCommitResult(
            ok=False, path=path, op=op_name, inverses=inverses, error=str(e),
            reverted=True,
        )
    except Exception as e:
        # Nothing landed (param validation, or the DRIVER call itself raised
        # before any write) — no revert needed, mirroring the pre-verify-gate
        # invariant this branch always had.
        return NoteCommitResult(
            ok=False, path=path, op=op_name, inverses=inverses, error=str(e)
        )

    # 3. optional lint on this note; revert on failure
    if lint and path:
        lr = silica_lint(path, op_type=op_name, hub=hub or op.hub or "")
        if not lr.get("success", True):
            silica_restore(
                txn_id=txn.id,
                inverses=[inv.model_dump() for inv in inverses],
            )
            return NoteCommitResult(
                ok=False,
                path=path,
                op=op_name,
                inverses=inverses,
                error=f"lint failed: {lr.get('errors')}",
                reverted=True,
            )

    # 4. success — capture post-write content hash
    post_hash: str | None = None
    if path:
        try:
            post_hash = _sha256(DRIVER.read_note(path).content)
        except Exception:
            post_hash = None

    return NoteCommitResult(
        ok=True, path=path, op=op_name, inverses=inverses, post_hash=post_hash
    )


@dataclass
class AtomicBulkResult:
    committed: list[NoteCommitResult] = field(default_factory=list)
    failed: list[NoteCommitResult] = field(default_factory=list)
    total: int = 0

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0


def bulk_write_atomic(ops: list[Op], *, hub: str | None = None, lint: bool = True) -> AtomicBulkResult:
    """Apply ops one note at a time; each note is self-atomic. No shared Txn.

    Skip ops are excluded. A note that fails (exec or lint) is reverted in place
    and recorded in `failed`; its siblings are untouched and land in `committed`.
    """
    committed: list[NoteCommitResult] = []
    failed: list[NoteCommitResult] = []
    actionable = [o for o in ops if o.op != OpType.skip]
    for op in actionable:
        res = commit_note_atomic(op, hub=hub, lint=lint)
        (committed if res.ok else failed).append(res)
    return AtomicBulkResult(committed=committed, failed=failed, total=len(actionable))
