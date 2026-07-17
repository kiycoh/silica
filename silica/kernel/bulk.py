# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Vault write executor — the single authorised path for applying Ops.

Every write in Silica (FSM bulk write, deferred retry, interactive patch)
funnels through here. The per-op-type logic lives in small helpers; `execute_one`
is the shared single-op entry point and `execute_operations` is the batch loop
that aggregates results into a BulkResult.

Keeping one executor means any future change to how a write reaches disk
(e.g. atomic file writes at the DRIVER level) is inherited by every caller.

Post-write verify (falsifiable gate): every successful write/overwrite/patch/
delete dispatch is followed by a read-back through the DRIVER (_verify_landed /
_verify_deleted) that is compared against the body actually composed for the
DRIVER call — never raw op.content. `success: True` used to mean only "the
DRIVER call didn't raise"; a mismatch here raises VerifyMismatchError, which
the existing exception handling in execute_one/execute_operations already
turns into a failed op — no new failure machinery, just a stricter definition
of success.

VerifyMismatchError is a distinct type (not a plain ValueError) because it
carries a different fact than every other execute_one failure: the DRIVER
call already happened and something landed on disk (possibly corrupted).
Every other failure (missing params, a driver error raised before/without a
successful write) still means "nothing landed" — commit_note_atomic relies on
this distinction to decide whether a revert is needed.
"""
from __future__ import annotations

from silica.driver import DRIVER
from silica.kernel import templates
from silica.kernel.merge import three_way_merge
from silica.kernel.ops import Op, OpType, FailedOp, BulkResult


class VerifyMismatchError(RuntimeError):
    """Raised by _verify_landed/_verify_deleted: the write/delete already hit
    the DRIVER and the post-write read-back proves it didn't land as intended
    (or, for delete, didn't land at all). Unlike other execute_one failures,
    this means disk state may have changed — see commit_note_atomic."""


def _verify_landed(op: Op, path: str, intended: str | None) -> str | None:
    """Falsifiable gate: rilegge dal DRIVER e confronta. None = ok, str = errore.

    `intended` is the final body composed by the caller (post frontmatter/merge/
    patch enrichment — write, overwrite AND patch all pass their exact composed
    body) — never raw op.content or op.snippet — so this is a check against
    exactly what was handed to the DRIVER.

    STOPGAP (deliberate, ceiling below): comparison strips edge whitespace
    before comparing. The tolerance dates from the deleted cli backend, whose
    read channel stripped stdout edges; fs reads bytes verbatim and ws returns
    Obsidian's cachedRead. Interior content still compares byte-exact, so real
    corruption (e.g. the 2026-06-30 backslash-doubling bug) is still caught.
    Ceiling: verify ws edge-faithfulness live, then go byte-exact — do not
    widen the tolerance further.
    """
    try:
        landed = DRIVER.read_note(path).content or ""
    except RuntimeError as e:
        return f"post-write verify: read-back failed: {e}"
    if intended is not None and landed.strip() != intended.strip():
        return "post-write verify: content mismatch (backend altered payload)"
    return None


def _verify_deleted(path: str) -> str | None:
    """Falsifiable gate for delete: read-back must now fail (existence negated).

    Only a genuine "note not found" style RuntimeError counts as confirmation.
    Both backends' read_note raise that shape for a missing note — fs_backend:
    "File not found: {path}"; ws_backend surfaces the plugin's "not found"
    reply — both contain "file" and "not found". A dead read channel (bridge
    down, socket error) also raises RuntimeError but with a different shape
    and must NOT be read as "verified deleted".
    """
    try:
        DRIVER.read_note(path)
    except RuntimeError as e:
        msg = str(e).lower()
        if "file" in msg and "not found" in msg:
            return None
        return f"post-write verify: delete check inconclusive: {e}"
    return "post-write verify: note still present after delete"


def _execute_write(op: Op, path: str) -> dict:
    """Create a new note from the resolved template. Requires heading + hub."""
    heading = op.heading
    snippet = op.snippet or ""
    hub = op.hub

    if not heading or not hub:
        raise ValueError("Missing 'heading' or 'hub' parameter for write operation")

    fields = templates.prepare_fields(
        title=op.title or heading,
        body=snippet,
        hub=hub,
        tags=op.tags,
        related=op.related,
        parent=op.parent,
    )
    content = templates.render_note(templates.resolve_template(), fields)
    content = templates.ensure_system_floor(content)
    DRIVER.create(path, content)
    err = _verify_landed(op, path, content)
    if err:
        raise VerifyMismatchError(err)
    return {"path": path, "op": "write", "success": True}


def _execute_patch(op: Op, path: str) -> dict:
    """Append a distilled snippet to an existing note. Requires heading + snippet + source_basename."""
    heading = op.heading
    snippet = op.snippet
    source_basename = op.source_basename

    if not heading or not snippet or not source_basename:
        raise ValueError(
            "Missing 'heading', 'snippet', or 'source_basename' for patch operation"
        )

    try:
        nc = DRIVER.read_note(path)
    except RuntimeError as e:
        # Preserve the historical message so callers/tests can match on it.
        raise ValueError(f"Cannot patch; {e}") from e

    # Idempotent re-injection: if this exact provenance block already exists,
    # skip the append (deterministic, no LLM). Re-running the same source is a no-op.
    if templates.block_present(nc.content, heading, source_basename):
        return {"path": path, "op": "patch", "success": True, "skipped": "duplicate"}

    new_content = templates.patch_snippet(
        heading=heading,
        snippet=snippet,
        source_basename=source_basename,
        hub=op.hub,
        existing_content=nc.content,
    )
    if op.contested_by:
        from silica.kernel.contested import mark_contested
        new_content = mark_contested(new_content, op.contested_by)
    new_content = templates.ensure_ai_flag(new_content)
    DRIVER.overwrite(path, new_content)
    err = _verify_landed(op, path, new_content)
    if err:
        raise VerifyMismatchError(err)
    return {"path": path, "op": "patch", "success": True}


def _execute_overwrite(op: Op, path: str) -> dict:
    """Rewrite a whole note. Requires content.

    When the op carries base_content (the note as it was at op-build time) and
    the note on disk has changed since, the incoming content is written with a
    conflict callout prepended instead of stomping silently (ADR-0007 soft
    failure; charter UC6).
    """
    content = op.content
    if content is None:
        raise ValueError("Missing 'content' for overwrite operation")

    had_conflict = False
    try:
        prior: str | None = DRIVER.read_note(path).content
    except Exception:
        prior = None
    if op.base_content is not None:
        content, had_conflict = three_way_merge(op.base_content, prior, content)

    content = templates.ensure_system_floor(content, prior=prior)
    DRIVER.overwrite(path, content)
    err = _verify_landed(op, path, content)
    if err:
        raise VerifyMismatchError(err)
    result = {"path": path, "op": "overwrite", "success": True}
    if had_conflict:
        result["conflict"] = True
    return result


def _execute_delete(op: Op, path: str) -> dict:
    """Delete a note."""
    DRIVER.delete(path)
    err = _verify_deleted(path)
    if err:
        raise VerifyMismatchError(err)
    return {"path": path, "op": "delete", "success": True}


_DISPATCH = {
    OpType.write: _execute_write,
    OpType.patch: _execute_patch,
    OpType.overwrite: _execute_overwrite,
    OpType.delete: _execute_delete,
}


def execute_one(op: Op) -> dict:
    """Execute a single Op and return its success-result dict.

    Shared single-op entry point reused by execute_operations and by the
    interactive silica_patch_note tool. Raises ValueError on missing required
    params, a missing path, or an unknown op type — callers decide how to record
    the failure. Skip ops are a no-op success.
    """
    op_type = op.op

    if op_type == OpType.skip:
        return {"op": "skip", "success": True}

    path = op.touched_ref()
    if not path:
        raise ValueError("Missing 'path' parameter")

    handler = _DISPATCH.get(op_type)
    if handler is None:
        raise ValueError(f"Unknown operation type: {op_type}")

    return handler(op, path)


def execute_operations(ops: list[Op]) -> BulkResult:
    """Apply a batch of Ops, aggregating outcomes into a BulkResult.

    Each op is executed via execute_one; failures become FailedOp entries
    without aborting the batch. The return shape (ok/failed/results/total/
    successful) is the contract consumed by silica_bulk_write and the FSM.
    """
    results: list[dict] = []
    failed_ops: list[FailedOp] = []
    success_count = 0

    for idx, op in enumerate(ops):
        op_type = op.op
        path = op.touched_ref() or ""
        try:
            res = execute_one(op)
        except Exception as e:
            failed_ops.append(
                FailedOp(index=idx, path=path, op=op_type.value, error=str(e))
            )
            failure: dict = {"index": idx, "success": False, "error": str(e)}
            if path:
                failure["path"] = path
            results.append(failure)
            continue

        success_count += 1
        results.append({"index": idx, **res})

    return BulkResult(
        ok=len(failed_ops) == 0,
        failed=failed_ops,
        results=results,
        total=len(ops),
        successful=success_count,
    )
