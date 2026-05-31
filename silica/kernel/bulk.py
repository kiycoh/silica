"""Vault write executor — the single authorised path for applying Ops.

Every write in Silica (FSM bulk write, deferred retry, interactive patch)
funnels through here. The per-op-type logic lives in small helpers; `execute_one`
is the shared single-op entry point and `execute_operations` is the batch loop
that aggregates results into a BulkResult.

Keeping one executor means any future change to how a write reaches disk
(e.g. atomic file writes at the DRIVER level) is inherited by every caller.
"""
from __future__ import annotations

from silica.driver import DRIVER
from silica.kernel import templates
from silica.kernel.ops import Op, OpType, FailedOp, BulkResult


def _execute_write(op: Op, path: str) -> dict:
    """Create a new note from the spoke template. Requires heading + hub."""
    heading = op.heading
    snippet = op.snippet or ""
    hub = op.hub

    if not heading or not hub:
        raise ValueError("Missing 'heading' or 'hub' parameter for write operation")

    content = templates.template_spoke(
        heading=heading,
        snippet=snippet,
        hub=hub,
        tags=op.tags,
        related=op.related,
        parent=op.parent,
    )
    DRIVER.create(path, content)
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

    new_content = templates.patch_snippet(
        heading=heading,
        snippet=snippet,
        source_basename=source_basename,
        hub=op.hub,
        existing_content=nc.content,
    )
    DRIVER.overwrite(path, new_content)
    return {"path": path, "op": "patch", "success": True}


def _execute_overwrite(op: Op, path: str) -> dict:
    """Rewrite a whole note. Requires content."""
    content = op.content
    if content is None:
        raise ValueError("Missing 'content' for overwrite operation")
    DRIVER.overwrite(path, content)
    return {"path": path, "op": "overwrite", "success": True}


def _execute_delete(op: Op, path: str) -> dict:
    """Delete a note."""
    DRIVER.delete(path)
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
