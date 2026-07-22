# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""commit_ops — the reusable validate→snapshot→write→lint micro-gate.

Lets bounded sub-agents commit their proposed ops through the *same*
deterministic safety machinery the
main pipeline uses: structural validation, an inverse-op snapshot, the bulk
write, a lint gate, and automatic rollback on failure.

Two guarantees layered on top for sub-agents:
  * CapabilityBounds filters ops to the permitted envelope *before* anything is written;
  * every touched note is held under `path_lease` for the whole snapshot→write→lint
    window, so a concurrent writer (another sub-agent, or a lease-aware Injector
    write) cannot interleave on the same note.
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import ExitStack, suppress
from typing import Any, Callable

import orjson

from silica.kernel.ops import Op, OpType
from silica.kernel.ops_io import load_ops
from silica.kernel.paths import silica_tmp_dir

logger = logging.getLogger(__name__)


def _write_ops_tmp(ops: list[Op]) -> str:
    path = str(silica_tmp_dir() / f"{uuid.uuid4().hex}.json")
    payload = [op.model_dump() for op in ops]
    with open(path, "wb") as f:
        f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return path


def commit_derived(rel: str, content: str) -> dict[str, Any]:
    """Commit ONE machine-derived note: lease → snapshot → write → lint → rollback.

    Derived artifacts (code wiki) have their ground truth outside the vault
    (the deterministic digest), so the nucleate concept validators do not apply:
    the hub fallback would inject a junk hub note for the target dir, and the
    anti-info-loss bounds would reject a legitimately shrunk regen. The
    transactional guarantees stay: prior content is captured before the write
    and restored when the OFM lint fails. Creates or overwrites as needed —
    the single write path for first writes and regens alike.

    Returns {"status": committed | rolled_back | error, "reason": ...}.
    """
    from silica.driver import DRIVER
    from silica.kernel.templates import ensure_system_floor
    from silica.kernel.workqueue import path_lease
    from silica.tools.composed import silica_lint

    with path_lease(rel):
        try:
            prior: str | None = DRIVER.read_note(rel).content
        except Exception:
            prior = None
        content = ensure_system_floor(content, prior=prior)
        try:
            if prior is None:
                DRIVER.create(rel, content)
            else:
                DRIVER.overwrite(rel, content)
        except Exception as e:
            return {"status": "error", "reason": str(e)}
        lr = silica_lint(rel, op_type="overwrite")
        if not lr.get("success", True):
            reason = "; ".join(str(e) for e in lr.get("errors") or []) or "lint failed"
            try:
                if prior is None:
                    DRIVER.delete(rel)
                else:
                    DRIVER.overwrite(rel, prior)
            except Exception as e:
                logger.error("commit_derived rollback failed for %s: %s", rel, e)
                return {"status": "error", "reason": f"{reason}; rollback failed: {e}"}
            return {"status": "rolled_back", "reason": reason}
    return {"status": "committed"}


def commit_ops(
    ops: list[Op],
    *,
    target_dir: str = "",
    hub: str | None = None,
    bounds: Any | None = None,
    read_note: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Commit `ops` through leash → validate → snapshot → write → lint (+rollback).

    Returns a result dict with a `status` of:
      committed | rolled_back | no_ops | error
    plus `committed` (count), `txn_id`, and `rejected_by_bounds` (ops the bounds dropped).
    """
    from silica.tools.composed import silica_validate_ops, silica_bulk_write, silica_lint
    from silica.tools.wrapped import silica_snapshot, silica_restore
    from silica.kernel.workqueue import path_lease

    rejected_by_bounds: list[dict] = []
    if bounds is not None:
        ops, rejected_by_bounds = bounds.enforce(ops, read_note=read_note)

    actionable = [o for o in ops if o.op != OpType.skip]
    if not actionable:
        return {"status": "no_ops", "committed": 0, "rejected_by_bounds": rejected_by_bounds}

    ops_path = _write_ops_tmp(ops)
    try:
        # Validate (C4: overwrites ops_path with the validated, coerced ops).
        vres = silica_validate_ops(ops_path, payload_paths=[], target_dir=target_dir, hub=hub or "")
        if "error" in vres:
            return {"status": "error", "error": vres["error"], "rejected_by_bounds": rejected_by_bounds}
        if vres.get("validated_count", 0) == 0:
            return {"status": "no_ops", "committed": 0, "rejected_by_bounds": rejected_by_bounds, "validate": vres}

        touched = [
            (p, op.op.value if op.op else "", op.hub or "")
            for op in load_ops(ops_path)
            if (p := op.touched_ref()) and op.op is not OpType.skip
        ]
        lease_paths = sorted({p for p, _, _ in touched})

        with ExitStack() as stack:
            # Deterministic lock ordering (sorted) avoids deadlock between sub-agents.
            for p in lease_paths:
                stack.enter_context(path_lease(p))

            sres = silica_snapshot(ops_path)
            if "error" in sres:
                return {"status": "error", "error": sres["error"], "rejected_by_bounds": rejected_by_bounds}
            txn_id = sres["txn_id"]
            inverses = sres.get("inverses", [])

            def _rollback(reason: dict) -> dict:
                try:
                    silica_restore(txn_id=txn_id, inverses=inverses)
                except Exception as e:
                    logger.error("commit_ops rollback failed: %s", e)
                    reason["rollback_error"] = str(e)
                reason.update({"status": "rolled_back", "committed": 0, "txn_id": txn_id,
                               "rejected_by_bounds": rejected_by_bounds})
                return reason

            wres = silica_bulk_write(ops_path)
            if "error" in wres:
                return _rollback({"error": wres["error"]})
            if wres.get("successful", 0) == 0 and wres.get("total", 0) > 0:
                return _rollback({"error": "all write ops failed"})

            lint_failures: list[dict] = []
            for p, op_type, h in touched:
                lr = silica_lint(p, op_type=op_type, hub=h)
                if not lr.get("success", True):
                    lint_failures.append({"path": p, "errors": lr.get("errors")})
            if lint_failures:
                return _rollback({"lint_failures": lint_failures})

        return {
            "status": "committed",
            "committed": wres.get("successful", len(lease_paths)),
            "txn_id": txn_id,
            "rejected_by_bounds": rejected_by_bounds,
        }
    finally:
        # Staging file is scoped to this call; nothing references it after return.
        # Unlink here — commit_ops runs outside the FSM's _cleanup_tmp lifecycle.
        with suppress(OSError):
            os.unlink(ops_path)
