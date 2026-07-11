# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Wrapped tools — L0 atomics with domain invariants (Golden Rules) baked in.

From SILICA.md §4.4:
  Wrapped tools enforce invariants in the toolset, not in the system prompt.
  - silica_move always updates wikilinks (graph-safe).
  - silica_delete refuses to delete if it loses density.

C3 rollback strategy (ADR-009):
  - write ops   → InverseOp(delete_created, path)
  - patch/overwrite ops → InverseOp(restore_version, path, version=N)
  - Txn.inverses: list[InverseOp] replaces the ad-hoc created_paths field.

C3 clarification: silica_snapshot DOES NOT leak _txn_obj through the tool
registry. The orchestrator holds the Txn directly (it calls snapshot
programmatically and receives the return value). The tool is kept for CLI
discoverability only — the FSM bypasses it.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.driver.base import NoteRef, Txn
from silica.kernel.ops import InverseOp, InverseOpKind, OpType, Op
from silica.kernel.ops_io import parse_ops, load_ops
from silica.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# silica_move
# ---------------------------------------------------------------------------

class MoveArgs(BaseModel):
    ref: str = Field(description="Name or path of the note to move")
    to: str = Field(description="Destination path")

@tool(MoveArgs, cls="wrapped", collapse="eager")
def silica_move(ref: str, to: str) -> dict[str, Any]:
    """Move/rename a note safely. Obsidian updates all wikilinks (graph-safe)."""
    try:
        DRIVER.move(ref, to)
        return {"success": True, "moved": ref, "to": to}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# silica_delete
# ---------------------------------------------------------------------------

class DeleteArgs(BaseModel):
    ref: str = Field(description="Name or path of the note to delete")
    confirm: bool = Field(default=False, description="Explicit confirmation for density loss")

@tool(DeleteArgs, cls="wrapped", collapse="eager")
def silica_delete(ref: str, confirm: bool = False) -> dict[str, Any]:
    """Delete a note. Requires confirmation if density is lost."""
    if not confirm:
        return {"error": "Anti-deletion policy: must pass confirm=True to acknowledge no density is lost."}

    try:
        DRIVER.delete(ref)
        return {"success": True, "deleted": ref}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# build_txn — internal helper (not a tool)
# ---------------------------------------------------------------------------

def build_txn(ops_data: list[Op] | list[dict]) -> Txn:
    """Build a Txn with InverseOp entries before WRITE executes.

    Rollback strategies (C3):
      write   → delete_created(path)         — note didn't exist; undo = delete
      patch / overwrite → restore_version(path, prior_content=<full body>)
                          — note existed; undo = overwrite with saved content.
                          prior_content is the primary rollback path; version is
                          kept as a best-effort hint for backends that support it.
      delete  → recreate_deleted(path, prior_content=<full body>)
    """
    ops = parse_ops(ops_data)
    patch_refs: list[NoteRef] = []
    prior_contents: dict[str, str | None] = {}
    inverses: list[InverseOp] = []

    for op in ops:
        op_type = op.op
        path = op.touched_ref()
        if not path or op_type == OpType.skip:
            continue

        if op_type == OpType.write:
            # write's contract is "path MUST NOT exist", but nothing enforces it
            # at the FS boundary (DRIVER.create overwrites verbatim). If the path
            # already holds a note, undo must RESTORE it, not delete it — else
            # /revert turns an accidental clobber into data loss. Snapshot the
            # prior body and pick the inverse accordingly.
            try:
                prior = DRIVER.read_note(path).content
            except Exception:
                prior = None
            if prior:
                inverses.append(InverseOp(
                    kind=InverseOpKind.restore_version, path=path, prior_content=prior,
                ))
            else:
                inverses.append(InverseOp(kind=InverseOpKind.delete_created, path=path))
        elif op_type in (OpType.patch, OpType.overwrite):
            name = path.rsplit("/", 1)[-1].removesuffix(".md")
            ref = NoteRef(name=name, path=path)
            patch_refs.append(ref)
            # Read current content now (before WRITE) for content-based rollback.
            # This is more reliable than history:restore whose version numbering
            # shifts after each new write (position 1 becomes position 2, etc.).
            try:
                nc = DRIVER.read_note(ref)
                prior_contents[path] = nc.content
            except Exception as e:
                logger.warning("build_txn: could not read prior content for %s: %s", path, e)
                prior_contents[path] = None
        elif op_type == OpType.delete:
            name = path.rsplit("/", 1)[-1].removesuffix(".md")
            ref = NoteRef(name=name, path=path)
            try:
                nc = DRIVER.read_note(ref)
                prior_content = nc.content
            except Exception as e:
                # Can't snapshot the body we're about to delete → /revert will be
                # unable to recreate it. Surface it now, not silently at revert.
                logger.warning(
                    "build_txn: cannot snapshot %s before delete; /revert won't "
                    "recreate it: %s", path, e)
                prior_content = None
            inverses.append(InverseOp(
                kind=InverseOpKind.recreate_deleted,
                path=path,
                prior_content=prior_content
            ))

    # Snapshot versions as a best-effort hint (used by backends that support
    # history:restore; the primary rollback path is prior_content above).
    base_txn = DRIVER.snapshot_versions(patch_refs)

    for ref in patch_refs:
        key = ref.path or ref.name
        version = base_txn.versions.get(key)
        inverses.append(InverseOp(
            kind=InverseOpKind.restore_version,
            path=ref.path,
            version=version,
            prior_content=prior_contents.get(ref.path or ref.name),
        ))

    created_paths = [
        inv.path for inv in inverses
        if inv.kind == InverseOpKind.delete_created
    ]
    txn = Txn(
        id=base_txn.id,
        refs=patch_refs,
        versions=base_txn.versions,
        created_paths=created_paths,
        inverses=inverses,
    )
    return txn


# ---------------------------------------------------------------------------
# silica_snapshot
# ---------------------------------------------------------------------------

class SnapshotArgs(BaseModel):
    ops_json_path: str = Field(description="Path to validated operations JSON to snapshot before writing")

@tool(SnapshotArgs, cls="wrapped", collapse="eager", internal=True)
def silica_snapshot(ops_json_path: str) -> dict[str, Any]:
    """Snapshot the current state of notes before they are modified.

    Builds InverseOp entries (C3):
      - write ops   → delete_created(path)     — rollback by deleting the new note
      - patch / overwrite ops → restore_version(path, N) — rollback via history:restore

    The orchestrator holds the returned Txn object directly.
    The tool result is JSON-serialisable (no _txn_obj leak per addendum note).
    """
    try:
        ops = load_ops(ops_json_path)
    except Exception as e:
        return {"error": f"Failed to load operations for snapshot: {e}"}

    try:
        txn = build_txn(ops)
    except Exception as e:
        return {"error": f"Snapshot failed: {e}"}

    return {
        "success": True,
        "txn_id": txn.id,
        "refs": [r.name for r in txn.refs],
        "versions": txn.versions,
        "created_paths": txn.created_paths,
        "inverses": txn.inverses_serialized,
        # _txn_obj intentionally absent — orchestrator calls build_txn() directly
    }


# ---------------------------------------------------------------------------
# silica_restore (real tool, usable by YAML recipe engine at S3.3)
# ---------------------------------------------------------------------------

class RestoreArgs(BaseModel):
    txn_id: str = Field(description="Transaction ID to restore (for audit log only)")
    inverses: list[dict] = Field(description="InverseOp list from silica_snapshot result")

@tool(RestoreArgs, cls="wrapped", collapse="eager", internal=True)
def silica_restore(txn_id: str, inverses: list[dict]) -> dict[str, Any]:
    """Apply InverseOp list to rollback a transaction.

    Accepts the 'inverses' list produced by silica_snapshot — fully
    JSON-serialisable, no hidden Python objects.
    """
    errors: list[str] = []
    applied: list[str] = []

    for raw in inverses:
        try:
            inv = InverseOp(**raw)
        except Exception as e:
            errors.append(f"Invalid InverseOp {raw}: {e}")
            continue

        path = inv.path
        try:
            if inv.kind == InverseOpKind.delete_created:
                try:
                    DRIVER.delete(path)
                    applied.append(f"deleted_created:{path}")
                except Exception as e:
                    err_str = str(e).lower()
                    if "not found" in err_str or "no such file" in err_str:
                        applied.append(f"deleted_created:{path} (already_absent)")
                    else:
                        raise

            elif inv.kind == InverseOpKind.restore_version:
                if inv.prior_content is not None:
                    # Primary path: overwrite with captured content (reliable across
                    # backends; avoids Obsidian history position ambiguity).
                    DRIVER.overwrite(path, inv.prior_content)
                    applied.append(f"restored_content:{path}")
                elif inv.version is not None:
                    # Fallback: history:restore (only if prior_content unavailable)
                    name = path.rsplit("/", 1)[-1].removesuffix(".md")
                    ref = NoteRef(name=name, path=path)
                    DRIVER.restore(Txn(
                        id=txn_id,
                        refs=[ref],
                        versions={path: inv.version},
                        created_paths=[],
                    ))
                    applied.append(f"restored_version:{path}@{inv.version}")
                else:
                    logger.warning("restore_version: no prior_content or version for %s — skipped", path)

            elif inv.kind == InverseOpKind.recreate_deleted:
                if inv.prior_content is not None:
                    DRIVER.create(path, inv.prior_content)
                    applied.append(f"recreated_deleted:{path}")
                else:
                    errors.append(f"recreate_deleted missing prior_content for {path}")

        except Exception as e:
            errors.append(f"Inverse op {inv.kind} on {path} failed: {e}")
            logger.error("Rollback error: %s", e)

    return {
        "success": not errors,
        "txn_id": txn_id,
        "applied": applied,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# silica_cleanup (real tool for S3.3 YAML recipe)
# ---------------------------------------------------------------------------

class CleanupArgs(BaseModel):
    inbox_file: str = Field(description="Vault-relative path of the inbox file to archive")
    done_dir: str = Field(default="done", description="Destination folder for processed inbox files")

@tool(CleanupArgs, cls="wrapped", collapse="eager", internal=True)
def silica_cleanup(inbox_file: str, done_dir: str = "done") -> dict[str, Any]:
    """Move the inbox file to done/ after successful pipeline completion.

    C5: Only callable from DONE state — the orchestrator enforces this.
    """
    import os
    base_name = os.path.basename(inbox_file)
    target = f"{done_dir}/{base_name}"
    try:
        DRIVER.move(inbox_file, target)
        return {"success": True, "moved": inbox_file, "to": target}
    except Exception as e:
        return {"error": str(e)}
