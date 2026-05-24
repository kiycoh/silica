"""Wrapped tools — L0 atomics with domain invariants (Golden Rules) baked in.

From SILICA.md §4.4:
  Wrapped tools enforce invariants in the toolset, not in the system prompt.
  - silica_move always updates wikilinks (graph-safe).
  - silica_delete refuses to delete if it loses density.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.driver.base import NoteRef, Txn
from silica.tools import tool


class MoveArgs(BaseModel):
    ref: str = Field(description="Name or path of the note to move")
    to: str = Field(description="Destination path")

@tool(MoveArgs, cls="wrapped")
def silica_move(ref: str, to: str) -> dict[str, Any]:
    """Move/rename a note safely. Obsidian updates all wikilinks (graph-safe)."""
    try:
        DRIVER.move(ref, to)
        return {"success": True, "moved": ref, "to": to}
    except Exception as e:
        return {"error": str(e)}


class DeleteArgs(BaseModel):
    ref: str = Field(description="Name or path of the note to delete")
    confirm: bool = Field(default=False, description="Explicit confirmation for density loss")

@tool(DeleteArgs, cls="wrapped")
def silica_delete(ref: str, confirm: bool = False) -> dict[str, Any]:
    """Delete a note. Requires confirmation if density is lost."""
    # TODO: In future phases, we could implement a density-loss check here.
    if not confirm:
        return {"error": "Anti-deletion policy: must pass confirm=True to acknowledge no density is lost."}
        
    try:
        DRIVER.delete(ref)
        return {"success": True, "deleted": ref}
    except Exception as e:
        return {"error": str(e)}


class SnapshotArgs(BaseModel):
    ops_json_path: str = Field(description="Path to operations JSON to extract refs for snapshot")

@tool(SnapshotArgs, cls="wrapped")
def silica_snapshot(ops_json_path: str) -> dict[str, Any]:
    """Snapshot the current versions of notes before they are modified."""
    import json
    
    try:
        with open(ops_json_path, 'r', encoding='utf-8') as f:
            ops_data = json.load(f)
            ops = ops_data.get("updates", ops_data)
            if not isinstance(ops, list):
                ops = [ops]
    except Exception as e:
        return {"error": f"Failed to load operations for snapshot: {e}"}
        
    # Extract unique note refs touched by the ops
    refs_to_snapshot = set()
    for op in ops:
        if op.get("name"):
            refs_to_snapshot.add(op["name"])
            
    refs = [NoteRef(name=name) for name in refs_to_snapshot]
    
    try:
        txn = DRIVER.snapshot_versions(refs)
        # We need a way to store the Txn so restore can use it.
        # For this prototype, we'll serialize it or store it globally.
        # A proper implementation would return the Txn structure to the FSM.
        return {
            "success": True,
            "txn_id": txn.id,
            "refs": [r.name for r in txn.refs],
            "versions": txn.versions,
            "_txn_obj": txn  # Hidden property for the orchestrator to use
        }
    except Exception as e:
        return {"error": f"Snapshot failed: {e}"}


class RestoreArgs(BaseModel):
    txn_id: str = Field(description="Transaction ID to restore")
    # In a real setup, we might pass the txn object directly if called programmatically,
    # but tools take primitives. The FSM can call the DRIVER directly if it holds the Txn.

@tool(RestoreArgs, cls="wrapped")
def silica_restore(txn_id: str) -> dict[str, Any]:
    """Rollback to a previous snapshot via history/sync restore. Not fully supported via CLI text alone."""
    return {"error": "Use the orchestrator's rollback mechanism instead, as it holds the Txn object."}
