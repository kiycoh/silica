# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Single-note tools — fast-path create/patch with /undo checkpoints.

No temp-file + bulk_write round-trip: these are the interactive-edit
counterparts of the batch pipeline in silica.tools.pipeline.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool
from silica.kernel.ops import Op, OpType


class PatchNoteArgs(BaseModel):
    name: str = Field(description="Name or vault-relative path of the note to patch")
    heading: str = Field(description="Concept/section heading the snippet is filed under")
    snippet: str = Field(description="Distilled body text to append to the note")
    source_basename: str = Field(description="Provenance: source filename this snippet derives from")
    hub: str | None = Field(default=None, description="Optional [[Hub]] to link in frontmatter if missing")

@tool(PatchNoteArgs, cls="composed", collapse="eager")
def silica_patch_note(
    name: str,
    heading: str,
    snippet: str,
    source_basename: str,
    hub: str | None = None,
) -> dict[str, Any]:
    """Append a snippet under a heading in a single EXISTING note — the fast path
    for interactive edits.

    To create a new note use silica_write_note; for nucleating whole documents
    into many notes use silica_run_injector. Every successful patch is
    checkpointed and can be reverted with /undo.
    """
    from silica.kernel.bulk import execute_one
    from silica.kernel.checkpoints import get_checkpoint_store

    # Resolve the note and capture its pre-patch content for the undo floor.
    try:
        nc = DRIVER.read_note(name)
    except Exception as e:
        return {"error": f"Failed to read note '{name}': {e}"}

    path = nc.ref.path or name
    prior_content = nc.content

    op = Op(
        op=OpType.patch,
        heading=heading,
        source_basename=source_basename,
        path=path,
        snippet=snippet,
        hub=hub,
    )

    try:
        result = execute_one(op)
    except Exception as e:
        return {"error": f"Failed to patch '{name}': {e}"}

    # Record the resulting on-disk content as a restore point.
    checkpoint_depth = None
    try:
        new_content = DRIVER.read_note(path).content
        checkpoint_depth = get_checkpoint_store().push(path, prior_content, new_content)
    except Exception:
        # A patch that succeeded must not be reported as failed just because
        # the undo bookkeeping hiccuped; undo is best-effort.
        pass

    return {**result, "note": name, "path": path, "checkpoint_depth": checkpoint_depth}


class WriteNoteArgs(BaseModel):
    path: str = Field(description="Vault-relative path for the new note (e.g. 'Computer Science/Computer Vision.md')")
    content: str = Field(description="Full markdown content including YAML frontmatter")

@tool(WriteNoteArgs, cls="composed", collapse="eager")
def silica_write_note(path: str, content: str) -> dict[str, Any]:
    """Create a new note in the vault — the fast path for single-note creation.

    Fails if the note already exists: use silica_patch_note to append to an
    existing note, or silica_run_injector for multi-note nucleation with
    quality gates and rollback. The creation is checkpointed and can be
    reverted with /undo.
    """
    from silica.kernel.checkpoints import get_checkpoint_store

    # The fs backend's create() overwrites silently — enforce the documented
    # contract here, at the tool seam, so no backend can clobber an existing note.
    try:
        DRIVER.read_note(path)
    except Exception:
        pass  # missing note — the happy path
    else:
        return {"error": f"Note '{path}' already exists: use silica_patch_note to modify it."}

    try:
        ref = DRIVER.create(path, content)
    except Exception as e:
        return {"error": f"Failed to create note '{path}': {e}"}

    checkpoint_depth = None
    try:
        checkpoint_depth = get_checkpoint_store().push(path, "", content)
    except Exception:
        pass

    return {"op": "write", "success": True, "path": ref.path or path, "checkpoint_depth": checkpoint_depth}
