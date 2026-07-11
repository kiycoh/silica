# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""RenderEvent -> JSON. The single source of truth for the wire event map.

Mirrors the table in docs/spec-gui-web.md. Reasoning/thinking events are
dropped in v1 (return None -> the callback skips them).
"""
from __future__ import annotations

from silica.agent.events import (
    BatchRunStartEvent,
    LLMStreamEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ToolStartEvent,
)
from silica.ui.renderer import _tool_verb  # same human verb the TUI shows

# Arg keys that name a note across the tool surface (read=name, write=path,
# related=note, mindmap=note_path, move/delete=ref). A small allowlist, not
# per-tool logic: missing one only omits a chip from the chat 'sources' footer,
# it never reports a wrong note.
_NOTE_KEYS = ("name", "path", "note", "note_path", "ref")


def _note_refs(args: dict) -> list[str]:
    refs = [args[k].strip() for k in _NOTE_KEYS
            if isinstance(args.get(k), str) and args[k].strip()]
    paths = args.get("note_paths")
    if isinstance(paths, list):
        refs += [p.strip() for p in paths if isinstance(p, str) and p.strip()]
    return refs


def event_to_json(ev) -> dict | None:
    if isinstance(ev, LLMStreamEvent):
        return {"type": "delta", "kind": ev.chunk_type, "text": ev.content}
    if isinstance(ev, ToolStartEvent):
        return {"type": "tool_start", "name": _tool_verb(ev.name), "id": ev.call_id,
                "notes": _note_refs(ev.args)}
    if isinstance(ev, ToolCompleteEvent):
        return {"type": "tool_done", "name": _tool_verb(ev.name), "id": ev.call_id}
    if isinstance(ev, ToolErrorEvent):
        return {"type": "tool_error", "name": _tool_verb(ev.name), "id": ev.call_id, "error": ev.error}
    if isinstance(ev, BatchRunStartEvent):
        return {"type": "batch", "kind": ev.kind, "label": ev.label}
    return None  # ReasoningEvent / Thinking* — ignored in v1
