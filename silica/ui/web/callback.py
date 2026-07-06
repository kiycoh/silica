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


def event_to_json(ev) -> dict | None:
    if isinstance(ev, LLMStreamEvent):
        return {"type": "delta", "kind": ev.chunk_type, "text": ev.content}
    if isinstance(ev, ToolStartEvent):
        return {"type": "tool_start", "name": _tool_verb(ev.name), "id": ev.call_id}
    if isinstance(ev, ToolCompleteEvent):
        return {"type": "tool_done", "name": _tool_verb(ev.name), "id": ev.call_id}
    if isinstance(ev, ToolErrorEvent):
        return {"type": "tool_error", "name": _tool_verb(ev.name), "id": ev.call_id, "error": ev.error}
    if isinstance(ev, BatchRunStartEvent):
        return {"type": "batch", "kind": ev.kind, "label": ev.label}
    return None  # ReasoningEvent / Thinking* — ignored in v1
