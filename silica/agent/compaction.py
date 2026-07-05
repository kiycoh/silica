"""Context-window hygiene for the top-level agent loop.

Pure, I/O-free, LLM-free helpers. Two levers:
  • eager projection — write/gate tool results are summarised at emission so
    the fat JSON never enters the message history (the TUI event still gets it).
  • lazy compaction — old read-tool results are rewritten in place to one-line
    elision stubs once the context crosses a token budget, protecting the last
    K turns. Loss is recoverable: the stub names the call to re-issue.

Nothing here touches the network, the disk, or any global state.
"""
from __future__ import annotations

import json
from typing import Any

MIN_COLLAPSE_CHARS = 200  # don't collapse a body smaller than its own stub


def generic_projection(result: dict) -> str:
    """Conservative fallback stub for an eager tool with no `summarize`.

    Keeps scalars and short strings verbatim; elides long collections to a
    `<N items>` placeholder. Appends a hint that the full body is recoverable.
    """
    parts: list[str] = []
    for k, v in result.items():
        if isinstance(v, (list, dict)) and len(v) > 3:
            parts.append(f"{k}=<{len(v)} items>")
        elif isinstance(v, str) and len(v) > 80:
            parts.append(f"{k}=<{len(v)} chars>")
        else:
            parts.append(f"{k}={v}")
    return "; ".join(parts) + " ⟨↻ re-call to expand⟩"


def read_stub(tool_name: str, arguments: str) -> str:
    """Elision marker for a collapsed read result. Names the call to re-issue."""
    return (
        f"⟪silica: result elided to save context — "
        f"re-call {tool_name} with {arguments} to view again⟫"
    )


def eager_stub(tool: Any, result_str: str) -> str:
    """Project a write/gate tool's JSON result to its one-line summary.

    Uses the tool's own `summarize(dict)->str` when declared, else a generic
    projection. Non-JSON / non-dict payloads pass through unchanged so a tool
    that returns a bare string is never corrupted.
    """
    try:
        parsed = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return result_str
    if not isinstance(parsed, dict):
        return result_str
    summarize = getattr(tool, "summarize", None)
    if summarize is not None:
        try:
            return summarize(parsed)
        except Exception:
            return generic_projection(parsed)
    return generic_projection(parsed)


def compact_read_history(
    messages: list[dict],
    collapsed: set[int],
    prompt_tokens: int,
    budget: int,
    floor_turns: int,
    tools: dict,
) -> set[int]:
    """Collapse old read-tool results to one-line stubs when over budget.

    Strategy (i): when triggered, collapse *all* eligible lazy reads beyond the
    recency floor in one sweep — self-hysteresing, no per-message estimation.
    Eager/never/unknown tools and bodies <= MIN_COLLAPSE_CHARS are left alone.
    """
    if prompt_tokens <= budget:
        return collapsed

    # Map tool_call_id -> (name, arguments) from assistant messages; collect turn markers.
    id_to_call: dict[str, tuple[str, str]] = {}
    assistant_indices: list[int] = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            assistant_indices.append(i)
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                id_to_call[tc.get("id", "")] = (fn.get("name", ""), fn.get("arguments", "{}"))

    if len(assistant_indices) <= floor_turns:
        return collapsed  # not enough turns for anything to be "old"

    boundary = assistant_indices[-floor_turns]
    updated = set(collapsed)
    for i, m in enumerate(messages):
        if i >= boundary:
            break
        if i in updated or m.get("role") != "tool":
            continue
        if len(m.get("content") or "") <= MIN_COLLAPSE_CHARS:
            continue
        name, arguments = id_to_call.get(m.get("tool_call_id", ""), ("", "{}"))
        tool = tools.get(name) if name else None
        if tool is None or getattr(tool, "collapse", "lazy") != "lazy":
            continue
        m["content"] = read_stub(name, arguments)
        updated.add(i)
    return updated
