"""Shared building blocks for capabilities.

A capability is a plain ``run(item, config) -> dict`` function living in its own
module. The behaviours share a small skeleton — emit a feedback phase, read the
target note (or skip), check the cancel token — so those steps live here as free
functions each ``run()`` composes, keeping the per-behaviour variation explicit
rather than hidden in a base-class template method.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from silica.kernel.workqueue import WorkItem

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


class NoteContent(BaseModel):
    """The structured result of a note-rewriting decision (refine / enrich)."""

    content: str = ""


def emit_feedback(item: WorkItem, phase: str, detail: str = "") -> None:
    """Publish a WorkFeedbackEvent to the global bus (best-effort)."""
    from silica.agent.bus import BUS
    from silica.agent.events import WorkFeedbackEvent
    BUS.publish("work/feedback", WorkFeedbackEvent(item.id, item.kind, phase, detail))


def read_or_skip(path: str) -> tuple[str | None, dict | None]:
    """Read a note body. Returns ``(body, None)`` on success, or
    ``(None, {"status": "skipped", ...})`` if the note is unreadable."""
    from silica.driver import DRIVER
    try:
        return DRIVER.read_note(path).content or "", None
    except Exception as e:
        return None, {"status": "skipped", "reason": f"unreadable: {e}"}


def load_prompt(name: str) -> str:
    path = _PROMPT_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""
