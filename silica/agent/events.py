from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class ToolStartEvent:
    name: str
    args: dict[str, Any]
    call_id: str
    iteration: int

@dataclass(slots=True)
class ToolCompleteEvent:
    name: str
    args: dict[str, Any]
    call_id: str
    result: str          # già stringa, come da Tool.run()
    duration_s: float
    iteration: int

@dataclass(slots=True)
class ToolErrorEvent:
    name: str
    call_id: str
    error: str
    iteration: int

ToolProgressEvent = ToolStartEvent | ToolCompleteEvent | ToolErrorEvent

@dataclass(slots=True)
class ReasoningEvent:
    text: str
    iteration: int

@dataclass(slots=True)
class ThinkingStartEvent:
    iteration: int

@dataclass(slots=True)
class ThinkingEndEvent:
    iteration: int

@dataclass(slots=True)
class LLMStreamEvent:
    chunk_type: str
    content: str
    iteration: int

@dataclass(slots=True)
class BatchRunStartEvent:
    run_id: str
    kind: str    # "refine" | "enrich"
    label: str   # display label, e.g. "Concepts/ML"
    total: int   # total number of batches

RenderEvent = ToolProgressEvent | ReasoningEvent | ThinkingStartEvent | ThinkingEndEvent | LLMStreamEvent | BatchRunStartEvent


# --- work-queue events (published on silica.agent.bus.BUS) -------------------

@dataclass(slots=True)
class WorkFeedbackEvent:
    item_id: str    # WorkItem.id
    kind: str       # "dedup" | "refine" | "orphan" | "enrich"
    phase: str      # "reading" | "calling_llm" | "committing"
    detail: str = ""


@dataclass(slots=True)
class WorkCompleteEvent:
    item_id: str
    kind: str
    status: str     # "done" | "no_merge" | "no_change" | "skipped" | "error"
    duration_s: float


@dataclass(slots=True)
class WorkCancelledEvent:
    item_id: str
    kind: str
    phase: str      # where cancellation was detected

