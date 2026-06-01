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

RenderEvent = ToolProgressEvent | ReasoningEvent | ThinkingStartEvent | ThinkingEndEvent | LLMStreamEvent

