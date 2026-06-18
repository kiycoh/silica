"""Typed worker shapes and the PROFILES registry.

A WorkerProfile is the typed shape of a worker: its permitted tool subset, its
(optional) bounds factory, its iteration cap, its system prompt, and a parser that
turns the worker's final text + tool trace into a structured WorkerResult. The
registry mirrors the CAPABILITIES pattern: production uses the global PROFILES;
tests inject a fake dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class WorkerProfile:
    name: str
    tools: tuple[str, ...]
    bounds_factory: Callable[..., Any] | None  # None ⇒ read-only profile (Phase A)
    max_iterations: int
    system_prompt: str
    result_parser: Callable[[str, list[dict]], "WorkerResult"]


@dataclass
class WorkerTask:
    profile: str                 # WorkerProfile.name
    goal: str                    # natural-language sub-task
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerResult:
    status: str                  # "ok" | "deferred" | "error" | "no_op"
    output: Any = None           # profile-typed: digest | Op | applied-status
    detail: str = ""


# Global registry — populated by silica/capabilities/profiles_builtin.py.
PROFILES: dict[str, WorkerProfile] = {}
