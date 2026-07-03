"""Typed worker shapes.

A WorkerProfile is the typed shape of a worker: its permitted tool subset, its
(optional) bounds factory, its iteration cap, its system prompt, and a parser that
turns the worker's final text + tool trace into a structured WorkerResult.
Built-in profiles live in profiles_builtin.py and are registered straight into
CAPABILITIES (silica/capabilities/__init__.py) — there is no second registry.
"""
from __future__ import annotations

from dataclasses import dataclass
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
class WorkerResult:
    status: str                  # "ok" | "deferred" | "error" | "no_op"
    output: Any = None           # profile-typed: digest | Op | applied-status
    detail: str = ""
