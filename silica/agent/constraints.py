"""Optional constraints that turn run_agent into a bounded worker loop.

Carries only the three generic dials (tools, model, iteration cap). The leash is
deliberately NOT here — write safety lives inside the write tool / apply_op, so
run_agent stays domain-agnostic (Rune 1 / ADR set).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConstraints:
    tools: tuple[str, ...]          # subset of TOOLS the loop may expose + dispatch
    model: str | None = None        # override the model arg when set
    max_iterations: int | None = None  # override the default safety cap when set
