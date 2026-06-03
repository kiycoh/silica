"""Tool registry — the @tool decorator, TOOLS dict, and JSON-schema generation.

This is the contract layer between the LLM and Silica's toolset.
Every tool is a function decorated with @tool(ParamsModel, cls="atomic|composed|wrapped").
The decorator auto-registers the tool in the global TOOLS dict.
The LLM receives the JSON-schema of each tool's ParamsModel as its function definition.

Design (from SILICA.md §8.4):
  - Pydantic BaseModel for params → validates input AND generates JSON-schema
  - Three tool classes: atomic (1:1 CLI), composed (promoted scripts), wrapped (Golden Rule enforced)
  - TOOLS dict is the single source of truth for tool dispatch
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Tool:
    """Metadata and executor for a single registered tool."""

    __slots__ = ("fn", "name", "description", "params_model", "cls")

    def __init__(
        self,
        fn: Callable,
        name: str,
        description: str,
        params_model: type[BaseModel],
        cls: str,
    ):
        self.fn = fn
        self.name = name
        self.description = description
        self.params_model = params_model
        self.cls = cls  # "atomic" | "composed" | "wrapped"

    def json_schema(self) -> dict:
        """Return the OpenAI-compatible function schema for this tool."""
        # Build a clean JSON Schema, removing pydantic's 'title' noise
        schema = self.params_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def run(self, _cancel_token: Any = None, **kwargs: Any) -> str:
        """Validate args via pydantic, then execute the tool function.

        `_cancel_token` is injected by the agent loop and forwarded to the
        underlying function only when that function declares a `cancel_token`
        parameter. It is never part of the params model / JSON schema.
        Always returns a JSON string — either the result or an error.
        """
        try:
            validated = self.params_model(**kwargs)
            call_kwargs = validated.model_dump()
            if _cancel_token is not None:
                sig = inspect.signature(self.fn)
                if "cancel_token" in sig.parameters:
                    call_kwargs["cancel_token"] = _cancel_token
            result = self.fn(**call_kwargs)
            # Ensure result is always a string
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.exception("Tool %s execution error: %s", self.name, e)
            return json.dumps(
                {"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False
            )


# Global tool registry — the single source of truth
TOOLS: dict[str, Tool] = {}


def tool(params_model: type[BaseModel], cls: str = "atomic"):
    """Decorator that registers a function as a Silica tool.

    Usage:
        class ReadNoteArgs(BaseModel):
            name: str

        @tool(ReadNoteArgs, cls="atomic")
        def silica_read_note(name: str):
            '''Read a vault note by name (wikilink-style resolution).'''
            return DRIVER.read_note(name)
    """

    def decorator(fn: Callable) -> Callable:
        tool_name = fn.__name__
        tool_desc = fn.__doc__ or ""
        TOOLS[tool_name] = Tool(fn, tool_name, tool_desc.strip(), params_model, cls)
        logger.debug("Registered tool: %s (class=%s)", tool_name, cls)
        return fn

    return decorator
