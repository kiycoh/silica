"""Shared litellm.completion mock-response builder.

Consolidates three near-identical MagicMock builders that used to live in
test_distiller_schema.py, test_organize_fsm.py, and
test_llm_structured_output.py. All three needed the same choice/message
shape (content, tool_calls, reasoning_content, reasoning, thinking_blocks,
finish_reason, usage tokens 10/20/30) — only the finish_reason default and
an optional parsed_obj→JSON convenience differed.
"""
from __future__ import annotations

import json

from pydantic import BaseModel
from unittest.mock import MagicMock


def litellm_mock_response(
    text: str | None = None,
    *,
    parsed_obj: BaseModel | None = None,
    finish_reason: str = "stop",
):
    """Build a litellm.completion-style mock response.

    `text` is the message content. `parsed_obj`, if given, is dumped to JSON
    and used as the content instead (convenience for structured-output
    tests). Exactly one of them is expected to carry the content.
    """
    message = MagicMock()
    message.content = text if text is not None else (
        json.dumps(parsed_obj.model_dump()) if parsed_obj is not None else None
    )
    message.tool_calls = None
    message.reasoning_content = None
    message.reasoning = None
    message.thinking_blocks = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return response
