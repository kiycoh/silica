from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from silica.agent.loop import run_agent
from silica.tools import TOOLS, Tool


class _Args(BaseModel):
    pass


def _resp(tool_calls=None, text="done"):
    return SimpleNamespace(
        assistant_message={"role": "assistant", "content": text},
        tool_calls=tool_calls or [],
        text=text,
        reasoning=None,
    )


def _tc(name, call_id):
    return SimpleNamespace(name=name, args={}, id=call_id)


def test_tool_receives_cancel_token_when_declared():
    seen = {}

    def long_tool(cancel_token=None):
        seen["token"] = cancel_token
        return "ok"

    TOOLS["long_tool"] = Tool(long_tool, "long_tool", "doc", _Args, "composed")

    token = threading.Event()
    calls = [0]

    def fake_call_llm(model, messages, tools=None):
        calls[0] += 1
        if calls[0] == 1:
            return _resp(tool_calls=[_tc("long_tool", "c1")])
        return _resp(text="done")

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(messages=[{"role": "user", "content": "hi"}], model="m", cancel_token=token)

    assert seen["token"] is token


def test_tool_without_cancel_token_is_unaffected():
    seen = {}

    def plain_tool():
        seen["ran"] = True
        return "ok"

    TOOLS["plain_tool"] = Tool(plain_tool, "plain_tool", "doc", _Args, "atomic")

    calls = [0]

    def fake_call_llm(model, messages, tools=None):
        calls[0] += 1
        if calls[0] == 1:
            return _resp(tool_calls=[_tc("plain_tool", "c1")])
        return _resp(text="done")

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            cancel_token=threading.Event(),
        )

    assert seen.get("ran") is True
