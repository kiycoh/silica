from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from silica.agent.loop import run_agent
from silica.agent.constraints import AgentConstraints
from silica.tools import TOOLS, Tool
from pydantic import BaseModel


class _EmptyArgs(BaseModel):
    pass


def _install_tool(name, fn):
    TOOLS[name] = Tool(fn, name, "test tool", _EmptyArgs, "atomic")


def _resp(tool_calls=None, text="done"):
    return SimpleNamespace(
        assistant_message={"role": "assistant", "content": text},
        tool_calls=tool_calls or [],
        text=text,
        reasoning=None,
    )


def _tc(name, call_id):
    return SimpleNamespace(name=name, args={}, id=call_id)


def test_schemas_restricted_to_subset():
    captured = {}

    def fake_call_llm(model, messages, tools=None, cancel=None):
        captured["tools"] = tools
        return _resp(text="ok")

    _install_tool("allowed_tool", lambda: "ok")
    _install_tool("forbidden_tool", lambda: "nope")

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(
            messages=[{"role": "user", "content": "hi"}],
            model="router",
            constraints=AgentConstraints(tools=("allowed_tool",), model="worker", max_iterations=3),
        )

    names = {t["function"]["name"] for t in (captured["tools"] or [])}
    assert names == {"allowed_tool"}


def test_out_of_subset_dispatch_is_rejected():
    """A hallucinated call to a global-but-not-subset tool must NOT execute.

    run_agent appends tool results to the passed `messages` list in place, so we
    inspect that list directly — no module globals needed.
    """
    ran = {"forbidden": False}

    def forbidden():
        ran["forbidden"] = True
        return "should not run"

    _install_tool("forbidden_tool", forbidden)
    _install_tool("allowed_tool", lambda: "ok")

    calls = [0]

    def fake_call_llm(model, messages, tools=None, cancel=None):
        calls[0] += 1
        if calls[0] == 1:
            return _resp(tool_calls=[_tc("forbidden_tool", "c1")])
        return _resp(text="done")

    messages = [{"role": "user", "content": "hi"}]
    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(
            messages=messages,
            model="router",
            constraints=AgentConstraints(tools=("allowed_tool",), model="worker", max_iterations=5),
        )

    assert ran["forbidden"] is False
    # The tool result fed back to the model must be an error.
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert tool_msgs and any("error" in (m.get("content") or "") for m in tool_msgs)


def test_model_override_used():
    seen = {}

    def fake_call_llm(model, messages, tools=None, cancel=None):
        seen["model"] = model
        return _resp(text="ok")

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(
            messages=[{"role": "user", "content": "hi"}],
            model="router",
            constraints=AgentConstraints(tools=(), model="worker-model-x", max_iterations=2),
        )

    assert seen["model"] == "worker-model-x"


def test_iteration_cap_overridden():
    calls = [0]

    def fake_call_llm(model, messages, tools=None, cancel=None):
        calls[0] += 1
        # Always emit a tool call so the loop keeps going until the cap.
        return _resp(tool_calls=[_tc("allowed_tool", f"c{calls[0]}")])

    _install_tool("allowed_tool", lambda: "ok")

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        result = run_agent(
            messages=[{"role": "user", "content": "hi"}],
            model="router",
            constraints=AgentConstraints(tools=("allowed_tool",), model="worker", max_iterations=2),
        )

    assert calls[0] == 2
    assert result == "(silica: maximum iterations reached)"


def test_no_constraints_unchanged():
    def fake_call_llm(model, messages, tools=None, cancel=None):
        return _resp(text="normal")

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        result = run_agent(messages=[{"role": "user", "content": "hi"}], model="router")

    assert result == "normal"
