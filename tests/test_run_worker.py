from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from silica.config import SilicaConfig
from silica.tools import TOOLS, Tool
from silica.capabilities.profile import WorkerProfile, WorkerTask, WorkerResult
from silica.capabilities.runtime import run_worker


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


def _profile(**over):
    base = dict(
        name="reader",
        tools=("probe_tool",),
        bounds_factory=None,
        max_iterations=4,
        system_prompt="be brief",
        result_parser=lambda text, trace: WorkerResult(
            status="ok", output={"text": text, "trace_len": len(trace)}
        ),
    )
    base.update(over)
    return WorkerProfile(**base)


def test_run_worker_uses_worker_model_and_returns_structured_result():
    TOOLS["probe_tool"] = Tool(lambda: "probe-ok", "probe_tool", "doc", _Args, "atomic")
    seen = {}

    calls = [0]

    def fake_call_llm(model, messages, tools=None):
        seen["model"] = model
        seen["tool_names"] = {t["function"]["name"] for t in (tools or [])}
        calls[0] += 1
        if calls[0] == 1:
            return _resp(tool_calls=[_tc("probe_tool", "c1")])
        return _resp(text="final digest")

    cfg = SilicaConfig()
    cfg.worker_model = "worker/model-x"

    profiles = {"reader": _profile()}

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        result = run_worker(
            WorkerTask(profile="reader", goal="gather", inputs={}),
            config=cfg,
            cancel_token=None,
            profiles=profiles,
        )

    assert isinstance(result, WorkerResult)
    assert result.status == "ok"
    assert result.output["text"] == "final digest"
    assert result.output["trace_len"] == 1          # one tool call captured
    assert seen["model"] == "worker/model-x"         # worker model, not router
    assert seen["tool_names"] == {"probe_tool"}      # subset enforced


def test_run_worker_unknown_profile_is_error():
    cfg = SilicaConfig()
    result = run_worker(
        WorkerTask(profile="nope", goal="x", inputs={}),
        config=cfg,
        cancel_token=None,
        profiles={},
    )
    assert result.status == "error"


def test_run_worker_honours_cancel_token():
    token = threading.Event()
    token.set()

    def fake_call_llm(model, messages, tools=None):
        return _resp(text="should not be reached")

    cfg = SilicaConfig()
    cfg.worker_model = "worker/model-x"

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        result = run_worker(
            WorkerTask(profile="reader", goal="x", inputs={}),
            config=cfg,
            cancel_token=token,
            profiles={"reader": _profile()},
        )

    # Token is pre-set, so run_agent short-circuits with its cancel sentinel before
    # any LLM call; the default _profile parser surfaces that text in output["text"].
    assert "cancelled" in str(result.output).lower()
