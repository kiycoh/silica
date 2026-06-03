from __future__ import annotations

from unittest.mock import patch

from silica.tools.delegate_tool import silica_delegate
from silica.workers.profile import WorkerResult


def _fake_run_worker(task, *, config, cancel_token=None, profiles=None):
    # echo the goal so we can assert ordering + count
    return WorkerResult(status="ok", output={"goal": task.goal})


def test_fan_out_aggregates_in_order():
    tasks = [{"goal": f"t{i}", "inputs": {}} for i in range(5)]
    with patch("silica.tools.delegate_tool.run_worker", _fake_run_worker):
        out = silica_delegate(profile="reader", tasks=tasks, max_workers=4)

    assert out["summary"]["ok"] == 5
    goals = [r["output"]["goal"] for r in out["results"]]
    assert goals == ["t0", "t1", "t2", "t3", "t4"]   # task order preserved


def test_more_than_ten_tasks_are_chunked_not_rejected():
    tasks = [{"goal": f"t{i}", "inputs": {}} for i in range(23)]
    with patch("silica.tools.delegate_tool.run_worker", _fake_run_worker):
        out = silica_delegate(profile="reader", tasks=tasks, max_workers=10)

    assert out["summary"]["ok"] == 23
    assert len(out["results"]) == 23
    goals = [r["output"]["goal"] for r in out["results"]]
    assert goals == [f"t{i}" for i in range(23)]     # order preserved across waves


def test_empty_tasks_returns_empty():
    out = silica_delegate(profile="reader", tasks=[], max_workers=7)
    assert out["results"] == []
    assert out["summary"] == {}


def test_worker_error_is_captured_not_raised():
    def boom(task, *, config, cancel_token=None, profiles=None):
        return WorkerResult(status="error", detail="kaboom")

    with patch("silica.tools.delegate_tool.run_worker", boom):
        out = silica_delegate(profile="reader", tasks=[{"goal": "x", "inputs": {}}])

    assert out["summary"]["error"] == 1
