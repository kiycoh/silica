"""silica_delegate — fan a list of worker tasks out to parallel workers.

Built on delegate() (ThreadPoolExecutor). delegate() hard-stops at >10 tasks, so
this tool chunks internally into waves of <=10; the real concurrency ceiling is
the global worker semaphore inside run_worker, not the pool size. Returns
aggregated results in submission order plus a status summary.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from silica.tools import tool
from silica.config import CONFIG
from silica.agent.delegate import delegate
from silica.workers.profile import WorkerTask, WorkerResult
from silica.workers.runtime import run_worker
import silica.workers.profiles_builtin  # noqa: F401  (registers built-in profiles)


_WAVE = 10  # delegate() hard cap


class DelegateArgs(BaseModel):
    profile: str = Field(description="WorkerProfile name, e.g. 'reader' or 'router'")
    tasks: list[dict] = Field(
        description="List of {goal: str, inputs: dict} task specs for the workers"
    )
    max_workers: int = Field(default=7, description="Parallel workers per wave (cap 10)")


def _result_to_dict(r: WorkerResult) -> dict:
    return {"status": r.status, "output": r.output, "detail": r.detail}


@tool(DelegateArgs, cls="composed")
def silica_delegate(profile: str, tasks: list[dict], max_workers: int = 7) -> dict:
    """Fan a list of worker tasks out to parallel workers; return aggregated results.

    Each task is {goal, inputs}. Tasks beyond 10 are processed in successive waves
    (delegate() caps a single fan-out at 10). Concurrency is bounded globally by
    the worker semaphore. Returns {"results": [...], "summary": {status: count}}.
    """
    if not tasks:
        return {"results": [], "summary": {}}

    def run_one(spec: dict) -> dict:
        task = WorkerTask(
            profile=profile,
            goal=spec.get("goal", ""),
            inputs=spec.get("inputs", {}) or {},
        )
        return _result_to_dict(run_worker(task, config=CONFIG))

    results: list[dict] = []
    for start in range(0, len(tasks), _WAVE):
        wave = tasks[start : start + _WAVE]
        results.extend(delegate(wave, run_one, max_workers=min(max_workers, _WAVE)))

    summary: dict[str, int] = {}
    for r in results:
        s = r.get("status", "ok")
        summary[s] = summary.get(s, 0) + 1

    return {"results": results, "summary": summary}
