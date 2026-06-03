"""LeashedSubAgent — a small, tightly-bounded worker that runs on the worker model.

A leashed sub-agent consumes one WorkItem at a time and dispatches it to the
capability registered under ``item.kind``. Each capability (see
``silica/capabilities/``) is a self-contained ``run(item, config) -> dict``
function that writes only through its Leash + the commit_ops micro-gate. The
sub-agent runs on the *worker* model (role="worker"), concurrently with the
Injector.

``LeashedSubAgent`` itself is just the dispatch seam: it owns the worker config,
catches capability errors so a single item never crashes the pool, and returns a
status dict. Adding a behaviour means adding a capability module + one registry
line — never editing this file.
"""
from __future__ import annotations

import logging
from typing import Any

from silica.config import CONFIG
from silica.capabilities import CAPABILITIES, Capability
from silica.planner.workqueue import WorkItem

logger = logging.getLogger(__name__)


def run_subagent_batch(
    items: list[WorkItem],
    config: Any = CONFIG,
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run a batch of WorkItems through leashed sub-agents in parallel.

    Used by the ad-hoc /dedup and /refine commands (out of the inject pipeline).
    LeashedSubAgent is stateless beyond its config, so one instance is safely
    shared across threads; commit_ops serialises same-note writes via path_lease.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not items:
        return {"items": 0, "summary": {}, "results": []}

    mw = max(1, int(max_workers or getattr(config, "subagent_max_concurrent", 3)))
    agent = LeashedSubAgent(config)

    with ThreadPoolExecutor(max_workers=mw, thread_name_prefix="subagent") as ex:
        paired = list(ex.map(lambda it: (it, agent.handle(it)), items))

    summary: dict[str, int] = {}
    for _it, res in paired:
        s = res.get("status", "done")
        summary[s] = summary.get(s, 0) + 1
    return {
        "items": len(items),
        "summary": summary,
        "results": [{"target": it.target_path, **res} for it, res in paired],
    }


class LeashedSubAgent:
    """Dispatches a WorkItem to the capability registered under its kind."""

    def __init__(
        self,
        config: Any = CONFIG,
        capabilities: dict[str, Capability] | None = None,
    ):
        self.config = config
        # Injected registry defaults to the global one, so production is
        # unchanged while tests supply a fake registry without mutating state.
        self._capabilities = capabilities if capabilities is not None else CAPABILITIES

    def handle(self, item: WorkItem) -> dict[str, Any]:
        run = self._capabilities.get(item.kind)
        if run is None:
            return {"status": "skipped", "reason": f"no capability for kind '{item.kind}'"}
        try:
            return run(item, self.config)
        except Exception as e:  # never let a sub-agent crash the pool
            logger.warning("Capability '%s' error on item %s: %s", item.kind, item.id, e)
            return {"status": "error", "error": str(e)}
