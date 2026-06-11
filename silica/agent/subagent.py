"""BoundedSubAgent — a small, tightly-bounded worker that runs on the worker model.

A bounded sub-agent consumes one WorkItem at a time and dispatches it to the
capability registered under ``item.kind``. Each capability (see
``silica/capabilities/``) is a self-contained ``run(item, config) -> dict``
function that writes only through its CapabilityBounds + the commit_ops micro-gate.
The sub-agent runs on the *worker* model (role="worker"), concurrently with the
Injector.

``BoundedSubAgent`` itself is just the dispatch seam: it owns the worker config,
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


def consume(wq: Any, agent: "BoundedSubAgent", stop: Any = None) -> None:
    """One consumer thread: claim → handle → complete until the queue closes.

    THE consumer loop — shared by the Coordinator's in-run pool and the ad-hoc
    ``run_subagent_batch`` path, so cancel semantics, bookkeeping, and bus
    events live in exactly one place.  Blocks at OS level on ``wq.claim()`` —
    no polling; the sentinel injected by ``wq.close()`` cascades through all
    consumers so they all wake and exit cleanly.  ``stop`` (optional Event) is
    checked before each item so a producer crash or user cancel causes pending
    items to be marked cancelled rather than dispatched.
    """
    from silica.agent.bus import BUS
    from silica.agent.events import WorkCancelledEvent

    while True:
        item = wq.claim()               # blocks; no timeout, no polling
        if item is None:
            return                      # sentinel received — queue fully drained
        if (stop is not None and stop.is_set()) or item.cancel_token.is_set():
            wq.complete(item, "cancelled", {"status": "cancelled", "reason": "cancel_token"})
            BUS.publish(
                "work/cancelled",
                WorkCancelledEvent(item.id, item.kind, "pre_handle"),
            )
            continue
        res = agent.handle(item)
        wq.complete(item, res.get("status", "done"), res)


def run_subagent_batch(
    items: list[WorkItem],
    config: Any = CONFIG,
    *,
    max_workers: int | None = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Run a batch of WorkItems through leashed sub-agents in parallel.

    Used by the ad-hoc /dedup, /refine, /enrich commands and silica_delegate
    (out of the inject pipeline).  A pre-closed WorkQueue drained by the shared
    ``consume`` loop — the exact engine the Coordinator runs in-pipeline — so
    both paths get identical cancel/bookkeeping semantics.  BoundedSubAgent is
    stateless beyond its config, so one instance is safely shared across
    threads; commit_ops serialises same-note writes via path_lease.
    """
    from concurrent.futures import ThreadPoolExecutor

    from silica.planner.workqueue import WorkQueue

    if not items:
        return {"items": 0, "summary": {}, "results": []}

    if cancel_token is not None:
        for it in items:
            it.cancel_token = cancel_token

    mw = max(1, int(max_workers or getattr(config, "subagent_max_concurrent", 3)))
    agent = BoundedSubAgent(config)

    wq = WorkQueue()
    for it in items:
        wq.enqueue(it)
    wq.close()

    with ThreadPoolExecutor(max_workers=mw, thread_name_prefix="subagent") as ex:
        futures = [ex.submit(consume, wq, agent, cancel_token) for _ in range(mw)]
    for f in futures:
        exc = f.exception()
        if exc:
            logger.warning("sub-agent consumer crashed: %s", exc)

    return {
        "items": len(items),
        "summary": wq.summary(),
        "results": [
            {"target": it.target_path, **(it.result or {"status": it.status})}
            for it in items
        ],
    }


class BoundedSubAgent:
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
