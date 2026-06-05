"""Coordinator — runs the Injector and a pool of bounded sub-agents concurrently.

Producer/consumer model:
  * the InjectorFSM (router model) runs on the calling thread and *produces*
    WorkItems as it commits batches — it never blocks on a sub-agent;
  * a ThreadPoolExecutor of BoundedSubAgents (worker model) *consumes* the queue
    in parallel, writing only through their CapabilityBounds + the commit_ops micro-gate;
  * after the Injector finishes, the Coordinator closes the queue and joins the
    pool (drains remaining items) before returning an aggregated status.

When `subagents_enabled` is false (or no items are ever produced) this collapses
to the legacy single-FSM behaviour with negligible overhead.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from silica.config import CONFIG

logger = logging.getLogger(__name__)


def _log_work_event(event: Any) -> None:
    logger.debug("work event: %s", event)


class Coordinator:
    def __init__(
        self,
        inbox_files: list[str] | None = None,
        target_dir: str = "",
        hub: str | None = None,
        *,
        resume_run_id: str | None = None,
        config: Any = CONFIG,
        cancel_token: "threading.Event | None" = None,
    ):
        # Lazy import keeps construction patchable at the orchestrator boundary
        # and avoids import-time coupling.
        from silica.router.orchestrator import InjectorFSM

        self.config = config
        self._stop = cancel_token if cancel_token is not None else threading.Event()
        self.fsm = InjectorFSM(
            inbox_files=inbox_files,
            target_dir=target_dir,
            hub=hub,
            resume_run_id=resume_run_id,
        )

    def run(self) -> dict[str, Any]:
        if not getattr(self.config, "subagents_enabled", True):
            return self.fsm.run()

        from silica.planner.workqueue import WorkQueue
        from silica.planner.warnings import WarningLedger
        from silica.agent.bus import BUS

        run_dir = getattr(self.fsm.progress, "run_dir", None)
        wq = WorkQueue(run_dir=run_dir)
        self.fsm.work_queue = wq
        self.fsm.warning_ledger = WarningLedger(run_dir=run_dir)
        BUS.subscribe("work/*", _log_work_event)

        max_workers = max(1, int(getattr(self.config, "subagent_max_concurrent", 3)))
        pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="subagent")
        futures = [pool.submit(self._consume, wq) for _ in range(max_workers)]

        try:
            result = self.fsm.run()  # producer; runs to completion on this thread
            # End-of-run repair: only the warnings still unresolved after the whole
            # run (incl. AUTOLINK/BACKLINK) become priority work for the sub-agents.
            self._enqueue_orphan_repairs(wq, result)
        except BaseException:
            # Interrupt or unexpected crash: signal consumers to exit immediately
            # so pool.shutdown() doesn't block on in-flight LLM calls.
            self._stop.set()
            raise
        finally:
            wq.close()
            if self._stop.is_set():
                # Best-effort: cancel queued futures and return without joining threads
                # that are mid-LLM-call. The process is shutting down anyway.
                pool.shutdown(wait=False, cancel_futures=True)
            else:
                pool.shutdown(wait=True)

        # Re-verify: recompute orphans after the repairs committed.
        self._reverify_orphans(result)

        # Surface any consumer-thread crashes (handle() already swallows per-item
        # errors, so this only catches unexpected pool failures).
        for f in futures:
            exc = f.exception()
            if exc:
                logger.warning("sub-agent consumer crashed: %s", exc)

        result["subagents"] = wq.summary()
        logger.info("Coordinator: sub-agent outcomes %s", result["subagents"])
        return result

    # --- end-of-run orphan resolution -------------------------------------

    def _current_orphans(self) -> set[str]:
        """Normalized set of notes currently orphaned in the target folder."""
        from silica.agent.bounds import _norm_path
        try:
            from silica.kernel.graph_report import compute_report
            report = compute_report(folder=getattr(self.fsm, "target_dir", "") or "")
            return {_norm_path(o) for o in report.orphans}
        except Exception as e:
            logger.debug("orphan recompute failed (non-fatal): %s", e)
            return set()

    def _orphan_candidates(self, path: str, k: int = 3) -> list[dict]:
        """Nearest semantic neighbours of an orphan note (existing notes only)."""
        from silica.agent.bounds import _norm_path
        try:
            from silica.kernel.embed import EmbedStore
            store = EmbedStore()
            key = _norm_path(path)
            vec = store.get_vec(key)
            if not vec:
                return []
            return [
                {"name": m.get("name", m.get("path", "")), "path": m.get("path", "")}
                for m in store.cosine_top_k(vec, k=k, exclude={key})
            ]
        except Exception as e:
            logger.debug("orphan candidate lookup failed (non-fatal): %s", e)
            return []

    def _enqueue_orphan_repairs(self, wq: Any, result: dict) -> None:
        from silica.agent.bounds import _norm_path
        from silica.planner.workqueue import WorkItem

        ledger = getattr(self.fsm, "warning_ledger", None)
        if ledger is None or len(ledger) == 0:
            return

        warned = ledger.paths("orphan")
        current = self._current_orphans()
        # Residual = warned notes that are STILL orphaned after the full run.
        residual = [p for p in warned if not current or _norm_path(p) in current]

        enqueued = 0
        for path in residual:
            candidates = self._orphan_candidates(path)
            if not candidates:
                continue
            wq.enqueue(WorkItem(
                kind="orphan",
                target_path=path,
                context={"candidates": candidates, "hub": getattr(self.fsm, "hub", None)},
                reason="residual_orphan",
            ))
            enqueued += 1

        result["orphan_warnings"] = {
            "warned": len(warned),
            "residual": len(residual),
            "enqueued": enqueued,
        }

    def _reverify_orphans(self, result: dict) -> None:
        """After repairs commit, recompute how many warned notes are still orphaned."""
        ow = result.get("orphan_warnings")
        if not ow or not ow.get("enqueued"):
            return
        ledger = getattr(self.fsm, "warning_ledger", None)
        if ledger is None:
            return
        from silica.agent.bounds import _norm_path
        warned_keys = {_norm_path(p) for p in ledger.paths("orphan")}
        still = self._current_orphans() & warned_keys
        ow["residual_after"] = len(still)
        logger.info(
            "Coordinator: orphan repair — %d warned, %d enqueued, %d still orphaned after",
            ow.get("warned", 0), ow.get("enqueued", 0), len(still),
        )

    def _consume(self, wq: Any) -> None:
        """One consumer thread: claim → handle → complete until the queue closes.

        Blocks at OS level on ``wq.claim()`` — no polling.  The sentinel
        injected by ``wq.close()`` cascades through all consumers so they
        all wake and exit cleanly.  ``_stop`` is checked after each item so
        a producer crash causes pending items to be marked cancelled rather
        than dispatched to the sub-agent.
        """
        from silica.agent.subagent import BoundedSubAgent
        from silica.agent.bus import BUS
        from silica.agent.events import WorkCancelledEvent

        agent = BoundedSubAgent(self.config)
        while True:
            item = wq.claim()           # blocks; no timeout, no polling
            if item is None:
                return                  # sentinel received — queue fully drained
            if self._stop.is_set() or item.cancel_token.is_set():
                wq.complete(item, "cancelled")
                BUS.publish(
                    "work/cancelled",
                    WorkCancelledEvent(item.id, item.kind, "pre_handle"),
                )
                continue
            res = agent.handle(item)
            wq.complete(item, res.get("status", "done"), res)
