"""Coordinator — runs the Injector and a pool of bounded sub-agents concurrently.

Producer/consumer model:
  * the InjectorFSM (router model) runs on the calling thread and *produces*
    WorkItems as it commits batches — it never blocks on a sub-agent;
  * a ThreadPoolExecutor of BoundedSubAgents (worker model) *consumes* the queue
    in parallel, writing only through their CapabilityBounds + the commit_ops micro-gate;
  * after the Injector finishes, the Coordinator closes the queue and joins the
    pool (drains remaining items) before returning an aggregated status.

When no items are ever produced this collapses to single-FSM behaviour with
negligible overhead.
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
        from silica.kernel.workqueue import WorkQueue
        from silica.router.warning_ledger import WarningLedger
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
        """Normalized set of notes currently orphaned, from the driver graph.

        Fix B: orphan status is just in-degree==0 — it needs no Louvain/PageRank.
        ``DRIVER.orphans()`` reads the maintained graph directly (~sub-ms), where
        the old ``compute_report`` rebuilt the whole report (~3.8s at 10k notes)
        and this fires up to twice per run (enqueue + reverify).
        """
        from silica.agent.bounds import _norm_path
        try:
            from silica.driver import DRIVER
            return {_norm_path(o.path) for o in DRIVER.orphans()}
        except Exception as e:
            logger.debug("orphan recompute failed (non-fatal): %s", e)
            return set()

    def _orphan_candidates(self, path: str, k: int = 3) -> list[dict]:
        """Related notes for an orphan, via the relatedness facade.

        Fuses embeddings + co-occurrence (RRF): pure candidate generation, no
        cosine thresholding, so the facade is a clean drop-in. Still produces
        link targets when the embedding index is empty — the co-occurrence leg
        carries the routing on its own.
        """
        from silica.agent.bounds import _norm_path
        try:
            from silica.config import CONFIG
            from silica.kernel.cooccurrence import get_cooccur_store
            from silica.kernel.embed import get_store
            from silica.kernel.relatedness import related_notes

            key = _norm_path(path)
            results = related_notes(
                key,
                embed_store=get_store(),
                cooccur_store=get_cooccur_store(lang=CONFIG.cooccurrence_lang),
                k=k,
            )
            return [{"name": r.name, "path": r.path} for r in results]
        except Exception as e:
            logger.debug("orphan candidate lookup failed (non-fatal): %s", e)
            return []

    def _enqueue_orphan_repairs(self, wq: Any, result: dict) -> None:
        from silica.agent.bounds import _norm_path
        from silica.kernel.workqueue import WorkItem

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
        """One consumer thread — delegates to the shared ``consume`` loop in
        silica/agent/subagent.py (the same engine ad-hoc batches run on)."""
        from silica.agent.subagent import BoundedSubAgent, consume

        consume(wq, BoundedSubAgent(self.config), self._stop)
