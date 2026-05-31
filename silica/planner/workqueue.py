"""WorkQueue — the producer/consumer channel between the Injector and sub-agents.

The Injector (router) *produces* WorkItems as it commits batches; a pool of
leashed sub-agents *consumes* them concurrently on the worker model.  The router
never blocks on a sub-agent: it enqueues fire-and-forget and the Coordinator
drains/joins the queue only at the end of the run.

Concurrency safety has two layers:
  1. Temporal disjointness — items reference only notes from already-committed
     batches (or pre-existing vault notes), so the router is never mid-write on a
     sub-agent's target.
  2. Per-path lease — `path_lease()` serialises writes to the same note across
     sub-agents (and any lease-aware writer).  Both the Injector and the pool run
     as threads in one process, so an in-process lock registry is sufficient.

WorkItem kinds:
  "dedup"  — a borderline pair: append the candidate concept's unique info into an
             existing (larger) note.  context: {candidate, match_path, score, ...}
  "refine" — a freshly committed note flagged by lint: restyle without info loss.
  "orphan" — a note left orphaned by the graph gate: connect it.
"""
from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import orjson

# ---------------------------------------------------------------------------
# Per-path advisory lease (shared across the Injector and the sub-agent pool)
# ---------------------------------------------------------------------------

_LEASES: dict[str, threading.Lock] = {}
_LEASES_GUARD = threading.Lock()


def _lease_key(path: str) -> str:
    return (path or "").replace("\\", "/").removesuffix(".md").lower()


@contextmanager
def path_lease(path: str) -> Iterator[None]:
    """Serialise writes to a single vault note across threads.

    Advisory: only protects writers that opt in by acquiring the same lease.
    The sub-agent commit path and any lease-aware bulk writer should wrap their
    write in this context manager.
    """
    key = _lease_key(path)
    with _LEASES_GUARD:
        lock = _LEASES.setdefault(key, threading.Lock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# WorkItem
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    kind: str                       # "dedup" | "refine" | "orphan"
    target_path: str                # vault-rel note the sub-agent may write
    context: dict[str, Any] = field(default_factory=dict)
    reason: str = ""                # the FSM warning that produced this item
    id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "pending"         # pending | done | failed | rejected
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# WorkQueue
# ---------------------------------------------------------------------------

class WorkQueue:
    """Thread-safe producer/consumer queue with optional disk persistence."""

    def __init__(self, run_dir: str | Path | None = None):
        self._q: queue.Queue[WorkItem] = queue.Queue()
        self._items: list[WorkItem] = []        # every item ever enqueued (for inspect/resume)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        # Tolerate a non-path run_dir (e.g. a mocked progress object): persistence
        # is best-effort, so fall back to no-disk rather than raising.
        try:
            self._run_dir = Path(run_dir) if isinstance(run_dir, (str, Path)) else None
        except (TypeError, ValueError):
            self._run_dir = None

    # --- producer side ----------------------------------------------------

    def enqueue(self, item: WorkItem) -> None:
        """Add an item. No-op after close() (producers have finished)."""
        if self._closed.is_set():
            return
        with self._lock:
            self._items.append(item)
        self._q.put(item)
        self._persist()

    def close(self) -> None:
        """Signal that no further items will be produced."""
        self._closed.set()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    # --- consumer side ----------------------------------------------------

    def claim(self, *, timeout: float | None = 0.5) -> WorkItem | None:
        """Pop the next item, or None if none arrives within `timeout`."""
        try:
            return self._q.get(timeout=timeout) if timeout is not None else self._q.get_nowait()
        except queue.Empty:
            return None

    def complete(self, item: WorkItem, status: str, result: dict[str, Any] | None = None) -> None:
        """Mark an item finished and record the outcome."""
        with self._lock:
            item.status = status
            item.result = result
        self._q.task_done()
        self._persist()

    # --- draining ---------------------------------------------------------

    def drained(self) -> bool:
        """True when producers are done and every queued item is consumed."""
        return self._closed.is_set() and self._q.unfinished_tasks == 0

    # --- inspection / persistence ----------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def items(self) -> list[WorkItem]:
        with self._lock:
            return list(self._items)

    def summary(self) -> dict[str, int]:
        """Counts by status, for the final run report."""
        out: dict[str, int] = {}
        with self._lock:
            for it in self._items:
                out[it.status] = out.get(it.status, 0) + 1
        return out

    def _persist(self) -> None:
        if not self._run_dir:
            return
        try:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = [it.to_dict() for it in self._items]
            (self._run_dir / "workqueue.json").write_bytes(
                orjson.dumps(payload, option=orjson.OPT_INDENT_2)
            )
        except Exception:
            # Persistence is best-effort; never break the pipeline over it.
            pass
