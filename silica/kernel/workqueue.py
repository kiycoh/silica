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

WorkItem kinds are owned by the capability registry — see
silica/capabilities/__init__.py (CAPABILITIES), the single dispatch table for
background work. tests/test_capability_seam.py enforces that every kind
produced anywhere has a registered capability.
"""
from __future__ import annotations

import queue
import threading
from collections import Counter
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
    # Runtime-only: not serialised. Set by caller to request early exit.
    cancel_token: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
        compare=False,
        hash=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_path": self.target_path,
            "context": self.context,
            "reason": self.reason,
            "id": self.id,
            "status": self.status,
            "result": self.result,
        }


# ---------------------------------------------------------------------------
# WorkQueue
# ---------------------------------------------------------------------------

_SENTINEL = object()  # poison pill — injected by close() to unblock consumers


class WorkQueue:
    """Thread-safe producer/consumer queue with optional disk persistence."""

    def __init__(self, run_dir: str | Path | None = None):
        self._q: queue.Queue = queue.Queue()
        self._items: list[WorkItem] = []        # every item ever enqueued (for inspect/resume)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._pending: int = 0                  # items enqueued but not yet complete
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
            self._pending += 1
        self._q.put(item)
        self._persist()

    def close(self) -> None:
        """Signal that no further items will be produced.

        Injects a sentinel into the queue so blocking consumers wake up and
        drain cleanly.  The sentinel cascades: each consumer that receives it
        rebroadcasts it before returning, waking the next consumer.
        """
        self._closed.set()
        self._q.put(_SENTINEL)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    # --- consumer side ----------------------------------------------------

    def claim(self, *, timeout: float | None = None) -> WorkItem | None:
        """Pop the next item.

        With ``timeout=None`` (default): blocks at OS level until an item
        arrives or the queue is closed (sentinel-based — no polling).
        Returns None only when the queue has been closed and is empty.

        With ``timeout=<float>``: legacy mode — returns None if no item
        arrives within that many seconds (backward-compatible with tests).
        """
        if timeout is None:
            item = self._q.get()
        else:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                return None

        if item is _SENTINEL:
            self._q.put(_SENTINEL)  # rebroadcast so the next consumer wakes up
            return None
        return item

    def complete(self, item: WorkItem, status: str, result: dict[str, Any] | None = None) -> None:
        """Mark an item finished and record the outcome."""
        with self._lock:
            item.status = status
            item.result = result
            self._pending -= 1
        self._persist()

    # --- draining ---------------------------------------------------------

    def drained(self) -> bool:
        """True when producers are done and every queued item is consumed."""
        with self._lock:
            return self._closed.is_set() and self._pending == 0

    # --- inspection / persistence ----------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def items(self) -> list[WorkItem]:
        with self._lock:
            return list(self._items)

    def summary(self) -> dict[str, int]:
        """Counts by status, for the final run report."""
        with self._lock:
            return dict(Counter(it.status for it in self._items))

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
