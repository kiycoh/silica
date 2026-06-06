"""Sentinel-based drain tests for WorkQueue.

Verifies that N consumer threads all exit cleanly after close(), with no
thread leaks and no loss of items, when using the blocking claim() path.
"""
from __future__ import annotations

import threading
import time

from silica.planner.workqueue import WorkQueue, WorkItem


def _drain_thread(wq: WorkQueue, results: list) -> threading.Thread:
    """Return a started thread that drains wq via blocking claim()."""
    def _run():
        while True:
            item = wq.claim()
            if item is None:
                return
            wq.complete(item, "done")
            results.append(item.id)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def test_single_consumer_drains_and_exits():
    wq = WorkQueue()
    results = []
    t = _drain_thread(wq, results)

    for i in range(5):
        wq.enqueue(WorkItem(kind="dedup", target_path=f"N{i}.md"))
    wq.close()

    t.join(timeout=2.0)
    assert not t.is_alive(), "consumer thread must exit after close()"
    assert len(results) == 5
    assert wq.drained()


def test_n_consumers_all_exit_on_close():
    """N consumers receive the sentinel cascade and all exit cleanly."""
    N = 4
    wq = WorkQueue()
    results: list[str] = []
    lock = threading.Lock()

    def consumer():
        while True:
            item = wq.claim()
            if item is None:
                return
            time.sleep(0.005)
            wq.complete(item, "done")
            with lock:
                results.append(item.id)

    threads = [threading.Thread(target=consumer, daemon=True) for _ in range(N)]
    for t in threads:
        t.start()

    for i in range(10):
        wq.enqueue(WorkItem(kind="refine", target_path=f"N{i}.md"))
    wq.close()

    for t in threads:
        t.join(timeout=3.0)

    assert all(not t.is_alive() for t in threads), "all consumer threads must exit"
    assert len(results) == 10
    assert wq.drained()


def test_close_on_empty_queue_exits_immediately():
    """close() on an empty queue wakes consumers right away."""
    wq = WorkQueue()
    exited = threading.Event()

    def consumer():
        item = wq.claim()
        assert item is None
        exited.set()

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    wq.close()
    assert exited.wait(timeout=1.0), "consumer must exit within 1s of close()"
    t.join(timeout=1.0)


def test_sentinel_not_counted_in_pending():
    """drained() returns True after all real items are consumed even though
    the sentinel lingers in the internal queue."""
    wq = WorkQueue()
    item = WorkItem(kind="dedup", target_path="A.md")
    wq.enqueue(item)
    wq.close()

    claimed = wq.claim(timeout=0.5)
    assert claimed is item
    assert not wq.drained()   # item consumed but not completed
    wq.complete(claimed, "done")
    assert wq.drained()


def test_claim_timeout_backward_compat():
    """claim(timeout=<float>) still returns None on empty queue (legacy path)."""
    wq = WorkQueue()
    assert wq.claim(timeout=0.05) is None


def test_items_enqueued_before_and_after_close():
    """Items enqueued before close() are all consumed; post-close enqueue is no-op."""
    wq = WorkQueue()
    results: list[str] = []

    for i in range(3):
        wq.enqueue(WorkItem(kind="orphan", target_path=f"N{i}.md"))
    wq.close()
    wq.enqueue(WorkItem(kind="orphan", target_path="ignored.md"))  # no-op

    assert len(wq) == 3   # only 3 real items

    t = _drain_thread(wq, results)
    t.join(timeout=2.0)

    assert not t.is_alive()
    assert len(results) == 3
