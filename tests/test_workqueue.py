"""Tests for the WorkQueue + per-path lease (silica/planner/workqueue.py)."""
import json
import threading
import time

from silica.planner.workqueue import WorkQueue, WorkItem, path_lease


def test_enqueue_claim_complete_roundtrip():
    q = WorkQueue()
    item = WorkItem(kind="dedup", target_path="A.md", reason="borderline")
    q.enqueue(item)
    assert len(q) == 1

    claimed = q.claim(timeout=0.1)
    assert claimed is item
    q.complete(claimed, "done", {"patched": True})
    assert claimed.status == "done"
    assert q.summary() == {"done": 1}


def test_claim_returns_none_when_empty():
    q = WorkQueue()
    assert q.claim(timeout=0.05) is None


def test_close_blocks_further_enqueue_and_enables_drain():
    q = WorkQueue()
    q.enqueue(WorkItem(kind="refine", target_path="A.md"))
    q.close()
    assert q.closed
    # enqueue after close is a no-op
    q.enqueue(WorkItem(kind="refine", target_path="B.md"))
    assert len(q) == 1
    # not drained until the item is consumed
    assert not q.drained()
    it = q.claim(timeout=0.1)
    q.complete(it, "done")
    assert q.drained()


def test_persistence_to_run_dir(tmp_path):
    q = WorkQueue(run_dir=tmp_path)
    q.enqueue(WorkItem(kind="dedup", target_path="A.md"))
    assert (tmp_path / "workqueue.json").exists()


def test_path_lease_serialises_concurrent_writers():
    order: list[str] = []
    started = threading.Event()

    def worker(tag: str, hold: float):
        with path_lease("Notes/Same.md"):
            order.append(f"{tag}-enter")
            started.set()
            time.sleep(hold)
            order.append(f"{tag}-exit")

    t1 = threading.Thread(target=worker, args=("A", 0.2))
    t1.start()
    started.wait(timeout=1.0)
    t2 = threading.Thread(target=worker, args=("B", 0.0))
    t2.start()
    t1.join()
    t2.join()

    # B must not enter the same-path critical section before A exits.
    assert order == ["A-enter", "A-exit", "B-enter", "B-exit"]


def test_path_lease_independent_paths_do_not_block():
    # Different paths use different locks → both can hold simultaneously.
    both_in = threading.Barrier(2, timeout=1.0)

    def worker(path: str):
        with path_lease(path):
            both_in.wait()  # raises BrokenBarrierError if the other never arrives

    t1 = threading.Thread(target=worker, args=("A.md",))
    t2 = threading.Thread(target=worker, args=("B.md",))
    t1.start(); t2.start(); t1.join(); t2.join()
    # No exception ⇒ both entered concurrently.


def test_lease_key_normalises_md_and_case():
    # "A.md" and "a" should map to the same lease.
    acquired_second = threading.Event()

    def holder():
        with path_lease("A.md"):
            time.sleep(0.15)

    t = threading.Thread(target=holder)
    t.start()
    time.sleep(0.03)
    t2_start = time.monotonic()
    with path_lease("a"):
        waited = time.monotonic() - t2_start
        acquired_second.set()
    t.join()
    # Acquiring "a" had to wait for "A.md" to release.
    assert waited >= 0.1


def test_persist_is_atomic_under_concurrent_enqueue(tmp_path):
    """Two threads racing on enqueue must not produce a stale workqueue.json."""
    wq = WorkQueue(run_dir=tmp_path)
    barrier = threading.Barrier(2)

    def enqueue_many(items):
        barrier.wait()
        for item in items:
            wq.enqueue(WorkItem(kind="dedup", target_path=item, context={}, reason="test"))

    t1 = threading.Thread(target=enqueue_many, args=(["a.md", "b.md", "c.md"],))
    t2 = threading.Thread(target=enqueue_many, args=(["d.md", "e.md", "f.md"],))
    t1.start(); t2.start()
    t1.join(); t2.join()

    saved = json.loads((tmp_path / "workqueue.json").read_bytes())
    assert len(saved) == 6, f"Expected 6 items, got {len(saved)}: {[s['target_path'] for s in saved]}"
