from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from silica.config import CONFIG
import silica.agent.concurrency as conc


def test_worker_slot_enforces_global_ceiling(monkeypatch):
    monkeypatch.setattr(CONFIG, "worker_max_concurrent", 2, raising=False)
    conc.reset_for_tests()

    active = 0
    peak = 0
    lock = threading.Lock()

    def worker(_):
        nonlocal active, peak
        with conc.worker_slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(8)))

    assert peak <= 2, f"semaphore allowed {peak} concurrent, cap was 2"


def test_worker_slot_releases_on_exception(monkeypatch):
    monkeypatch.setattr(CONFIG, "worker_max_concurrent", 1, raising=False)
    conc.reset_for_tests()

    try:
        with conc.worker_slot():
            raise ValueError("boom")
    except ValueError:
        pass

    # If the slot leaked, this second acquire would block forever; guard with a
    # short-lived thread.
    acquired = threading.Event()

    def grab():
        with conc.worker_slot():
            acquired.set()

    t = threading.Thread(target=grab)
    t.start()
    t.join(timeout=1.0)
    assert acquired.is_set(), "worker_slot did not release on exception"
