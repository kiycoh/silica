# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Process-global worker-concurrency ceiling.

One BoundedSemaphore gates the scarce resource — the worker-model LLM call —
so nested fan-outs (a router fan-out inside a Coordinator sub-agent) can never
amplify past the real backend limit. Gating the *call*, not the pool, keeps
per-layer pools deadlock-free (workers are leaves; no permit-while-holding-permit).
See ADR-0004.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from silica.config import CONFIG

_lock = threading.Lock()
_sem: threading.BoundedSemaphore | None = None
_sem_size: int | None = None


def _get_sem() -> threading.BoundedSemaphore:
    global _sem, _sem_size
    size = max(1, int(getattr(CONFIG, "worker_max_concurrent", 4)))
    with _lock:
        if _sem is None or _sem_size != size:
            _sem = threading.BoundedSemaphore(size)
            _sem_size = size
        return _sem


@contextmanager
def worker_slot() -> Iterator[None]:
    """Acquire one global worker permit for the duration of the block."""
    sem = _get_sem()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def reset_for_tests() -> None:
    """Drop the cached semaphore so the next acquire re-reads CONFIG.

    Only safe at startup or between tests (never while a slot is held).
    """
    global _sem, _sem_size
    with _lock:
        _sem = None
        _sem_size = None
