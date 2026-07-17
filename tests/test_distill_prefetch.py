"""DistillPrefetcher unit tests (Tier 1 speed: k-way distillation)."""
import time

from silica.router.prefetch import DistillPrefetcher


def test_submit_and_pop_returns_result():
    p = DistillPrefetcher(max_workers=2)
    p.submit(0, lambda: {"updates": [0]})
    fut = p.pop(0)
    assert fut is not None
    assert fut.result(timeout=5) == {"updates": [0]}
    p.shutdown()


def test_submit_is_idempotent_even_after_pop():
    # Guard against re-dispatch on the steer-retry DELEGATE re-entry:
    # once an idx has been submitted, submit() must be a permanent no-op.
    calls = []
    p = DistillPrefetcher(max_workers=2)
    p.submit(1, lambda: calls.append("a"))
    p.pop(1).result(timeout=5)
    p.submit(1, lambda: calls.append("b"))
    assert p.pop(1) is None
    assert 1 in p
    time.sleep(0.1)
    assert calls == ["a"]
    p.shutdown()


def test_pop_unknown_idx_returns_none():
    p = DistillPrefetcher(max_workers=1)
    assert p.pop(42) is None
    assert 42 not in p
    p.shutdown()


def test_shutdown_cancels_pending():
    p = DistillPrefetcher(max_workers=1)
    p.submit(0, lambda: time.sleep(2))
    p.submit(1, lambda: {"updates": []})  # queued behind the sleeper
    p.shutdown()  # must not block for the full sleep
    assert p.pop(1) is None  # futures map cleared by shutdown
    assert 1 in p            # but submitted-history survives (no re-dispatch)


def test_config_knob_default():
    from silica.config import CONFIG
    assert getattr(CONFIG, "distill_concurrency", None) == 1


def test_handle_collision_honors_prefetch_marker():
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from silica.router.states import collision as coll

    fsm = MagicMock()
    fsm._current_chunk_idx = 2
    fsm.context = {"chunk_2_collision_done": True}
    with patch.object(coll, "collision_pass") as cp:
        coll.handle_collision(fsm)
    cp.assert_not_called()
    fsm._transition_success.assert_called_once()
    assert "chunk_2_collision_done" not in fsm.context  # marker consumed
