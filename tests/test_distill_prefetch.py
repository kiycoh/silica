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
    # Default flipped 1 -> 3 after the 2026-07-18 k=1-vs-k=3 staleness A/B.
    from silica.config import CONFIG
    assert getattr(CONFIG, "distill_concurrency", None) == 3


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


import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _stub_fsm(n_chunks=4, file_of=None, k_done=()):
    """Minimal FSM stand-in for _distill_inputs/_prefetch_ahead."""
    file_of = file_of or {}
    fsm = SimpleNamespace()
    fsm._chunks = [{"schema_version": 1,
                    "batches": [{"inbox_file": "in.md", "concepts": [{"name": f"c{i}"}]}]}
                   for i in range(n_chunks)]
    fsm._chunk_flat_to_fi_ci = {i: (file_of.get(i, 0), i) for i in range(n_chunks)}
    fsm._current_file_idx = 0
    fsm._file_chunks = {}
    fsm.inbox_file = "in.md"
    fsm.context = {"file_0_language": "English"}
    fsm.target_dir = "Notes"
    fsm.hub = "[[Hub]]"
    fsm.manifest = MagicMock()
    fsm.manifest.titles.return_value = []
    fsm.progress = MagicMock()
    fsm.progress.digest.return_value = "digest"
    fsm.progress.started_at = "2026-07-18T00:00:00"
    fsm.progress.is_checkpoint_done.side_effect = (
        lambda task_id, h: "done.json" if any(f"c{d}_" in task_id for d in k_done) else None
    )
    fsm._chunk_task_id = lambda cap, idx=None: f"f0_c{idx}_{cap}"
    fsm._prefetcher = None
    return fsm


def test_distill_inputs_snapshot_shape():
    from silica.router.states.distill import _distill_inputs
    fsm = _stub_fsm()
    kw = _distill_inputs(fsm, 1)
    assert kw["target"] == "Notes" and kw["hub"] == "[[Hub]]"
    assert kw["ledger_digest"] == "digest"
    assert kw["language"] == "English"
    assert kw["steer_context"] is None
    assert kw["payload"]["batches"][0]["concepts"] == [{"name": "c1"}]


def test_prefetch_ahead_noop_at_k1():
    from silica.router.states import distill as d
    fsm = _stub_fsm()
    with patch.object(d.orch.CONFIG, "distill_concurrency", 1, create=True):
        d._prefetch_ahead(fsm, 0)
    assert fsm._prefetcher is None


def test_prefetch_ahead_dispatches_window_and_runs_collision_for_lookahead():
    from silica.router.states import distill as d
    fsm = _stub_fsm(n_chunks=5)
    seen = []
    with patch.object(d.orch.CONFIG, "distill_concurrency", 3, create=True), \
         patch.object(d, "run_distiller", side_effect=lambda **kw: {"updates": []}), \
         patch("silica.router.states.collision.collision_pass",
               side_effect=lambda f, j: seen.append(j)):
        d._prefetch_ahead(fsm, 1)
    # window = chunks 1,2,3; collision_pass early only for lookahead (2,3)
    assert seen == [2, 3]
    assert fsm.context.get("chunk_2_collision_done") and fsm.context.get("chunk_3_collision_done")
    assert 1 in fsm._prefetcher and 2 in fsm._prefetcher and 3 in fsm._prefetcher
    assert 4 not in fsm._prefetcher
    fsm._prefetcher.shutdown()


def test_prefetch_ahead_stops_at_file_boundary():
    from silica.router.states import distill as d
    fsm = _stub_fsm(n_chunks=4, file_of={0: 0, 1: 0, 2: 1, 3: 1})
    with patch.object(d.orch.CONFIG, "distill_concurrency", 3, create=True), \
         patch.object(d, "run_distiller", side_effect=lambda **kw: {"updates": []}), \
         patch("silica.router.states.collision.collision_pass"):
        d._prefetch_ahead(fsm, 0)
    assert 0 in fsm._prefetcher and 1 in fsm._prefetcher
    assert 2 not in fsm._prefetcher  # next file — never prefetched
    fsm._prefetcher.shutdown()


def test_prefetch_ahead_skips_checkpoint_done_chunks():
    from silica.router.states import distill as d
    fsm = _stub_fsm(n_chunks=3, k_done=(2,))
    with patch.object(d.orch.CONFIG, "distill_concurrency", 3, create=True), \
         patch.object(d, "run_distiller", side_effect=lambda **kw: {"updates": []}), \
         patch("silica.router.states.collision.collision_pass"):
        d._prefetch_ahead(fsm, 0)
    assert 0 in fsm._prefetcher and 1 in fsm._prefetcher
    assert 2 not in fsm._prefetcher  # already done in a prior run
    fsm._prefetcher.shutdown()
