"""CLEANUP derives the run-level final_status (A24).

no_ops is a whole-run property: it holds only when nothing committed across the
entire run. A prior all-skip chunk must not pin the run to no_ops when a later
chunk writes real notes. Two chunks in the file group with ci=0 keeps this off
the archive branch so the status derivation is exercised in isolation.
"""
from __future__ import annotations

import types

from silica.router.states import finalize


def _cleanup_fsm(context):
    return types.SimpleNamespace(
        _get_chunks_from_context_if_empty=lambda: None,
        _chunk_flat_to_fi_ci={0: (0, 0)},
        _current_chunk_idx=0,
        _progress_note=lambda *a, **k: None,
        _write_ledger_for_file=lambda *a, **k: None,
        # two chunks, ci=0 -> not last chunk -> archive branch skipped
        _file_chunks={0: {"chunks": [{}, {}], "source_file": "Inbox/a.md"}},
        progress=types.SimpleNamespace(tasks=[]),
        inbox_file="Inbox/a.md",
        context=context,
        _undo_run_id=None,
        _run_inverses=[],
        _transition_success=lambda: None,
        _chunk_task_id=lambda *a: "cleanup",
    )


def test_no_ops_when_run_had_no_ops():
    fsm = _cleanup_fsm({})
    finalize.handle_cleanup(fsm)
    assert fsm.context["final_status"] == "no_ops"


def test_chunk_with_ops_lifts_prior_no_ops():
    # earlier all-skip chunk left provisional no_ops; the run had ops elsewhere
    fsm = _cleanup_fsm({"final_status": "no_ops", "run_had_ops": True})
    finalize.handle_cleanup(fsm)
    assert fsm.context["final_status"] == "Success"


def test_partial_failure_wins_over_success():
    fsm = _cleanup_fsm({"run_had_ops": True, "has_partial_failure": True})
    finalize.handle_cleanup(fsm)
    assert fsm.context["final_status"] == "partial"
