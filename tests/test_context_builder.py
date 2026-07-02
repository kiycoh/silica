"""Phase 2 tests — context_builder: build_context() contract."""
from __future__ import annotations

import json
from silica.kernel.context_builder import build_context


# ---------------------------------------------------------------------------
# Basic assembly
# ---------------------------------------------------------------------------

def test_empty_inputs_returns_empty():
    result = build_context("distill")
    assert result == ""


def test_payload_dict_serialised_to_json():
    payload = {"key": "value", "num": 42}
    result = build_context("distill", payload=payload)
    assert "Checkpoint: distill" in result
    assert json.dumps(payload, indent=2) in result


def test_payload_str_kept_verbatim():
    result = build_context("distill", payload="hello world")
    assert "hello world" in result


def test_ledger_digest_in_run_context_section():
    result = build_context("distill", ledger_digest="RUN abc12345 | inject")
    assert "Run Context" in result
    assert "RUN abc12345" in result


def test_substrate_in_related_notes_section():
    result = build_context("distill", substrate="[[Note A]] score=0.92")
    assert "Related Notes" in result
    assert "Note A" in result


def test_all_three_sources_present():
    result = build_context(
        "distill",
        payload={"batches": []},
        ledger_digest="RUN 00000001 | inject",
        substrate="[[Candidate]] score=0.85",
    )
    assert "Run Context" in result
    assert "Related Notes" in result
    assert "Checkpoint: distill" in result


# ---------------------------------------------------------------------------
# Ordering — ledger_digest < substrate < payload
# ---------------------------------------------------------------------------

def test_ordering_digest_before_payload():
    result = build_context("distill", payload="PAYLOAD", ledger_digest="DIGEST")
    assert result.index("DIGEST") < result.index("PAYLOAD")


def test_ordering_substrate_before_payload():
    result = build_context("distill", payload="PAYLOAD", substrate="SUBSTRATE")
    assert result.index("SUBSTRATE") < result.index("PAYLOAD")


def test_ordering_digest_before_substrate():
    result = build_context("distill", ledger_digest="DIGEST", substrate="SUBSTRATE")
    assert result.index("DIGEST") < result.index("SUBSTRATE")


# ---------------------------------------------------------------------------
# Empty / whitespace inputs are ignored
# ---------------------------------------------------------------------------

def test_empty_ledger_digest_omitted():
    result = build_context("distill", payload="PAYLOAD", ledger_digest="")
    assert "Run Context" not in result


def test_whitespace_substrate_omitted():
    result = build_context("distill", payload="PAYLOAD", substrate="   ")
    assert "Related Notes" not in result


# ---------------------------------------------------------------------------
# Pure function — no side effects
# ---------------------------------------------------------------------------

def test_multiple_calls_idempotent():
    args = dict(checkpoint_id="distill", payload={"x": 1}, ledger_digest="DIGEST")
    assert build_context(**args) == build_context(**args)


# ---------------------------------------------------------------------------
# ProgressLedger.is_checkpoint_done
# ---------------------------------------------------------------------------

def test_is_checkpoint_done_returns_output_ref(tmp_path):
    import silica.kernel.progress as _mod
    _mod._RUNS_DIR = tmp_path
    from silica.kernel.progress import ProgressLedger

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("validate", task_id="chunk_0_validate")
    p.mark_done("chunk_0_validate", output_ref="/tmp/ops.json", content_hash="abc123")

    result = p.is_checkpoint_done("chunk_0_validate", "abc123")
    assert result == "/tmp/ops.json"


def test_is_checkpoint_done_wrong_hash(tmp_path):
    import silica.kernel.progress as _mod
    _mod._RUNS_DIR = tmp_path
    from silica.kernel.progress import ProgressLedger

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("validate", task_id="chunk_0_validate")
    p.mark_done("chunk_0_validate", output_ref="/tmp/ops.json", content_hash="abc123")

    assert p.is_checkpoint_done("chunk_0_validate", "different_hash") is None


def test_is_checkpoint_done_pending_task(tmp_path):
    import silica.kernel.progress as _mod
    _mod._RUNS_DIR = tmp_path
    from silica.kernel.progress import ProgressLedger

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("validate", task_id="chunk_0_validate")

    assert p.is_checkpoint_done("chunk_0_validate", "abc123") is None


def test_is_checkpoint_done_unknown_task(tmp_path):
    import silica.kernel.progress as _mod
    _mod._RUNS_DIR = tmp_path
    from silica.kernel.progress import ProgressLedger

    p = ProgressLedger.new(mode="inject", inputs={})
    assert p.is_checkpoint_done("nonexistent_task", "abc123") is None


# ---------------------------------------------------------------------------
# ProgressLedger.run_dir
# ---------------------------------------------------------------------------

def test_run_dir_is_under_runs_dir(tmp_path):
    import silica.kernel.progress as _mod
    _mod._RUNS_DIR = tmp_path
    from silica.kernel.progress import ProgressLedger

    p = ProgressLedger.new(mode="inject", inputs={})
    assert p.run_dir == tmp_path / p.run_id


# ---------------------------------------------------------------------------
# mark_done with content_hash round-trip
# ---------------------------------------------------------------------------

def test_mark_done_content_hash_survives_roundtrip(tmp_path):
    import silica.kernel.progress as _mod
    _mod._RUNS_DIR = tmp_path
    from silica.kernel.progress import ProgressLedger

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("validate", task_id="chunk_0_validate")
    p.mark_done("chunk_0_validate", output_ref="/tmp/x.json", content_hash="deadbeef")
    p.save()

    p2 = ProgressLedger.load(p.run_id)
    t = p2.tasks[0]
    assert t.content_hash == "deadbeef"
    assert t.output_ref == "/tmp/x.json"
    assert t.status == "done"
