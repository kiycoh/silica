"""CLEANUP-phase wiring of the provenance ledger (spec-hermes-coherence §3).

`_record_provenance` is the seam: a sibling projection to
`_log_nucleate_completion` at the same CLEANUP point, appending one
.silica/provenance.json record per (source, run) using data the FSM already
has — the sha256 computed once at RUN start (`_file_content_hashes`) and the
write/patch note paths already recorded in the manifest. Exercised directly
against a minimal fake FSM, same style as tests/test_nucleate_log.py.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

from silica.kernel.progress import RunManifestEntry
from silica.kernel.provenance import DEFAULT_PROVENANCE_FILENAME
from silica.router.states import finalize


def _fake_fsm(entries, run_id, content_hashes):
    return types.SimpleNamespace(
        manifest=types.SimpleNamespace(entries=entries),
        progress=types.SimpleNamespace(run_id=run_id),
        _file_content_hashes=content_hashes,
    )


def _entry(source_basename: str, op: str, path: str) -> RunManifestEntry:
    return RunManifestEntry(
        title=path, path=path, parent=None, cluster_id=-1,
        source_basename=source_basename, op=op,
    )


def test_appends_record_with_sha_from_precomputed_hash(tmp_vault):
    from silica.config import CONFIG

    entries = [
        _entry("lezione-03.md", "write", "Concepts/A"),
        _entry("lezione-03.md", "patch", "Concepts/B"),
        _entry("other.md", "write", "Concepts/Other"),  # different source — excluded
    ]
    fsm = _fake_fsm(entries, run_id="run-abc123", content_hashes=["sha-v1"])

    finalize._record_provenance(fsm, 0, "Inbox/lezione-03.md")

    raw = json.loads((Path(CONFIG.vault_path) / DEFAULT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert len(raw) == 1
    rec = raw[0]
    assert rec["source"] == "lezione-03.md"
    assert rec["sha256"] == "sha-v1"
    assert rec["run_id"] == "run-abc123"
    assert sorted(rec["notes"]) == ["Concepts/A", "Concepts/B"]


def test_renucleate_appends_second_record_not_overwrite(tmp_vault):
    from silica.config import CONFIG

    fsm1 = _fake_fsm(
        [_entry("a.md", "write", "N1"), _entry("a.md", "write", "N2")],
        run_id="run1", content_hashes=["sha1"],
    )
    finalize._record_provenance(fsm1, 0, "Inbox/a.md")

    fsm2 = _fake_fsm(
        [_entry("a.md", "write", "N1")],
        run_id="run2", content_hashes=["sha2"],
    )
    finalize._record_provenance(fsm2, 0, "Inbox/a.md")

    raw = json.loads((Path(CONFIG.vault_path) / DEFAULT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert len(raw) == 2
    assert [r["sha256"] for r in raw] == ["sha1", "sha2"]


def test_never_raises_on_broken_fsm(tmp_vault):
    """Best-effort: a fsm missing the expected attributes must not blow up CLEANUP."""
    broken_fsm = types.SimpleNamespace()
    finalize._record_provenance(broken_fsm, 0, "Inbox/x.md")  # must not raise


def test_survives_unwritable_provenance_store(tmp_vault, monkeypatch):
    """Best-effort: an I/O failure on write must not blow up CLEANUP."""
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)

    fsm = _fake_fsm(
        [_entry("a.md", "write", "N1")], run_id="run1", content_hashes=["sha1"],
    )
    finalize._record_provenance(fsm, 0, "Inbox/a.md")  # must not raise


def test_no_sha_available_skips_silently(tmp_vault):
    """Missing/empty precomputed hash (fi out of range) -> no record, no raise."""
    from silica.config import CONFIG

    fsm = _fake_fsm(
        [_entry("a.md", "write", "N1")], run_id="run1", content_hashes=[],
    )
    finalize._record_provenance(fsm, 0, "Inbox/a.md")

    assert not (Path(CONFIG.vault_path) / DEFAULT_PROVENANCE_FILENAME).exists()


def test_wired_into_handle_cleanup_on_archive(monkeypatch, tmp_vault):
    """handle_cleanup calls _record_provenance alongside _log_nucleate_completion
    when a file's last chunk archives successfully."""
    calls = []
    monkeypatch.setattr(
        finalize, "_record_provenance",
        lambda fsm, fi, source_file: calls.append((fi, source_file)),
    )
    monkeypatch.setattr(finalize, "_log_nucleate_completion", lambda *a, **k: None)
    monkeypatch.setattr("silica.tools.wrapped.silica_cleanup", lambda *a, **k: {"success": True})

    fsm = types.SimpleNamespace(
        _get_chunks_from_context_if_empty=lambda: None,
        _chunk_flat_to_fi_ci={0: (0, 0)},
        _current_chunk_idx=0,
        _progress_note=lambda *a, **k: None,
        _write_ledger_for_file=lambda *a, **k: None,
        _file_chunks={0: {"chunks": [{}], "source_file": "Inbox/a.md"}},
        progress=types.SimpleNamespace(tasks=[]),
        inbox_file="Inbox/a.md",
        context={},
        _undo_run_id=None,
        _run_inverses=[],
        _transition_success=lambda: None,
        _chunk_task_id=lambda *a: "cleanup",
    )

    finalize.handle_cleanup(fsm)

    assert calls == [(0, "Inbox/a.md")]
