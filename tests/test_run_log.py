"""Tests for silica.kernel.run_log — the human-readable <vault>/log.md journal.

Pure kernel helper: format the ingest-completion event, append idempotently
per run_id, and read back the tail for vault-map injection.
"""
from __future__ import annotations

from pathlib import Path

from silica.kernel.run_log import (
    DEFAULT_LOG_FILENAME,
    append_log_line,
    format_ingest_event,
    tail_log,
)


def test_format_ingest_event_matches_brief_shape():
    assert format_ingest_event("lezione-03.md", 7, 3, 2) == (
        "ingest `lezione-03.md` → 7 nuove, 3 patch, 2 deferred"
    )


def test_append_creates_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    log_path = vault / DEFAULT_LOG_FILENAME
    assert not log_path.exists()

    ok = append_log_line(
        "ingest `a.md` → 1 nuove, 0 patch, 0 deferred",
        "deadbeef1234",
        vault_path=str(vault),
    )

    assert ok is True
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert content.startswith("- ")
    assert "ingest `a.md`" in content
    assert "run deadbeef" in content


def test_two_appends_two_lines_in_order(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    append_log_line(
        "ingest `a.md` → 1 nuove, 0 patch, 0 deferred",
        "runidone1234",
        vault_path=str(vault),
    )
    append_log_line(
        "ingest `b.md` → 2 nuove, 0 patch, 0 deferred",
        "runidtwo5678",
        vault_path=str(vault),
    )

    lines = (vault / DEFAULT_LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "a.md" in lines[0]
    assert "b.md" in lines[1]


def test_same_run_id_idempotent_no_duplicate(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    first = append_log_line(
        "ingest `a.md` → 1 nuove, 0 patch, 0 deferred",
        "samerunid123",
        vault_path=str(vault),
    )
    second = append_log_line(
        "ingest `a.md` → 1 nuove, 0 patch, 0 deferred",
        "samerunid123",
        vault_path=str(vault),
    )

    assert first is True
    assert second is False
    lines = (vault / DEFAULT_LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_same_run_id_different_dedup_keys_appends_both(tmp_path):
    """Multi-file run: one run_id, one line per file. dedup_key scopes the
    idempotency check to (run_id, key); re-appending the same key is a no-op."""
    vault = tmp_path / "vault"
    vault.mkdir()

    first = append_log_line(
        "ingest `a.md` → 1 nuove, 0 patch, 0 deferred",
        "sharedrunid1",
        vault_path=str(vault),
        dedup_key="`a.md`",
    )
    second = append_log_line(
        "ingest `b.md` → 2 nuove, 0 patch, 0 deferred",
        "sharedrunid1",
        vault_path=str(vault),
        dedup_key="`b.md`",
    )
    resumed = append_log_line(  # resume of file a under the same run
        "ingest `a.md` → 1 nuove, 0 patch, 0 deferred",
        "sharedrunid1",
        vault_path=str(vault),
        dedup_key="`a.md`",
    )

    assert first is True
    assert second is True
    assert resumed is False
    lines = (vault / DEFAULT_LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "a.md" in lines[0]
    assert "b.md" in lines[1]


def test_missing_vault_path_is_noop(monkeypatch):
    import silica.config as config_mod

    monkeypatch.setattr(config_mod.CONFIG, "vault_path", "")
    ok = append_log_line("event", "runid12345678")
    assert ok is False


def test_append_falls_back_to_config_vault_path(tmp_vault):
    from silica.config import CONFIG

    ok = append_log_line("ingest `a.md` → 1 nuove, 0 patch, 0 deferred", "cfgrunid1234")

    assert ok is True
    assert (Path(CONFIG.vault_path) / DEFAULT_LOG_FILENAME).exists()


def test_tail_log_returns_last_n_lines(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    for i in range(7):
        append_log_line(
            f"ingest `f{i}.md` → 1 nuove, 0 patch, 0 deferred",
            f"run{i:05d}abc",
            vault_path=str(vault),
        )

    tail = tail_log(5, vault_path=str(vault))

    assert len(tail) == 5
    assert "f2.md" in tail[0]
    assert "f6.md" in tail[-1]


def test_tail_log_missing_file_returns_empty(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    assert tail_log(5, vault_path=str(vault)) == []
