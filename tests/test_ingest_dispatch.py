"""/ingest — one verb, extension dispatch (spec D2).

md/.txt → Injector FSM message (agent loop); code → skeleton stub staged
inline, returns "" sentinel (fully handled, nothing for the agent)."""
import json
import subprocess
from pathlib import Path

import pytest

from silica.cli import _expand_workflow_shortcut
from silica.config import CONFIG


@pytest.fixture(autouse=True)
def _reset_manifest_cache():
    from silica.kernel.vault_manifest import reset_manifest_cache
    reset_manifest_cache()
    yield
    reset_manifest_cache()


def test_ingest_md_expands_to_injector_message():
    msg = _expand_workflow_shortcut("/ingest Inbox/a.md --target=Concepts/AI")
    assert msg is not None and "silica_run_injector" in msg
    assert "Inbox/a.md" in msg and "Concepts/AI" in msg


def test_ingest_md_missing_target_returns_error():
    msg = _expand_workflow_shortcut("/ingest Inbox/a.md")
    assert msg is not None and msg.startswith("Error:")
    assert "--target" in msg


def test_ingest_no_files_returns_error():
    msg = _expand_workflow_shortcut("/ingest --target=Concepts")
    assert msg is not None and msg.startswith("Error:")


def test_inject_shortcut_is_retired():
    assert _expand_workflow_shortcut("/inject Inbox/a.md --target=C") is None


@pytest.fixture
def repo_vault(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "m.py").write_text("def hi():\n    return 1\n", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    vault = tmp_path / ".silica"
    vault.mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(CONFIG, "inbox_dir", "Inbox")
    from silica.driver import fs_backend
    import silica.driver as driver_mod
    monkeypatch.setattr(driver_mod, "DRIVER", fs_backend.ObsidianFSBackend(str(vault)))
    return tmp_path, vault


def test_ingest_code_stages_stub_and_returns_sentinel(repo_vault):
    root, vault = repo_vault
    msg = _expand_workflow_shortcut("/ingest m.py")
    assert msg == ""  # fully handled inline, nothing for the agent
    stub = vault / "Inbox" / "m.md"
    assert stub.is_file()
    text = stub.read_text(encoding="utf-8")
    assert "def hi()" in text and "return 1" not in text


def test_ingest_mixed_batch_stages_code_and_expands_md(repo_vault):
    root, vault = repo_vault
    msg = _expand_workflow_shortcut("/ingest m.py Inbox/note.md --target=Concepts")
    assert msg is not None and "silica_run_injector" in msg
    assert '"Inbox/note.md"' in msg  # md file forwarded to the agent
    assert '"m.py"' not in msg       # code file NOT forwarded (staged inline)
    assert (vault / "Inbox" / "m.md").is_file()


def test_ingest_unsupported_extension_is_skipped(repo_vault, capsys):
    root, vault = repo_vault
    msg = _expand_workflow_shortcut("/ingest data.csv")
    assert msg == ""  # handled, nothing for the agent
    assert not (vault / "Inbox" / "data.md").exists()
    out = capsys.readouterr().out
    assert "data.csv" in out and "Skipped" in out  # warning is part of the contract


def test_ingest_pdf_converts_and_forwards_converted_md(repo_vault, monkeypatch):
    """No adapter claims .pdf → convert() runs and the CONVERTED .md is what
    the FSM is told to re-read (not the .pdf)."""
    import silica.sources.convert as conv_mod

    monkeypatch.setattr(conv_mod, "convert", lambda f, dest_dir="": "Inbox/paper.md")
    msg = _expand_workflow_shortcut("/ingest paper.pdf --target=Concepts/AI")
    assert msg is not None and "silica_run_injector" in msg
    assert '"Inbox/paper.md"' in msg   # converted .md forwarded
    assert "paper.pdf" not in msg      # original .pdf is NOT re-read


def test_ingest_pdf_converter_error_is_caught(repo_vault, monkeypatch, capsys):
    import silica.sources.convert as conv_mod

    def boom(f, dest_dir=""):
        raise ValueError("mineru not installed")

    monkeypatch.setattr(conv_mod, "convert", boom)
    msg = _expand_workflow_shortcut("/ingest paper.pdf --target=Concepts/AI")
    assert msg == ""  # nothing to run; batch did not crash
    assert "mineru not installed" in capsys.readouterr().out


def test_convert_command_returns_sentinel_and_reports(repo_vault, monkeypatch, capsys):
    import silica.sources.convert as conv_mod

    monkeypatch.setattr(conv_mod, "convert", lambda f, dest_dir="": "Inbox/paper.md")
    msg = _expand_workflow_shortcut("/convert paper.pdf")
    assert msg == ""  # fully handled inline
    assert "Converted" in capsys.readouterr().out


def test_convert_command_no_files_errors():
    msg = _expand_workflow_shortcut("/convert --target=X")
    assert msg is not None and msg.startswith("Error:")


# ---------------------------------------------------------------------------
# Re-ingest-of-modified-source warning (spec-hermes-coherence §3): a file
# about to be staged whose basename is already registered in
# .silica/provenance.json under a DIFFERENT sha256 means notes derived from
# it may now be stale.
# ---------------------------------------------------------------------------

def test_ingest_reingest_of_modified_source_warns(repo_vault, capsys):
    root, vault = repo_vault
    inbox = vault / "Inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "lezione.md").write_text("v2 content", encoding="utf-8")

    from silica.kernel.provenance import append_record
    append_record("lezione.md", "old-sha-not-matching", "run1", ["Concepts/A", "Concepts/B"])

    msg = _expand_workflow_shortcut("/ingest Inbox/lezione.md --target=Concepts/AI")
    assert msg is not None and "silica_run_injector" in msg

    out = capsys.readouterr().out
    assert "re-ingest di fonte modificata" in out
    assert "2 note" in out


def test_ingest_same_sha_no_warning(repo_vault, capsys):
    root, vault = repo_vault
    inbox = vault / "Inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "lezione.md").write_text("same content", encoding="utf-8")

    from silica.kernel.provenance import append_record, content_sha256
    sha = content_sha256("Inbox/lezione.md")
    append_record("lezione.md", sha, "run1", ["Concepts/A"])

    msg = _expand_workflow_shortcut("/ingest Inbox/lezione.md --target=Concepts/AI")
    assert msg is not None

    out = capsys.readouterr().out
    assert "re-ingest di fonte modificata" not in out


def test_ingest_no_prior_provenance_no_warning(repo_vault, capsys):
    root, vault = repo_vault
    inbox = vault / "Inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "fresh.md").write_text("brand new", encoding="utf-8")

    msg = _expand_workflow_shortcut("/ingest Inbox/fresh.md --target=Concepts/AI")
    assert msg is not None

    out = capsys.readouterr().out
    assert "re-ingest di fonte modificata" not in out


def test_ingest_missing_target_errors_before_reingest_warning(repo_vault, capsys):
    """An invocation missing --target is invalid regardless of provenance —
    the drift warning must not print before the guard rejects the call."""
    root, vault = repo_vault
    inbox = vault / "Inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "lezione.md").write_text("v2 content", encoding="utf-8")

    from silica.kernel.provenance import append_record
    append_record("lezione.md", "old-sha-not-matching", "run1", ["Concepts/A", "Concepts/B"])

    msg = _expand_workflow_shortcut("/ingest Inbox/lezione.md")

    assert msg is not None and msg.startswith("Error:") and "--target" in msg
    out = capsys.readouterr().out
    assert "re-ingest di fonte modificata" not in out
