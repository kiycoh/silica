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
