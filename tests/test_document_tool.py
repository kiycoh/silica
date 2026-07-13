import json
import subprocess
from pathlib import Path

import pytest

from silica.config import CONFIG

PY_SRC = '''\
import os
from silica.kernel import gitstate


def hi(name: str) -> str:
    """Say hi."""
    return f"hi {name}"
'''


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    f = path / "src" / "m.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(PY_SRC, encoding="utf-8")
    # real first-party file so `silica.kernel.gitstate` resolves to a wikilink
    (path / "silica" / "kernel").mkdir(parents=True, exist_ok=True)
    (path / "silica" / "kernel" / "gitstate.py").write_text(
        "def head_ref():\n    return None\n", encoding="utf-8"
    )
    (path / "data.csv").write_text("a,b\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


@pytest.fixture
def repo_vault(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    vault = tmp_path / ".silica"
    vault.mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(CONFIG, "inbox_dir", "Inbox")
    from silica.driver import fs_backend
    import silica.driver as driver_mod
    monkeypatch.setattr(driver_mod, "DRIVER", fs_backend.ObsidianFSBackend(str(vault)))
    return tmp_path, vault


def _run(path: str):
    from silica.tools.codedocs_tool import silica_document
    result = silica_document(path=path)
    return json.loads(result) if isinstance(result, str) else result


def test_document_writes_skeleton_not_source(repo_vault):
    root, vault = repo_vault
    data = _run("src/m.py")
    assert data["status"] == "ok"
    assert "Inbox/" in data["note_path"]
    written = (vault / data["note_path"]).read_text(encoding="utf-8")
    assert "documents:" in written
    assert "src/m.py" in written
    # skeleton present
    assert "def hi(name: str) -> str" in written
    assert "Say hi." in written
    # full source body is GONE
    assert 'return f"hi {name}"' not in written


def test_document_splits_first_party_and_external_imports(repo_vault):
    root, vault = repo_vault
    data = _run("src/m.py")
    written = (vault / data["note_path"]).read_text(encoding="utf-8")
    assert "First-party" in written
    assert "[[silica.kernel.gitstate]]" in written  # path-qualified wikilink
    assert "External" in written
    assert "`os`" in written


def test_document_unsupported_language_stub_without_dump(repo_vault):
    root, vault = repo_vault
    data = _run("data.csv")
    assert data["status"] == "ok"
    written = (vault / data["note_path"]).read_text(encoding="utf-8")
    assert "documents:" in written
    assert "skeleton unavailable" in written.lower()
    assert "a,b" not in written  # never the raw content


def test_document_rejects_path_outside_repo(repo_vault):
    data = _run("../../etc/passwd")
    assert data["status"] == "error"


def test_document_fence_integrity_with_backticks(repo_vault):
    root, vault = repo_vault
    evil = root / "src" / "evil.py"
    evil.write_text(
        'def attack():\n    """contains ``` fence and `ticks` inside."""\n    return 0\n',
        encoding="utf-8",
    )
    data = _run("src/evil.py")
    written = (vault / data["note_path"]).read_text(encoding="utf-8")
    assert "`" not in written.split("```text", 1)[1].split("```", 1)[0]
    # fence opens and closes exactly once each
    assert written.count("```text") == 1
    assert written.count("```") == 2
