from pathlib import Path

from silica.cli import resolve_repo_mode_vault


def test_inbox_resolves_under_silica(tmp_path, monkeypatch):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "docs" / "silica").mkdir(parents=True)
    vault = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=True)
    # Inbox is vault-relative, so it must land inside docs/silica, not docs/.
    from silica.config import CONFIG
    monkeypatch.setattr(CONFIG, "vault_path", vault)
    inbox = Path(CONFIG.vault_path) / (CONFIG.inbox_dir or "Inbox").strip("/")
    assert inbox.resolve() == (tmp_path / "docs" / "silica" / "Inbox").resolve()
