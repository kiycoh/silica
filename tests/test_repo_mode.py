import subprocess
from pathlib import Path

from silica.cli import resolve_repo_mode_vault


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def test_repo_mode_picks_docs_silica_when_present(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "docs" / "silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=True)
    assert Path(result).resolve() == (tmp_path / "docs" / "silica").resolve()


def test_repo_mode_skipped_when_vault_env_set(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "docs" / "silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="/explicit/vault", docs_exists_ok=True)
    assert result is None


def test_repo_mode_none_outside_repo(tmp_path):
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=True)
    assert result is None


def test_repo_mode_none_when_silica_missing_and_not_okd(tmp_path):
    _init_repo(tmp_path)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=False)
    assert result is None
