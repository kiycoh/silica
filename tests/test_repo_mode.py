import subprocess
from pathlib import Path

from silica.cli import _activate_repo_mode, default_user_vault, resolve_repo_mode_vault
from silica.config import CONFIG


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _obsidian(path: Path) -> None:
    (path / ".obsidian").mkdir(parents=True, exist_ok=True)


def test_explicit_vault_env_code_repo_adopts_docs_silica(tmp_path):
    # SILICA_VAULT at a code repo root (no .obsidian) → <repo>/docs/silica.
    _init_repo(tmp_path)
    orig = CONFIG.vault_path
    try:
        CONFIG.vault_path = str(tmp_path)
        _activate_repo_mode()
        assert Path(CONFIG.vault_path).resolve() == (tmp_path / "docs" / "silica").resolve()
        assert (tmp_path / "docs" / "silica").is_dir()
    finally:
        CONFIG.vault_path = orig


def test_explicit_vault_env_obsidian_vault_verbatim(tmp_path):
    # An Obsidian vault (even git-tracked) is adopted exactly — no docs/silica.
    _init_repo(tmp_path)
    _obsidian(tmp_path)
    orig = CONFIG.vault_path
    try:
        CONFIG.vault_path = str(tmp_path)
        _activate_repo_mode()
        assert Path(CONFIG.vault_path).resolve() == tmp_path.resolve()
        assert not (tmp_path / "docs" / "silica").exists()
    finally:
        CONFIG.vault_path = orig


def test_repo_mode_picks_docs_silica_when_present(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "docs" / "silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=True)
    assert Path(result).resolve() == (tmp_path / "docs" / "silica").resolve()


def test_repo_mode_obsidian_root_is_verbatim(tmp_path):
    # .obsidian at the repo root → adopt the root itself, even without docs_ok.
    _init_repo(tmp_path)
    _obsidian(tmp_path)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=False)
    assert Path(result).resolve() == tmp_path.resolve()


def test_repo_mode_skipped_when_vault_env_set(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "docs" / "silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="/explicit/vault", docs_exists_ok=True)
    assert result is None


def test_repo_mode_none_outside_repo(tmp_path):
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=True)
    assert result is None


def test_repo_mode_none_when_docs_silica_missing_and_not_okd(tmp_path):
    _init_repo(tmp_path)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=False)
    assert result is None


def test_repo_mode_skipped_for_silica_own_repo(tmp_path):
    # Running inside Silica's *own* source repo is dev mode, not a vault.
    _init_repo(tmp_path)
    (tmp_path / "docs" / "silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(
        cwd=tmp_path, vault_env="", docs_exists_ok=True, self_repo=tmp_path.resolve()
    )
    assert result is None


def test_repo_mode_unaffected_when_self_repo_differs(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "docs" / "silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(
        cwd=tmp_path, vault_env="", docs_exists_ok=True, self_repo=tmp_path / "elsewhere"
    )
    assert Path(result).resolve() == (tmp_path / "docs" / "silica").resolve()


def test_default_user_vault_under_home(tmp_path):
    assert default_user_vault(home=tmp_path) == tmp_path / ".silica" / "vault"
