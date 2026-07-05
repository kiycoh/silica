import subprocess
from pathlib import Path

from silica.cli import _activate_repo_mode, default_user_vault, resolve_repo_mode_vault
from silica.config import CONFIG


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def test_explicit_vault_env_at_repo_root_adopts_dot_silica(tmp_path):
    # SILICA_VAULT pointing at a git repo root → <repo>/.silica, created on demand.
    _init_repo(tmp_path)
    orig = CONFIG.vault_path
    try:
        CONFIG.vault_path = str(tmp_path)
        _activate_repo_mode()
        assert Path(CONFIG.vault_path).resolve() == (tmp_path / ".silica").resolve()
        assert (tmp_path / ".silica").is_dir()
    finally:
        CONFIG.vault_path = orig


def test_explicit_vault_env_plain_dir_verbatim(tmp_path):
    # A plain (non-repo) directory is adopted exactly as given — no .silica.
    orig = CONFIG.vault_path
    try:
        CONFIG.vault_path = str(tmp_path)
        _activate_repo_mode()
        assert Path(CONFIG.vault_path).resolve() == tmp_path.resolve()
        assert not (tmp_path / ".silica").exists()
    finally:
        CONFIG.vault_path = orig


def test_repo_mode_picks_dot_silica_when_present(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=True)
    assert Path(result).resolve() == (tmp_path / ".silica").resolve()


def test_repo_mode_skipped_when_vault_env_set(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="/explicit/vault", docs_exists_ok=True)
    assert result is None


def test_repo_mode_none_outside_repo(tmp_path):
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=True)
    assert result is None


def test_repo_mode_none_when_silica_missing_and_not_okd(tmp_path):
    _init_repo(tmp_path)
    result = resolve_repo_mode_vault(cwd=tmp_path, vault_env="", docs_exists_ok=False)
    assert result is None


def test_repo_mode_skipped_for_silica_own_repo(tmp_path):
    # Running inside Silica's *own* source repo is dev mode, not a vault.
    _init_repo(tmp_path)
    (tmp_path / ".silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(
        cwd=tmp_path, vault_env="", docs_exists_ok=True, self_repo=tmp_path.resolve()
    )
    assert result is None


def test_repo_mode_unaffected_when_self_repo_differs(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".silica").mkdir(parents=True)
    result = resolve_repo_mode_vault(
        cwd=tmp_path, vault_env="", docs_exists_ok=True, self_repo=tmp_path / "elsewhere"
    )
    assert Path(result).resolve() == (tmp_path / ".silica").resolve()


def test_default_user_vault_under_home(tmp_path):
    assert default_user_vault(home=tmp_path) == tmp_path / ".silica" / "vault"
