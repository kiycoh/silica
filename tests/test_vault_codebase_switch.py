# tests/test_vault_codebase_switch.py
"""Vault resolution is driven by `.obsidian/`, not git: an Obsidian vault is
adopted verbatim; anything else (code repo, plain or missing dir) is Silica
repo mode → <target>/docs/silica, created on demand."""
import subprocess

import silica.driver as driver_pkg
from silica.cli import resolve_vault_switch, _handle_direct_shortcut
from silica.config import CONFIG


def _git_init(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _obsidian(path):
    (path / ".obsidian").mkdir(parents=True, exist_ok=True)


def test_obsidian_vault_is_verbatim(tmp_path):
    _obsidian(tmp_path)

    target = resolve_vault_switch(str(tmp_path))

    assert target.vault == str(tmp_path.resolve())  # notes in the root
    assert target.created is False


def test_non_obsidian_dir_is_repo_mode(tmp_path):
    target = resolve_vault_switch(str(tmp_path))

    assert target.vault == str((tmp_path / "docs" / "silica").resolve())
    assert target.created is True  # docs/silica not there yet


def test_git_is_not_the_proxy(tmp_path):
    # A git repo with no .obsidian is repo mode, NOT verbatim/.silica.
    _git_init(tmp_path)

    target = resolve_vault_switch(str(tmp_path))

    assert target.vault == str((tmp_path / "docs" / "silica").resolve())


def test_git_tracked_obsidian_vault_stays_verbatim(tmp_path):
    # The original bug: a git-tracked Obsidian vault must not nest.
    _git_init(tmp_path)
    _obsidian(tmp_path)

    target = resolve_vault_switch(str(tmp_path))

    assert target.vault == str(tmp_path.resolve())  # not <vault>/docs/silica
    assert target.created is False


def test_existing_docs_silica_is_adopted(tmp_path):
    (tmp_path / "docs" / "silica").mkdir(parents=True)

    target = resolve_vault_switch(str(tmp_path))

    assert target.vault == str((tmp_path / "docs" / "silica").resolve())
    assert target.created is False


def test_nonexistent_path_is_repo_mode(tmp_path):
    # No .obsidian to check → repo mode, backend creates docs/silica.
    missing = tmp_path / "missing"

    target = resolve_vault_switch(str(missing))

    assert target.vault == str((missing / "docs" / "silica").resolve())
    assert target.created is True


def test_handler_on_code_repo_switches_to_docs_silica(tmp_path, monkeypatch):
    _git_init(tmp_path)
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path / "old"))
    monkeypatch.setattr(driver_pkg, "_driver", object())  # sentinel to observe reset

    handled = _handle_direct_shortcut(f"/vault {tmp_path}", [])

    assert handled is True
    docs_silica = tmp_path / "docs" / "silica"
    assert docs_silica.is_dir()  # created on demand
    assert CONFIG.vault_path == str(docs_silica.resolve())  # not the repo root
    assert driver_pkg._driver is None  # reset so next read uses docs/silica


def test_handler_on_obsidian_vault_is_verbatim(tmp_path, monkeypatch):
    _obsidian(tmp_path)
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path / "old"))
    monkeypatch.setattr(driver_pkg, "_driver", object())

    handled = _handle_direct_shortcut(f"/vault {tmp_path}", [])

    assert handled is True
    assert CONFIG.vault_path == str(tmp_path.resolve())  # root, no docs/silica
    assert not (tmp_path / "docs" / "silica").exists()
