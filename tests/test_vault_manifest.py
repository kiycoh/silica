"""vault.yaml manifest (ADR-0014): declared capabilities, not vault types."""
import subprocess

import pytest

from silica.config import CONFIG
from silica.kernel.vault_manifest import (
    VaultManifest,
    apply_manifest_to_config,
    default_sources,
    get_active_manifest,
    load_manifest,
    reset_manifest_cache,
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_manifest_cache()
    yield
    reset_manifest_cache()


def test_defaults_prose_only_outside_git(tmp_path):
    m = load_manifest(tmp_path)
    assert m.sources == ("prose",)
    assert m.overlay is None and m.cooccurrence_lang is None


def test_defaults_include_code_inside_git(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    vault = tmp_path / ".silica"
    vault.mkdir()
    assert load_manifest(vault).sources == ("prose", "code")


def test_manifest_file_overrides_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "sources: [prose]\noverlay: codebase\ncooccurrence_lang: italian\n",
        encoding="utf-8",
    )
    m = load_manifest(tmp_path)
    assert m.sources == ("prose",)
    assert m.overlay == "codebase"
    assert m.cooccurrence_lang == "italian"


def test_malformed_manifest_degrades_to_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text("sources: 42\n", encoding="utf-8")
    assert load_manifest(tmp_path).sources == ("prose",)
    (tmp_path / "vault.yaml").write_text(":\n  - not yaml mapping [", encoding="utf-8")
    assert load_manifest(tmp_path).sources == ("prose",)


def test_get_active_manifest_caches_until_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    assert get_active_manifest().sources == ("prose",)
    (tmp_path / "vault.yaml").write_text("sources: [prose, code]\n", encoding="utf-8")
    assert get_active_manifest().sources == ("prose",)  # cached
    reset_manifest_cache()
    assert get_active_manifest().sources == ("prose", "code")


def test_apply_manifest_env_wins(tmp_path, monkeypatch):
    (tmp_path / "vault.yaml").write_text(
        "overlay: codebase\ncooccurrence_lang: italian\n", encoding="utf-8"
    )
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(CONFIG, "domain", None)
    monkeypatch.setattr(CONFIG, "cooccurrence_lang", "english")

    monkeypatch.delenv("SILICA_DOMAIN", raising=False)
    monkeypatch.delenv("SILICA_COOCCURRENCE_LANG", raising=False)
    apply_manifest_to_config()
    assert CONFIG.domain == "codebase"
    assert CONFIG.cooccurrence_lang == "italian"

    reset_manifest_cache()
    monkeypatch.setattr(CONFIG, "domain", "legal")
    monkeypatch.setenv("SILICA_DOMAIN", "legal")
    apply_manifest_to_config()
    assert CONFIG.domain == "legal"  # env precedence


def test_apply_manifest_clears_overlay_on_switch_to_plain_vault(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "vault.yaml").write_text(
        "overlay: legal\ncooccurrence_lang: italian\n", encoding="utf-8"
    )
    monkeypatch.delenv("SILICA_DOMAIN", raising=False)
    monkeypatch.delenv("SILICA_COOCCURRENCE_LANG", raising=False)
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path / "a"))
    monkeypatch.setattr(CONFIG, "domain", None)
    monkeypatch.setattr(CONFIG, "cooccurrence_lang", "english")
    apply_manifest_to_config()
    assert CONFIG.domain == "legal" and CONFIG.cooccurrence_lang == "italian"

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path / "b"))
    reset_manifest_cache()
    apply_manifest_to_config()
    assert CONFIG.domain is None          # not leaked from vault a
    assert CONFIG.cooccurrence_lang == "english"


def test_ingest_gating_sources_prose_only(tmp_path, monkeypatch, capsys):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    vault = tmp_path / ".silica"
    vault.mkdir()
    (vault / "vault.yaml").write_text("sources: [prose]\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    from silica.driver import fs_backend
    import silica.driver as driver_mod
    monkeypatch.setattr(driver_mod, "DRIVER", fs_backend.ObsidianFSBackend(str(vault)))
    reset_manifest_cache()

    from silica.cli import _expand_workflow_shortcut

    msg = _expand_workflow_shortcut("/ingest m.py")
    assert msg == ""  # handled: skipped, nothing for the agent
    assert not (vault / "Inbox" / "m.md").exists()  # code source disabled
    assert "Skipped" in capsys.readouterr().out
