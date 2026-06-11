"""Tests for silica.onboarding.checks — pure doctor diagnostics."""
from __future__ import annotations

from pathlib import Path

import pytest

from silica.config import SilicaConfig


def _cfg(**overrides) -> SilicaConfig:
    """Fresh config with explicit fields — never depends on the dev's .env."""
    cfg = SilicaConfig()
    cfg.model = ""
    cfg._provider = None
    cfg.vault_path = ""
    cfg.backend = "fs"
    cfg.embedding_model = "test-embed"
    cfg.embedding_base_url = "http://localhost:9999/v1"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestCheckChatModel:
    def test_empty_model_fails_with_init_hint(self):
        from silica.onboarding.checks import check_chat_model
        r = check_chat_model(_cfg(model=""))
        assert r.status == "fail"
        assert "silica init" in r.hint

    def test_openrouter_without_key_fails(self, monkeypatch):
        from silica.onboarding.checks import check_chat_model
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        r = check_chat_model(_cfg(model="openrouter/openai/gpt-4o-mini"))
        assert r.status == "fail"
        assert "OPENROUTER_API_KEY" in r.detail

    def test_openrouter_with_key_ok(self, monkeypatch):
        from silica.onboarding.checks import check_chat_model
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        r = check_chat_model(_cfg(model="openrouter/openai/gpt-4o-mini"))
        assert r.status == "ok"

    def test_lmstudio_model_ok_without_key(self, monkeypatch):
        from silica.onboarding.checks import check_chat_model
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        r = check_chat_model(_cfg(model="qwen3-30b"))
        assert r.status == "ok"


class TestCheckChatEndpoint:
    def test_no_model_skips_with_warn(self):
        from silica.onboarding.checks import check_chat_endpoint
        r = check_chat_endpoint(_cfg(model=""))
        assert r.status == "warn"
        assert "skipped" in r.detail

    def test_openrouter_not_probed(self):
        from silica.onboarding.checks import check_chat_endpoint
        r = check_chat_endpoint(_cfg(model="openrouter/openai/gpt-4o-mini"))
        assert r.status == "ok"
        assert "not probed" in r.detail

    def test_lmstudio_unreachable_fails(self, monkeypatch):
        import silica.onboarding.checks as checks

        def boom(url, timeout):
            raise checks.httpx.ConnectError("refused")

        monkeypatch.setattr(checks.httpx, "get", boom)
        r = checks.check_chat_endpoint(_cfg(model="qwen3-30b"))
        assert r.status == "fail"
        assert "LM Studio" in r.hint

    def test_lmstudio_reachable_ok(self, monkeypatch):
        import silica.onboarding.checks as checks

        class FakeResp:
            pass

        monkeypatch.setattr(checks.httpx, "get", lambda url, timeout: FakeResp())
        r = checks.check_chat_endpoint(_cfg(model="qwen3-30b"))
        assert r.status == "ok"


class TestCheckVault:
    def test_explicit_path_missing_fails(self):
        from silica.onboarding.checks import check_vault
        r = check_vault(_cfg(vault_path="/nonexistent/vault"))
        assert r.status == "fail"

    def test_explicit_path_ok_with_inbox(self, tmp_path):
        from silica.onboarding.checks import check_vault
        (tmp_path / "Inbox").mkdir()
        r = check_vault(_cfg(vault_path=str(tmp_path), inbox_dir="Inbox"))
        assert r.status == "ok"

    def test_missing_inbox_warns(self, tmp_path):
        from silica.onboarding.checks import check_vault
        r = check_vault(_cfg(vault_path=str(tmp_path), inbox_dir="Inbox"))
        assert r.status == "warn"
        assert "Inbox" in r.detail

    def test_unset_no_repo_warns_with_init_hint(self, monkeypatch):
        import silica.onboarding.checks as checks
        monkeypatch.setattr(checks.gitstate, "find_repo_root", lambda p: None)
        r = checks.check_vault(_cfg(vault_path=""))
        assert r.status == "warn"
        assert "silica init" in r.hint

    def test_unset_with_repo_docs_silica_ok(self, monkeypatch, tmp_path):
        import silica.onboarding.checks as checks
        (tmp_path / "docs" / "silica").mkdir(parents=True)
        monkeypatch.setattr(checks.gitstate, "find_repo_root", lambda p: tmp_path)
        r = checks.check_vault(_cfg(vault_path=""))
        assert r.status == "ok"
        assert "repo mode" in r.detail

    def test_explicit_path_not_writable_fails(self, tmp_path):
        import os as os_mod

        from silica.onboarding.checks import check_vault

        vault = tmp_path / "ro_vault"
        vault.mkdir()
        vault.chmod(0o500)
        try:
            if os_mod.access(vault, os_mod.W_OK):
                pytest.skip("running with permissions that ignore chmod (e.g. root)")
            r = check_vault(_cfg(vault_path=str(vault)))
            assert r.status == "fail"
            assert "writable" in r.detail
        finally:
            vault.chmod(0o700)


class TestCheckObsidianBackend:
    def test_fs_backend_ok_headless(self):
        from silica.onboarding.checks import check_obsidian_backend
        r = check_obsidian_backend(_cfg(backend="fs"))
        assert r.status == "ok"
        assert "headless" in r.detail

    def test_cli_backend_binary_missing_fails(self, monkeypatch):
        import silica.onboarding.checks as checks
        monkeypatch.setattr(checks.shutil, "which", lambda name: None)
        r = checks.check_obsidian_backend(_cfg(backend="cli"))
        assert r.status == "fail"
        assert "SILICA_BACKEND=fs" in r.hint

    def test_cli_backend_timeout_fails(self, monkeypatch):
        import silica.onboarding.checks as checks
        monkeypatch.setattr(checks.shutil, "which", lambda name: "/usr/bin/obsidian")

        def fake_run(*args, **kwargs):
            raise checks.subprocess.TimeoutExpired(cmd="obsidian", timeout=8)

        monkeypatch.setattr(checks.subprocess, "run", fake_run)
        r = checks.check_obsidian_backend(_cfg(backend="cli"))
        assert r.status == "fail"
        assert "desktop" in r.hint

    def test_cli_backend_responds_ok(self, monkeypatch):
        import silica.onboarding.checks as checks
        monkeypatch.setattr(checks.shutil, "which", lambda name: "/usr/bin/obsidian")
        monkeypatch.setattr(checks.subprocess, "run", lambda *a, **k: None)
        r = checks.check_obsidian_backend(_cfg(backend="cli"))
        assert r.status == "ok"


class TestCheckEmbeddings:
    def test_unreachable_warns_never_fails(self, monkeypatch):
        import silica.onboarding.checks as checks

        def boom(url, timeout):
            raise checks.httpx.ConnectError("refused")

        monkeypatch.setattr(checks.httpx, "get", boom)
        r = checks.check_embeddings(_cfg())
        assert r.status == "warn"
        assert "co-occurrence" in r.hint

    def test_model_not_listed_warns(self, monkeypatch):
        import silica.onboarding.checks as checks

        class FakeResp:
            def json(self):
                return {"data": [{"id": "other-model"}]}

        monkeypatch.setattr(checks.httpx, "get", lambda url, timeout: FakeResp())
        r = checks.check_embeddings(_cfg(embedding_model="test-embed"))
        assert r.status == "warn"
        assert "test-embed" in r.detail

    def test_model_listed_ok(self, monkeypatch):
        import silica.onboarding.checks as checks

        class FakeResp:
            def json(self):
                return {"data": [{"id": "test-embed"}]}

        monkeypatch.setattr(checks.httpx, "get", lambda url, timeout: FakeResp())
        r = checks.check_embeddings(_cfg(embedding_model="test-embed"))
        assert r.status == "ok"


class TestAggregation:
    def test_run_checks_returns_all_five(self, monkeypatch, tmp_path):
        import silica.onboarding.checks as checks

        def boom(url, timeout):
            raise checks.httpx.ConnectError("refused")

        monkeypatch.setattr(checks.httpx, "get", boom)
        monkeypatch.setattr(checks.gitstate, "find_repo_root", lambda p: None)
        results = checks.run_checks(_cfg(vault_path=str(tmp_path)))
        assert [r.name for r in results] == [
            "chat model", "chat endpoint", "vault", "obsidian backend", "embeddings",
        ]

    def test_has_failures(self):
        from silica.onboarding.checks import CheckResult, has_failures
        ok = CheckResult("a", "ok", "")
        warn = CheckResult("b", "warn", "")
        fail = CheckResult("c", "fail", "")
        assert not has_failures([ok, warn])
        assert has_failures([ok, fail])

    def test_render_report_smoke(self, monkeypatch):
        import io

        from rich.console import Console

        import silica.onboarding.checks as checks
        from silica.ui import console as console_mod

        buf = io.StringIO()
        monkeypatch.setattr(console_mod, "CONSOLE", Console(file=buf, highlight=False, width=120))
        checks.render_report([
            checks.CheckResult("chat model", "ok", "qwen3-30b via lmstudio"),
            checks.CheckResult("vault", "fail", "missing", "run `silica init`"),
        ])
        out = buf.getvalue()
        assert "chat model" in out
        assert "silica init" in out
