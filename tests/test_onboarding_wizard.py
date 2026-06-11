"""Tests for silica.onboarding.wizard — .env merge and interactive flow."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestMergeEnv:
    def test_updates_managed_key_in_place(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("SILICA_MODEL=old\nFOO=bar\n", {"SILICA_MODEL": "new"})
        assert "SILICA_MODEL=new" in out
        assert "SILICA_MODEL=old" not in out

    def test_preserves_unknown_lines_and_comments(self):
        from silica.onboarding.wizard import merge_env
        existing = "# my comment\nFOO=bar\n\nexport OTHER=keep\n"
        out = merge_env(existing, {"SILICA_MODEL": "m"})
        assert "# my comment" in out
        assert "FOO=bar" in out
        assert "export OTHER=keep" in out

    def test_appends_new_keys(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("FOO=bar\n", {"SILICA_BACKEND": "fs"})
        assert out.splitlines()[-1] == "SILICA_BACKEND=fs"

    def test_empty_existing(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("", {"SILICA_MODEL": "m"})
        assert out == "SILICA_MODEL=m\n"

    def test_matches_export_prefix(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("export SILICA_MODEL=old\n", {"SILICA_MODEL": "new"})
        assert "SILICA_MODEL=new" in out
        assert "old" not in out


class TestRunWizard:
    def _scripted(self, answers):
        it = iter(answers)
        return lambda prompt: next(it)

    def test_full_lmstudio_flow_writes_env(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"
        env_path.write_text("# keep me\nFOO=bar\n")

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard.shutil, "which", lambda name: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        # os.environ.update inside the wizard must not leak into the test env
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            str(vault),    # vault path (no repo detected)
            "",            # backend → default fs (obsidian not on PATH)
            "",            # provider → default lmstudio
            "test-model",  # model id
            "skip",        # embeddings → skip
            "",            # write .env → default y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "# keep me" in content
        assert "FOO=bar" in content
        assert f"SILICA_VAULT={vault}" in content
        assert "SILICA_BACKEND=fs" in content
        assert "SILICA_PROVIDER=lmstudio" in content
        assert "SILICA_MODEL=test-model" in content
        assert "OPENROUTER_API_KEY" not in content

    def test_declining_write_aborts(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard.shutil, "which", lambda name: None)

        answers = [str(vault), "", "", "test-model", "skip", "n"]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 1
        assert not env_path.exists()

    def test_repo_mode_skips_vault_key(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        (tmp_path / "docs" / "silica").mkdir(parents=True)
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard.shutil, "which", lambda name: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # repo mode? → default y
            "",            # backend → fs
            "",            # provider → lmstudio
            "test-model",  # model
            "skip",        # embeddings
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert "SILICA_VAULT" not in env_path.read_text()

    def test_eof_mid_wizard_aborts_cleanly(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        env_path = tmp_path / ".env"
        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard.shutil, "which", lambda name: None)

        vault = tmp_path / "vault"
        vault.mkdir()
        # Input exhausts after the vault answer → simulates Ctrl+D mid-wizard
        rc = wizard.run_wizard(input_fn=self._scripted([str(vault)]), env_path=env_path)

        assert rc == 1
        assert not env_path.exists()


class TestAskSecret:
    def test_secret_default_is_masked_in_prompt(self):
        from silica.onboarding.wizard import _ask

        prompts: list[str] = []

        def fake_input(p: str) -> str:
            prompts.append(p)
            return ""

        result = _ask(fake_input, "OpenRouter API key", "sk-or-verysecret-a1b2", secret=True)
        assert result == "sk-or-verysecret-a1b2"   # empty input keeps the default
        assert "verysecret" not in prompts[0]       # secret never echoed
        assert "a1b2" in prompts[0]                 # last-4 hint shown
