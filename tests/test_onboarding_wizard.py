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
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        # os.environ.update inside the wizard must not leak into the test env
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            str(vault),    # vault path (no repo detected)
            "",            # force language? → Enter, follow source
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
        assert "SILICA_PROVIDER=lmstudio" in content
        assert "SILICA_MODEL=test-model" in content
        assert "OPENROUTER_API_KEY" not in content

    def test_declining_write_aborts(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)

        answers = [str(vault), "", "", "test-model", "skip", "n"]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 1
        assert not env_path.exists()

    def test_repo_mode_skips_vault_key(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        (tmp_path / "docs" / "silica").mkdir(parents=True)
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # repo mode? → default y
            "",            # force language? → Enter, follow source
            "",            # provider → lmstudio
            "test-model",  # model
            "skip",        # embeddings
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert "SILICA_VAULT" not in env_path.read_text()

    def test_repo_mode_writes_vault_manifest(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # repo mode? → default y
            "",            # force language? → Enter, follow source
            "",            # provider → lmstudio
            "test-model",  # model
            "skip",        # embeddings
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = tmp_path / "docs" / "silica" / "vault.yaml"
        assert manifest.is_file()
        text = manifest.read_text(encoding="utf-8")
        assert "sources:" in text and "code" in text and "overlay: codebase" in text
        assert "conventions" not in text and "language" not in text

    def test_repo_mode_writes_forced_language_into_manifest(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # repo mode? → default y
            "Italian",     # force language? → explicit answer
            "",            # provider → lmstudio
            "test-model",  # model
            "skip",        # embeddings
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = tmp_path / "docs" / "silica" / "vault.yaml"
        text = manifest.read_text(encoding="utf-8")
        assert "conventions:\n  language: Italian" in text

    def test_forced_language_roundtrips_through_load_manifest(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard
        from silica.kernel.vault_manifest import load_manifest

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = ["", "Italian", "", "test-model", "skip", ""]
        wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        manifest = load_manifest(str(tmp_path / "docs" / "silica"))
        assert manifest.conventions.language == "Italian"

    def test_enter_leaves_language_none_after_load_manifest(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard
        from silica.kernel.vault_manifest import load_manifest

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = ["", "", "", "test-model", "skip", ""]
        wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        manifest = load_manifest(str(tmp_path / "docs" / "silica"))
        assert manifest.conventions.language is None

    def test_repo_mode_preserves_existing_manifest(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        (tmp_path / "docs" / "silica").mkdir(parents=True)
        (tmp_path / "docs" / "silica" / "vault.yaml").write_text("sources: [prose]\n", encoding="utf-8")
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = ["", "", "test-model", "skip", ""]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert (tmp_path / "docs" / "silica" / "vault.yaml").read_text(encoding="utf-8") == "sources: [prose]\n"

    def test_eof_mid_wizard_aborts_cleanly(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        env_path = tmp_path / ".env"
        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)

        vault = tmp_path / "vault"
        vault.mkdir()
        # Input exhausts after the vault answer → simulates Ctrl+D mid-wizard
        rc = wizard.run_wizard(input_fn=self._scripted([str(vault)]), env_path=env_path)

        assert rc == 1
        assert not env_path.exists()

    def test_non_repo_mode_writes_forced_language_into_manifest(self, monkeypatch, tmp_path):
        """The design's language question is unscoped to repo mode: an explicit-path
        vault with no vault.yaml yet must also be asked, and an explicit answer writes
        a minimal manifest pinning both cooccurrence_lang (stemmer) and the conventions
        language (distiller) — no sources/overlay, unlike repo mode."""
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            str(vault),    # vault path (no repo detected)
            "Italian",     # force language? → explicit answer
            "",            # provider → lmstudio
            "test-model",  # model
            "skip",        # embeddings
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = vault / "vault.yaml"
        assert manifest.is_file()
        assert manifest.read_text(encoding="utf-8") == (
            "cooccurrence_lang: italian\nconventions:\n  language: Italian\n"
        )
        # both axes resolve through load_manifest
        from silica.kernel.vault_manifest import load_manifest

        m = load_manifest(str(vault))
        assert m.cooccurrence_lang == "italian"
        assert m.conventions.language == "Italian"

    def test_non_repo_mode_enter_writes_no_manifest(self, monkeypatch, tmp_path):
        """Enter on the language question must write nothing at all — no
        vault.yaml — mirroring repo mode's 'no conventions block' behavior for
        the case where nothing else was due to be written for this vault."""
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            str(vault),    # vault path (no repo detected)
            "",            # force language? → Enter, follow source
            "",            # provider → lmstudio
            "test-model",  # model
            "skip",        # embeddings
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert not (vault / "vault.yaml").exists()

    def test_repo_mode_yes_like_language_answer_not_written_as_bool(self, monkeypatch, tmp_path):
        """Finding 5 (final multilingua review): an unvalidated "yes" answer to the
        language question parses as a YAML boolean, which `_parse_conventions`
        folds to None — the user believes they forced a language but silently
        didn't. "yes" must be rejected and treated like Enter (no language forced)."""
        import silica.onboarding.wizard as wizard

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # repo mode? → default y
            "yes",         # force language? → looks like consent, not a language
            "",            # provider → lmstudio
            "test-model",  # model
            "skip",        # embeddings
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = tmp_path / "docs" / "silica" / "vault.yaml"
        text = manifest.read_text(encoding="utf-8")
        assert "language" not in text

    def test_repo_mode_colon_language_answer_does_not_corrupt_manifest(self, monkeypatch, tmp_path):
        """Finding 5: a colon-containing free-text answer, embedded raw into YAML,
        breaks the whole manifest (repo mode's sources/overlay would silently
        degrade to defaults on next load). The answer must be validated before
        it ever reaches the file."""
        import silica.onboarding.wizard as wizard
        from silica.kernel.vault_manifest import load_manifest

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",                   # repo mode? → default y
            "English: British",  # force language? → colon would corrupt raw YAML
            "",                   # provider → lmstudio
            "test-model",         # model
            "skip",               # embeddings
            "",                   # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = load_manifest(str(tmp_path / "docs" / "silica"))
        # sources/overlay are written unconditionally in repo mode — an invalid
        # language answer must never degrade the whole manifest to defaults.
        assert "code" in manifest.sources
        assert manifest.overlay == "codebase"

    def test_non_repo_mode_preserves_existing_manifest_no_question_asked(self, monkeypatch, tmp_path):
        """An existing vault.yaml must never be touched — and the question must not
        even be asked (proven by NOT including an extra scripted answer for it; the
        wizard would abort on EOF if it asked and this test would fail rc != 0)."""
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "vault.yaml").write_text("sources: [prose]\n", encoding="utf-8")
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            str(vault),    # vault path (no repo detected)
            # NO language answer — vault.yaml already exists, must not be asked
            "",            # provider → lmstudio
            "test-model",  # model
            "skip",        # embeddings
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert (vault / "vault.yaml").read_text(encoding="utf-8") == "sources: [prose]\n"


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
