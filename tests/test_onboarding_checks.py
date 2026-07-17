"""Tests for silica.onboarding.checks — pure doctor diagnostics."""
from __future__ import annotations


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

    def test_unset_no_repo_fs_backend_fails(self, monkeypatch):
        """fs + no vault_path and no repo → fail with actionable hint."""
        import silica.onboarding.checks as checks
        monkeypatch.setattr(checks.gitstate, "find_repo_root", lambda p: None)
        r = checks.check_vault(_cfg(vault_path="", backend="fs"))
        assert r.status == "fail"
        assert "SILICA_VAULT" in r.hint
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


def test_check_manifest_absent_is_ok(tmp_path):
    from silica.config import SilicaConfig
    from silica.onboarding.checks import check_manifest

    cfg = SilicaConfig()
    cfg.vault_path = str(tmp_path)
    assert check_manifest(cfg).status == "ok"


def test_check_manifest_unknown_source_warns(tmp_path):
    from silica.config import SilicaConfig
    from silica.onboarding.checks import check_manifest

    (tmp_path / "vault.yaml").write_text("sources: [prose, zotero]\n", encoding="utf-8")
    cfg = SilicaConfig()
    cfg.vault_path = str(tmp_path)
    res = check_manifest(cfg)
    assert res.status == "warn" and "zotero" in res.detail


class TestCheckRerank:
    """The rerank pass was invisible: off by default, undocumented, unchecked.
    Doctor must make its state a fact the user can see, never a silent no-op."""

    def _rr_cfg(self, **kw):
        return _cfg(rerank_base_url="", rerank_model="", **kw)

    def test_extra_installed_reports_ok(self, monkeypatch):
        import silica.onboarding.checks as checks

        monkeypatch.setattr(checks, "has_local_rerank", lambda: True)
        r = checks.check_rerank(self._rr_cfg())
        assert r.status == "ok" and "in-process" in r.detail

    def test_nothing_installed_warns_with_install_hint(self, monkeypatch):
        import silica.onboarding.checks as checks

        monkeypatch.setattr(checks, "has_local_rerank", lambda: False)
        r = checks.check_rerank(self._rr_cfg())
        assert r.status == "warn"
        assert "silica[rerank]" in r.hint

    def test_configured_endpoint_reachable_is_ok(self, monkeypatch):
        import silica.onboarding.checks as checks

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

        monkeypatch.setattr(checks.httpx, "post", lambda *a, **k: FakeResp())
        r = checks.check_rerank(_cfg(rerank_base_url="http://x/v1", rerank_model="m"))
        assert r.status == "ok" and "http://x/v1" in r.detail

    def test_configured_endpoint_down_warns(self, monkeypatch):
        """Never 'fail': a down reranker degrades to the fused pool's order."""
        import silica.onboarding.checks as checks

        def boom(*a, **k):
            raise checks.httpx.ConnectError("refused")

        monkeypatch.setattr(checks.httpx, "post", boom)
        r = checks.check_rerank(_cfg(rerank_base_url="http://x/v1", rerank_model="m"))
        assert r.status == "warn" and "unreachable" in r.detail


class TestCheckLanguage:
    def _seed_italian_notes(self, tmp_path):
        (tmp_path / "n1.md").write_text(
            "Questo è un appunto scritto in italiano con molte parole comuni "
            "come il, la, di, che, per, con, sono, questo, quella.",
            encoding="utf-8",
        )
        (tmp_path / "n2.md").write_text(
            "Un altro appunto: la nota descrive come e perché il sistema funziona, "
            "con esempi e osservazioni sulla struttura.",
            encoding="utf-8",
        )

    def _store_with_lang(self, index_path, lang):
        from silica.kernel.cooccurrence import CooccurStore

        store = CooccurStore(path=index_path, lang=lang)
        store.upsert_note("n1", {"nodes": {}, "edges": []})
        store.save()

    def test_no_vault_is_ok(self):
        from silica.onboarding.checks import check_language
        r = check_language(_cfg(vault_path=""))
        assert r.status == "ok"
        assert "no vault" in r.detail

    def test_no_notes_is_ok(self, tmp_path):
        from silica.onboarding.checks import check_language
        r = check_language(_cfg(vault_path=str(tmp_path)))
        assert r.status == "ok"
        assert "no notes" in r.detail

    def test_no_store_is_ok_and_names_detected_language(self, tmp_path, monkeypatch):
        import silica.kernel.cooccurrence as cooc_mod
        from silica.onboarding.checks import check_language

        self._seed_italian_notes(tmp_path)
        monkeypatch.setattr(cooc_mod, "_index_path_for", lambda vault: tmp_path / "no_such_store.json")

        r = check_language(_cfg(vault_path=str(tmp_path)))
        assert r.status == "ok"
        assert "italian" in r.detail
        assert "no store" in r.detail

    def test_matching_store_is_ok(self, tmp_path, monkeypatch):
        import silica.kernel.cooccurrence as cooc_mod
        from silica.onboarding.checks import check_language

        self._seed_italian_notes(tmp_path)
        index_path = tmp_path / "cooc.json"
        monkeypatch.setattr(cooc_mod, "_index_path_for", lambda vault: index_path)
        self._store_with_lang(index_path, "italian")

        r = check_language(_cfg(vault_path=str(tmp_path)))
        assert r.status == "ok"
        assert "language=italian" in r.detail
        assert "store=italian" in r.detail

    def test_declared_language_supersedes_misfiring_detection(self, tmp_path, monkeypatch):
        """User's bug: a vault DECLARES italian in vault.yaml and its store is
        frozen italian, but a frontmatter-heavy sample makes `detect` say
        english. The declaration is authority — no false 'mismatch' warning."""
        import silica.kernel.cooccurrence as cooc_mod
        from silica.onboarding.checks import check_language, detect_vault_language

        (tmp_path / "n.md").write_text(
            "---\nlast: 2026-07-09\nrelated:\n  - null\n---\nappunto italiano",
            encoding="utf-8",
        )
        # Sanity: without the declaration, this sample is a real detection trap.
        assert detect_vault_language(str(tmp_path)) == "english"
        (tmp_path / "vault.yaml").write_text("cooccurrence_lang: italian\n", encoding="utf-8")
        index_path = tmp_path / "cooc.json"
        monkeypatch.setattr(cooc_mod, "_index_path_for", lambda vault: index_path)
        self._store_with_lang(index_path, "italian")

        r = check_language(_cfg(vault_path=str(tmp_path)))
        assert r.status == "ok"
        assert "language=italian" in r.detail

    def test_mismatched_store_warns_and_suggests_cooccur(self, tmp_path, monkeypatch):
        import silica.kernel.cooccurrence as cooc_mod
        from silica.onboarding.checks import check_language

        self._seed_italian_notes(tmp_path)
        index_path = tmp_path / "cooc.json"
        monkeypatch.setattr(cooc_mod, "_index_path_for", lambda vault: index_path)
        self._store_with_lang(index_path, "english")

        r = check_language(_cfg(vault_path=str(tmp_path)))
        assert r.status == "warn"
        assert "italian" in r.detail and "english" in r.detail
        assert "/cooccur" in r.hint

    def test_corrupt_store_degrades_to_ok_no_traceback(self, tmp_path, monkeypatch):
        import silica.kernel.cooccurrence as cooc_mod
        from silica.onboarding.checks import check_language

        self._seed_italian_notes(tmp_path)
        index_path = tmp_path / "cooc.json"
        index_path.write_text("not json", encoding="utf-8")
        monkeypatch.setattr(cooc_mod, "_index_path_for", lambda vault: index_path)

        r = check_language(_cfg(vault_path=str(tmp_path)))
        assert r.status == "ok"
        assert "italian" in r.detail

    def test_does_not_cross_check_a_different_vaults_frozen_store(self, tmp_path, monkeypatch):
        """Regression for the split-source-of-truth bug: check_language(config) must
        resolve BOTH halves from config.vault_path, never from the global CONFIG
        singleton. Simulates the wizard's step 6 (`run_checks(SilicaConfig())` right
        after a vault switch): global CONFIG still points at an OLD vault with a
        store frozen "english"; the freshly-built `config` passed in points at a
        DIFFERENT, brand-new Italian vault with no store of its own yet. The old
        vault's frozen store must never leak into this vault's verdict.
        """
        import silica.kernel.cooccurrence as cooc_mod
        from silica.config import CONFIG
        from silica.onboarding.checks import check_language

        old_vault = tmp_path / "old_vault"
        old_vault.mkdir()
        old_index_path = tmp_path / "old_cooc.json"
        no_store_path = tmp_path / "no_such_store_for_new_vault.json"
        monkeypatch.setattr(
            cooc_mod, "_index_path_for",
            lambda vault: old_index_path if vault == str(old_vault) else no_store_path,
        )
        self._store_with_lang(old_index_path, "english")
        monkeypatch.setattr(CONFIG, "vault_path", str(old_vault))

        new_vault = tmp_path / "new_vault"
        new_vault.mkdir()
        self._seed_italian_notes(new_vault)

        r = check_language(_cfg(vault_path=str(new_vault)))
        # Must NOT report a mismatch by comparing against old_vault's "english"
        # store — new_vault has no store of its own, so this is the "no store
        # frozen yet" ok state, not a false warn.
        assert r.status == "ok"
        assert "italian" in r.detail
        assert "english" not in r.detail


class TestSampleVaultTextSpread:
    """Finding 2 (final multilingua review): the char budget must be spread
    across up to _LANG_SAMPLE_MAX_FILES files, not exhausted by the first
    handful of alphabetically-sorted ones — otherwise an alphabetical head of
    minority-language files (e.g. "AAA api notes.md") mis-reports the vault's
    dominant language.
    """

    @staticmethod
    def _gen(words: list[str], n: int, seed: int) -> str:
        import random
        rng = random.Random(seed)
        return " ".join(rng.choice(words) for _ in range(n))

    def test_alphabetical_head_minority_does_not_dominate_detection(self, tmp_path):
        from silica.onboarding.checks import detect_vault_language

        en_words = [
            "the", "company", "report", "market", "update", "system", "project",
            "team", "review", "plan", "with", "for", "and", "that", "this",
            "from", "have", "will", "not", "are",
        ]
        it_words = [
            "della", "azienda", "progetto", "sistema", "squadra", "relazione",
            "mercato", "aggiornamento", "con", "per", "che", "questo", "dal",
            "hanno", "sono", "del", "alla", "nella", "sulla", "non",
        ]
        # 4 English files sort first alphabetically, each long enough to fully
        # consume the OLD per-file cap (1000 chars) on their own.
        for i in range(4):
            (tmp_path / f"a{i}_notes.md").write_text(
                self._gen(en_words, 300, seed=i), encoding="utf-8",
            )
        # A larger population of Italian notes sorting after them — the
        # actual majority of the vault.
        for i in range(10):
            (tmp_path / f"z_nota_{i}.md").write_text(
                self._gen(it_words, 300, seed=100 + i), encoding="utf-8",
            )

        assert detect_vault_language(str(tmp_path)) == "italian"


def test_render_report_does_not_eat_bracketed_text(capsys):
    """rich reads a bare [word] as a style tag: unescaped, `silica[rerank]` renders
    as `silica` and the hint tells the user to run the wrong command."""
    from silica.onboarding.checks import CheckResult, render_report

    render_report([CheckResult("rerank", "warn", "disabled", "pip install silica[rerank]")])
    out = capsys.readouterr().out
    assert "silica[rerank]" in out.replace("\n", "")


class TestAggregation:
    def test_run_checks_returns_all_eight(self, monkeypatch, tmp_path):
        import silica.onboarding.checks as checks

        def boom(url, timeout):
            raise checks.httpx.ConnectError("refused")

        monkeypatch.setattr(checks.httpx, "get", boom)
        monkeypatch.setattr(checks.gitstate, "find_repo_root", lambda p: None)
        results = checks.run_checks(_cfg(vault_path=str(tmp_path)))
        assert [r.name for r in results] == [
            "chat model", "chat endpoint", "vault", "vault manifest",
            "language", "embeddings", "rerank", "quarantine",
        ]

    def test_check_quarantine_surfaces_corrupt_files(self, tmp_path):
        from silica.onboarding.checks import check_quarantine

        assert check_quarantine(_cfg(vault_path=str(tmp_path))).status == "ok"
        (tmp_path / "provenance.json.corrupt.20260710T120000").write_text("junk")
        result = check_quarantine(_cfg(vault_path=str(tmp_path)))
        assert result.status == "warn"
        assert "provenance.json.corrupt.20260710T120000" in result.detail

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
