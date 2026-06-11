"""Tests for the `silica doctor` / `silica init` subcommand dispatch."""
from __future__ import annotations

import pytest


class TestDispatchSubcommand:
    def test_no_subcommand_returns_none(self):
        from silica.cli import _dispatch_subcommand
        assert _dispatch_subcommand([]) is None
        assert _dispatch_subcommand(["--something"]) is None

    def test_doctor_ok_returns_zero(self, monkeypatch):
        import silica.onboarding.checks as checks
        from silica.cli import _dispatch_subcommand
        ok = checks.CheckResult("chat model", "ok", "x")
        monkeypatch.setattr(checks, "run_checks", lambda cfg: [ok])
        monkeypatch.setattr(checks, "render_report", lambda results: None)
        assert _dispatch_subcommand(["doctor"]) == 0

    def test_doctor_failure_returns_one(self, monkeypatch):
        import silica.onboarding.checks as checks
        from silica.cli import _dispatch_subcommand
        bad = checks.CheckResult("vault", "fail", "missing", "run `silica init`")
        monkeypatch.setattr(checks, "run_checks", lambda cfg: [bad])
        monkeypatch.setattr(checks, "render_report", lambda results: None)
        assert _dispatch_subcommand(["doctor"]) == 1

    def test_init_delegates_to_wizard(self, monkeypatch):
        import silica.onboarding.wizard as wizard_mod
        from silica.cli import _dispatch_subcommand
        monkeypatch.setattr(wizard_mod, "run_wizard", lambda: 0)
        assert _dispatch_subcommand(["init"]) == 0
