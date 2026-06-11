"""Fail-fast model default: no magic hosted model when SILICA_MODEL is unset."""
from __future__ import annotations

import pytest

from silica.config import SilicaConfig


def test_model_defaults_to_empty(monkeypatch):
    monkeypatch.delenv("SILICA_MODEL", raising=False)
    cfg = SilicaConfig()
    assert cfg.model == ""


def test_empty_model_provider_falls_back_to_lmstudio(monkeypatch):
    monkeypatch.delenv("SILICA_MODEL", raising=False)
    monkeypatch.delenv("SILICA_PROVIDER", raising=False)
    cfg = SilicaConfig()
    assert cfg.provider == "lmstudio"


def test_model_configured_guard(monkeypatch):
    from silica import cli
    monkeypatch.setattr(cli.CONFIG, "model", "")
    assert cli._model_configured() is False
    monkeypatch.setattr(cli.CONFIG, "model", "qwen3-30b")
    assert cli._model_configured() is True
