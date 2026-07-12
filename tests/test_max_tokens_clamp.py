# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""clamp_max_tokens: output budget fits both the out_cap and the window."""

from silica.agent import providers


def test_default_is_64k(monkeypatch):
    monkeypatch.delenv("MAX_TOKENS", raising=False)
    monkeypatch.setattr(providers, "model_limits", lambda p, m: (0, 0))
    assert providers.clamp_max_tokens("", "gpt-4", None) == 65536


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("MAX_TOKENS", "8192")
    monkeypatch.setattr(providers, "model_limits", lambda p, m: (0, 0))
    assert providers.clamp_max_tokens("", "gpt-4", None) == 8192


def test_default_clamped_to_out_cap(monkeypatch):
    monkeypatch.delenv("MAX_TOKENS", raising=False)
    monkeypatch.setattr(providers, "model_limits", lambda p, m: (262144, 32000))
    assert providers.clamp_max_tokens("openrouter", "anthropic/claude-sonnet-5", None) == 32000


def test_explicit_below_cap_untouched(monkeypatch):
    monkeypatch.setattr(providers, "model_limits", lambda p, m: (262144, 128000))
    assert providers.clamp_max_tokens("openrouter", "anthropic/claude-sonnet-5", 4096) == 4096


def test_explicit_above_cap_clamped(monkeypatch):
    monkeypatch.setattr(providers, "model_limits", lambda p, m: (262144, 128000))
    assert providers.clamp_max_tokens("openrouter", "anthropic/claude-sonnet-5", 300000) == 128000


def test_window_shrinks_budget(monkeypatch):
    # 8k window, input estimated at 6000 tokens (18000 chars // 3) → 2192 left
    monkeypatch.setattr(providers, "model_limits", lambda p, m: (8192, 0))
    assert providers.clamp_max_tokens("ollama", "llama3", 65536, input_chars=18000) == 2192


def test_window_floor_when_input_fills_it(monkeypatch):
    monkeypatch.setattr(providers, "model_limits", lambda p, m: (8192, 0))
    assert providers.clamp_max_tokens("ollama", "llama3", 65536, input_chars=100000) == 1024
