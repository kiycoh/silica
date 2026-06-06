"""Distiller resilience: dynamic output budget (#2) + partial-output salvage (#1).

These cover the two structural fixes for the truncated-JSON failure mode:

  #1  A truncated distiller response no longer kills the whole batch — the
      complete `updates` entries are salvaged from the malformed JSON.
  #2  `max_tokens` is computed from the actual prompt size and the model's
      context window instead of a hardcoded ceiling, so dense batches get the
      full available output headroom.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from silica.agent.llm import LLMResponse
from silica.kernel.prep_delegation import (
    compute_distiller_max_tokens,
    estimate_prompt_tokens,
    run_distiller,
    salvage_distiller_json,
)


# ---------------------------------------------------------------------------
# #2 — dynamic max_tokens math (pure functions)
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_estimate_uses_four_chars_per_token(self):
        assert estimate_prompt_tokens("x" * 400) == 100

    def test_estimate_rounds_up_partial_token(self):
        assert estimate_prompt_tokens("xyz") == 1

    def test_budget_uses_available_headroom_when_no_ceiling(self):
        # 400 chars => 100 prompt tokens; 10000 - 100 - 500 = 9400
        budget = compute_distiller_max_tokens(
            "x" * 400, context_window=10000, safety_margin=500, ceiling=0
        )
        assert budget == 9400

    def test_ceiling_caps_available_headroom(self):
        budget = compute_distiller_max_tokens(
            "x" * 400, context_window=10000, safety_margin=500, ceiling=2000
        )
        assert budget == 2000

    def test_ceiling_ignored_when_above_available(self):
        budget = compute_distiller_max_tokens(
            "x" * 400, context_window=10000, safety_margin=500, ceiling=999999
        )
        assert budget == 9400

    def test_floor_applies_when_prompt_exceeds_window(self):
        # Degenerate: prompt bigger than window => clamp to the floor, not negative.
        budget = compute_distiller_max_tokens(
            "x" * 8000, context_window=100, safety_margin=500, ceiling=0
        )
        assert budget == 1024


# ---------------------------------------------------------------------------
# #1 — partial-output salvage (pure function)
# ---------------------------------------------------------------------------

class TestSalvage:
    def test_recovers_complete_updates_from_truncated_array(self):
        raw = (
            '{"main_thematic_axes": ["a", "b"], "updates": ['
            '{"heading": "X", "op": "skip", "reason": "r"}, '
            '{"heading": "Y", "op": "write", "snippet": "incompl'
        )
        out = salvage_distiller_json(raw)
        assert out is not None
        assert out["main_thematic_axes"] == ["a", "b"]
        assert len(out["updates"]) == 1
        assert out["updates"][0]["heading"] == "X"

    def test_recovers_multiple_complete_updates(self):
        raw = (
            '{"updates": ['
            '{"heading": "A", "op": "skip"}, '
            '{"heading": "B", "op": "skip"}, '
            '{"heading": "C", "op": "wr'
        )
        out = salvage_distiller_json(raw)
        assert out is not None
        assert [u["heading"] for u in out["updates"]] == ["A", "B"]

    def test_returns_none_when_no_updates_key(self):
        assert salvage_distiller_json('{"main_thematic_axes": ["a"]}') is None

    def test_returns_none_when_no_complete_update(self):
        raw = '{"updates": [{"heading": "X", "op": "wri'
        assert salvage_distiller_json(raw) is None

    def test_handles_nested_braces_and_strings(self):
        raw = (
            '{"updates": ['
            '{"heading": "X", "snippet": "a }] {[ b", "meta": {"k": [1, 2]}}, '
            '{"heading": "Y"'
        )
        out = salvage_distiller_json(raw)
        assert out is not None
        assert len(out["updates"]) == 1
        assert out["updates"][0]["snippet"] == "a }] {[ b"


# ---------------------------------------------------------------------------
# #1 — run_distiller integration: truncation is non-fatal
# ---------------------------------------------------------------------------

class TestRunDistillerSalvage:
    @patch("silica.agent.providers.get_provider")
    def test_truncated_length_response_salvages_partial(self, mock_get_provider):
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        truncated = (
            '{"main_thematic_axes": ["x"], "updates": ['
            '{"heading": "Done", "op": "skip", "source_basename": "f.md"}, '
            '{"heading": "Cut", "op": "write", "snippet": "half'
        )
        mock_provider.call_llm.return_value = LLMResponse(
            text=truncated, finish_reason="length"
        )

        result = run_distiller(
            payload={"schema_version": 1, "batches": []},
            target="1 Cultura/Test",
        )

        assert "error" not in result
        assert len(result["updates"]) == 1
        assert result["updates"][0]["heading"] == "Done"

    @patch("silica.agent.providers.get_provider")
    def test_unrecoverable_truncation_returns_error(self, mock_get_provider):
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        # First update already truncated => nothing complete to salvage.
        mock_provider.call_llm.return_value = LLMResponse(
            text='{"updates": [{"heading": "Cut", "op": "wri',
            finish_reason="length",
        )

        result = run_distiller(
            payload={"schema_version": 1, "batches": []},
            target="1 Cultura/Test",
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# #2 — run_distiller wires a dynamic max_tokens
# ---------------------------------------------------------------------------

class TestRunDistillerDynamicBudget:
    @patch("silica.agent.providers.get_provider")
    def test_passes_headroom_above_old_ceiling(self, mock_get_provider, monkeypatch):
        monkeypatch.setenv("MODEL_CONTEXT_WINDOW", "262144")
        monkeypatch.delenv("DISTILLER_MAX_TOKENS", raising=False)
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_provider.call_llm.return_value = LLMResponse(
            text=json.dumps({"updates": []})
        )

        run_distiller(payload={"schema_version": 1, "batches": []}, target="t")

        sent = mock_provider.call_llm.call_args.kwargs["max_tokens"]
        # No longer artificially pinned at the old 32768; uses real headroom.
        assert sent > 32768
        assert sent <= 262144

    @patch("silica.agent.providers.get_provider")
    def test_explicit_ceiling_is_respected(self, mock_get_provider, monkeypatch):
        monkeypatch.setenv("MODEL_CONTEXT_WINDOW", "262144")
        monkeypatch.setenv("DISTILLER_MAX_TOKENS", "5000")
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_provider.call_llm.return_value = LLMResponse(
            text=json.dumps({"updates": []})
        )

        run_distiller(payload={"schema_version": 1, "batches": []}, target="t")

        sent = mock_provider.call_llm.call_args.kwargs["max_tokens"]
        assert sent == 5000
