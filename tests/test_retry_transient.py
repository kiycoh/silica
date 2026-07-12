# SPDX-License-Identifier: AGPL-3.0-or-later
"""retry_transient: 429s get extra attempts and lift a run-wide pacing floor."""
from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

import silica.agent.llm as llm


class _RateLimit(Exception):
    status_code = 429


class _Transient(Exception):
    pass


@patch("time.sleep", return_value=None)
def test_rate_limit_gets_more_than_three_attempts(mock_sleep):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 4:  # would exhaust the old 3-attempt budget
            raise _RateLimit("429")
        return "ok"

    assert llm.retry_transient(fn, (_RateLimit,)) == "ok"
    assert calls["n"] == 4


@patch("time.sleep", return_value=None)
def test_non_rate_limit_still_exhausts_at_three(mock_sleep):
    def fn():
        raise _Transient("boom")

    with pytest.raises(_Transient):
        llm.retry_transient(fn, (_Transient,))


@patch("time.sleep", return_value=None)
def test_429_lifts_run_cooldown_paced_before_next_call(mock_sleep):
    # First call hits one 429 then succeeds → cooldown lifted off zero.
    seen = {"n": 0}

    def flaky():
        seen["n"] += 1
        if seen["n"] == 1:
            raise _RateLimit("429")
        return "a"

    llm.retry_transient(flaky, (_RateLimit,))
    assert llm._run_cooldown == pytest.approx(llm._COOLDOWN_STEP)

    # A subsequent clean call sleeps the cooldown once before its first attempt.
    mock_sleep.reset_mock()
    llm.retry_transient(lambda: "b", (_RateLimit,))
    mock_sleep.assert_called_once_with(pytest.approx(llm._COOLDOWN_STEP))


@patch("time.sleep", return_value=None)
def test_cooldown_capped(mock_sleep):
    def fn():
        raise _RateLimit("429")

    with pytest.raises(_RateLimit):
        llm.retry_transient(fn, (_RateLimit,))
    assert llm._run_cooldown <= llm._COOLDOWN_CAP


@patch("time.sleep", return_value=None)
def test_cooldown_decays_on_clean_calls(mock_sleep):
    """A clean first-try success halves the 429 cooldown floor (and zeroes it
    below 0.5s) so one bad episode doesn't slow a long-lived GUI server forever."""
    llm._run_cooldown = 8.0
    llm.retry_transient(lambda: "ok", (_Transient,))
    assert llm._run_cooldown == pytest.approx(4.0)

    llm._run_cooldown = 0.4
    llm.retry_transient(lambda: "ok", (_Transient,))
    assert llm._run_cooldown == 0.0


@patch("time.sleep", return_value=None)
def test_cancel_event_stops_retries(mock_sleep):
    """An abandoned call (cancel set) re-raises after the in-flight attempt
    instead of retrying — orphaned workers must not keep hammering the API."""
    cancel = threading.Event()
    cancel.set()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _Transient("boom")

    with pytest.raises(_Transient):
        llm.retry_transient(fn, (_Transient,), cancel=cancel)
    assert calls["n"] == 1


def test_call_llm_forwards_cancel_to_retry():
    """call_llm(cancel=…) reaches retry_transient: a pre-set event means a
    transient litellm error is raised after exactly one completion attempt."""
    import litellm

    cancel = threading.Event()
    cancel.set()
    calls = {"n": 0}

    def boom(**kwargs):
        calls["n"] += 1
        raise litellm.APIConnectionError(
            message="down", llm_provider="openrouter", model="m"
        )

    with patch("litellm.completion", side_effect=boom):
        with pytest.raises(litellm.APIConnectionError):
            llm.call_llm(model="m", messages=[{"role": "user", "content": "hi"}], cancel=cancel)
    assert calls["n"] == 1
