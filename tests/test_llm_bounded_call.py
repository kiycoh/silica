"""Wall-clock bound around a hanging LLM call.

Regression: litellm's own `timeout` kwarg did not fire on an OpenRouter read-hang
(process wedged ~58min on one call). `_bounded` is the timeout we control.
"""
import threading

import litellm
import pytest

from silica.agent.llm import _bounded


def test_bounded_raises_timeout_on_hang():
    release = threading.Event()
    try:
        with pytest.raises(litellm.Timeout):
            _bounded(lambda: release.wait(30), 0.15, "openrouter/deepseek/deepseek-v4-flash")
    finally:
        release.set()  # let the abandoned worker exit instead of blocking 30s


def test_bounded_returns_value_when_fast():
    assert _bounded(lambda: 42, 5.0, "m/x") == 42


def test_bounded_propagates_worker_exception():
    def boom():
        raise ValueError("upstream error")

    with pytest.raises(ValueError, match="upstream error"):
        _bounded(boom, 5.0, "m/x")
