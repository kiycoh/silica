# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""call_llm temperature passthrough: the eval harness needs temperature=0 for
reproducible A/Bs (a byte-identical prompt flipped correct->wrong across runs
at the provider default). Product callers omit it and are unaffected."""
from __future__ import annotations

from unittest.mock import patch

from silica.agent.llm import call_llm

from tests.llm_mocks import litellm_mock_response as _mock_completion


def test_call_llm_forwards_temperature():
    mock_resp = _mock_completion(text="ok")
    with patch("litellm.completion", return_value=mock_resp) as mock_lit:
        call_llm(model="lmstudio/test-model",
                 messages=[{"role": "user", "content": "test"}],
                 temperature=0.0)
    assert mock_lit.call_args[1]["temperature"] == 0.0


def test_call_llm_default_omits_temperature():
    mock_resp = _mock_completion(text="ok")
    with patch("litellm.completion", return_value=mock_resp) as mock_lit:
        call_llm(model="lmstudio/test-model",
                 messages=[{"role": "user", "content": "test"}])
    assert "temperature" not in mock_lit.call_args[1]
