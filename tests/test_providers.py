from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from silica.agent.providers import get_provider, OpenAICompatibleProvider
from silica.agent.llm import LLMResponse


class DummyConfig:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model


class SchemaModel(BaseModel):
    key: str
    value: int


class TestProviders(unittest.TestCase):
    def test_get_provider_presets(self):
        # Test default preset (lmstudio)
        config_lm = DummyConfig("lmstudio", "my-model")
        provider_lm = get_provider(config_lm)
        self.assertIsInstance(provider_lm, OpenAICompatibleProvider)
        self.assertEqual(provider_lm.model, "my-model")

        # Test openrouter preset
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            config_or = DummyConfig("openrouter", "or-model")
            provider_or = get_provider(config_or)
            self.assertIsInstance(provider_or, OpenAICompatibleProvider)
            self.assertEqual(provider_or.model, "or-model")

    def test_get_provider_custom_uses_config_endpoint(self):
        class CustomConfig:
            provider = "custom"
            model = "custom/my-model"
            provider_base_url = "http://localhost:8000/v1"
            provider_api_key = "sk-local"

        provider = get_provider(CustomConfig())
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertEqual(provider.model, "my-model")  # custom/ prefix stripped
        self.assertIn("localhost:8000", str(provider.client.base_url))
        self.assertEqual(provider.client.api_key, "sk-local")

    def test_get_provider_worker(self):
        class DummyWorkerConfig:
            def __init__(self, provider, model, worker_provider=None, worker_model=None, worker_api_key=None):
                self.provider = provider
                self.model = model
                self.worker_provider = worker_provider
                self.worker_model = worker_model
                self.worker_api_key = worker_api_key

        # 1. Fallback to router when worker not configured
        config_fallback = DummyWorkerConfig("openrouter", "or-model")
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-key"}):
            provider = get_provider(config_fallback, role="worker")
            self.assertEqual(provider.model, "or-model")
            self.assertIn("openrouter.ai", str(provider.client.base_url))

        # 2. Worker explicit preset (openrouter) without overrides
        config_worker_or = DummyWorkerConfig("lmstudio", "lm-model", worker_provider="openrouter", worker_model="worker-or-model")
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "worker-or-key"}):
            provider = get_provider(config_worker_or, role="worker")
            self.assertEqual(provider.model, "worker-or-model")
            self.assertIn("openrouter.ai", str(provider.client.base_url))

        # 3. Worker explicit api-key override (endpoint always from the preset)
        config_worker_override = DummyWorkerConfig(
            "lmstudio", "lm-model",
            worker_provider="openrouter", worker_model="worker-or-model",
            worker_api_key="custom-key"
        )
        provider = get_provider(config_worker_override, role="worker")
        self.assertEqual(provider.model, "worker-or-model")
        self.assertIn("openrouter.ai", str(provider.client.base_url))
        self.assertEqual(provider.client.api_key, "custom-key")


    @patch("openai.OpenAI")
    def test_call_llm_structured_success(self, mock_openai_cls):
        # Setup mock client
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Mock beta.chat.completions.parse
        mock_parsed_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_parsed_obj = SchemaModel(key="test", value=123)
        
        mock_message.content = '{"key": "test", "value": 123}'
        mock_message.parsed = mock_parsed_obj
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_parsed_response.choices = [mock_choice]
        mock_parsed_response.usage = {"prompt_tokens": 10}
        
        mock_client.beta.chat.completions.parse.return_value = mock_parsed_response

        # Execute
        provider = OpenAICompatibleProvider(base_url="http://dummy", api_key="dummy", model="test-model")
        response = provider.call_llm(
            messages=[{"role": "user", "content": "hi"}],
            response_schema=SchemaModel
        )

        # Assertions
        mock_client.beta.chat.completions.parse.assert_called_once()
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.text, '{"key": "test", "value": 123}')

    @patch("openai.OpenAI")
    def test_call_llm_structured_fallback(self, mock_openai_cls):
        # Setup mock client
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # beta.chat.completions.parse raises an exception (not supported, e.g., older server)
        mock_client.beta.chat.completions.parse.side_effect = Exception("Not supported")

        # Non-structured fallback path now streams.
        mock_delta = MagicMock()
        mock_delta.content = '{"key": "fallback", "value": 456}'
        mock_delta.tool_calls = None
        mock_chunk_choice = MagicMock()
        mock_chunk_choice.finish_reason = "stop"
        mock_chunk_choice.delta = mock_delta
        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_chunk_choice]

        mock_client.chat.completions.create.return_value = [mock_chunk]

        # Execute
        provider = OpenAICompatibleProvider(base_url="http://dummy", api_key="dummy", model="test-model")
        response = provider.call_llm(
            messages=[{"role": "user", "content": "hi"}],
            response_schema=SchemaModel
        )

        # Assertions
        mock_client.beta.chat.completions.parse.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.text, '{"key": "fallback", "value": 456}')

    @patch("openai.OpenAI")
    @patch("time.sleep", return_value=None)
    def test_call_llm_retries_on_timeout(self, mock_sleep, mock_openai_cls):
        import openai
        # Setup mock client
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First two calls raise APITimeoutError, third call succeeds.
        # Non-structured path now streams: successful response is an iterable of chunks.
        mock_delta = MagicMock()
        mock_delta.content = "Success after retries"
        mock_delta.tool_calls = None
        mock_chunk_choice = MagicMock()
        mock_chunk_choice.finish_reason = "stop"
        mock_chunk_choice.delta = mock_delta
        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_chunk_choice]
        success_stream = [mock_chunk]

        # Set up side effect to fail twice then succeed
        mock_client.chat.completions.create.side_effect = [
            openai.APITimeoutError(request=MagicMock()),
            openai.APIConnectionError(request=MagicMock(), message="Connection issue"),
            success_stream,
        ]

        provider = OpenAICompatibleProvider(base_url="http://dummy", api_key="dummy", model="test-model")
        response = provider.call_llm(
            messages=[{"role": "user", "content": "hi"}]
        )

        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
        self.assertEqual(response.text, "Success after retries")
        self.assertEqual(mock_sleep.call_count, 2)


def test_presets_are_a_subset_of_known_prefixes():
    """Every preset name must be an auto-prefixable provider prefix, else its
    bare model never gets `provider/` prepended and routing silently breaks."""
    from silica.agent.providers import PROVIDER_PRESETS
    from silica.config import PROVIDER_PREFIXES

    assert set(PROVIDER_PRESETS) <= PROVIDER_PREFIXES


def test_call_llm_custom_routes_via_openai(monkeypatch):
    """A custom/ model reaches litellm as openai/<id> with an explicit api_base,
    since litellm has no `custom/` provider."""
    from silica.agent import llm

    captured: dict = {}

    class _Msg:
        content = "ok"
        tool_calls = None
        reasoning_content = None
        reasoning = None
        thinking_blocks = None

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        usage = None

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    monkeypatch.setattr(llm.CONFIG, "provider_base_url", "http://localhost:9999/v1")
    monkeypatch.setattr(llm.CONFIG, "provider_api_key", "sk-local")

    resp = llm.call_llm("custom/qwen3", [{"role": "user", "content": "hi"}])

    assert captured["model"] == "openai/qwen3"
    assert captured["api_base"] == "http://localhost:9999/v1"
    assert captured["api_key"] == "sk-local"
    assert resp.text == "ok"


def test_call_llm_ollama_routes_via_ollama_chat(monkeypatch):
    """An ollama/ model reaches litellm as ollama_chat/<id> so tool calls use
    /api/chat (native) rather than /api/generate (prompt-emulated)."""
    from silica.agent import llm, providers

    captured: dict = {}

    class _Msg:
        content = "ok"
        tool_calls = None
        reasoning_content = None
        reasoning = None
        thinking_blocks = None

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        usage = None

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    # Don't let clamp_max_tokens probe a live Ollama during the test.
    monkeypatch.setattr(providers, "model_limits", lambda p, m: (0, 0))

    llm.call_llm("ollama/llama3.2:3b", [{"role": "user", "content": "hi"}])

    assert captured["model"] == "ollama_chat/llama3.2:3b"


def test_streaming_path_collects_usage():
    """Non-structured streaming must return real token counts from the final usage chunk."""
    from unittest.mock import MagicMock, patch
    from silica.agent.providers import OpenAICompatibleProvider

    # Simulate: first chunk has content, second (final) chunk has usage
    mock_chunk_content = MagicMock()
    mock_chunk_content.choices = [MagicMock()]
    mock_chunk_content.choices[0].delta.content = "hello"
    mock_chunk_content.choices[0].delta.tool_calls = None
    mock_chunk_content.choices[0].finish_reason = None
    mock_chunk_content.usage = None

    mock_chunk_usage = MagicMock()
    mock_chunk_usage.choices = [MagicMock()]
    mock_chunk_usage.choices[0].delta.content = None
    mock_chunk_usage.choices[0].delta.tool_calls = None
    mock_chunk_usage.choices[0].finish_reason = "stop"
    mock_chunk_usage.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter([mock_chunk_content, mock_chunk_usage])

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.client = mock_client
    provider.model = "test-model"
    provider.base_url = "http://dummy"
    provider.timeout = 30
    provider.max_tokens = 1000

    resp = provider.call_llm([{"role": "user", "content": "hi"}])

    assert resp.text == "hello"
    assert resp.usage.get("prompt_tokens") == 10, f"Expected 10, got: {resp.usage}"
    assert resp.usage.get("completion_tokens") == 5
    assert resp.usage.get("total_tokens") == 15
    # Confirm stream_options was passed to request usage data
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("stream_options") == {"include_usage": True}, \
        f"stream_options not passed: {call_kwargs}"


def test_streaming_path_usage_empty_when_no_usage_chunk():
    """When no streaming chunk carries usage, resp.usage must be {} (not raise)."""
    from unittest.mock import MagicMock
    from silica.agent.providers import OpenAICompatibleProvider

    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "hi"
    mock_chunk.choices[0].delta.tool_calls = None
    mock_chunk.choices[0].finish_reason = "stop"
    mock_chunk.usage = None  # no usage on any chunk

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter([mock_chunk])

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.client = mock_client
    provider.model = "test-model"
    provider.base_url = "http://dummy"
    provider.timeout = 30
    provider.max_tokens = 1000

    resp = provider.call_llm([{"role": "user", "content": "hi"}])
    assert resp.usage == {}, f"Expected empty usage dict, got: {resp.usage}"


@patch("openai.OpenAI")
@patch("time.sleep", return_value=None)
def test_call_llm_retries_on_rate_limit(mock_sleep, mock_openai_cls):
    import openai
    # Setup mock client
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    # First attempt raises RateLimitError, second attempt succeeds.
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Success after rate limit retry"
    mock_chunk.choices[0].delta.tool_calls = None
    mock_chunk.choices[0].finish_reason = "stop"
    mock_chunk.usage = None
    success_stream = [mock_chunk]

    # Set up side effect to fail once with 429 then succeed
    mock_client.chat.completions.create.side_effect = [
        openai.RateLimitError(
            message="Rate limit hit",
            response=MagicMock(status_code=429),
            body={}
        ),
        success_stream,
    ]

    provider = OpenAICompatibleProvider(base_url="http://dummy", api_key="dummy", model="test-model")
    response = provider.call_llm(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert mock_client.chat.completions.create.call_count == 2
    assert response.text == "Success after rate limit retry"
    assert mock_sleep.call_count == 1


@patch("openai.OpenAI")
@patch("time.sleep", return_value=None)
def test_call_llm_structured_rate_limit_propagates_without_fallback(mock_sleep, mock_openai_cls):
    import openai
    # Setup mock client
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    # structured parse raises RateLimitError on all attempts
    mock_client.beta.chat.completions.parse.side_effect = openai.RateLimitError(
        message="Rate limit hit in parse",
        response=MagicMock(status_code=429),
        body={}
    )

    provider = OpenAICompatibleProvider(base_url="http://dummy", api_key="dummy", model="test-model")
    
    # We expect the RateLimitError to propagate out of call_llm once the 429 retry
    # budget (_RATE_LIMIT_ATTEMPTS) is exhausted.
    import pytest
    from silica.agent.llm import _RATE_LIMIT_ATTEMPTS
    with pytest.raises(openai.RateLimitError):
        provider.call_llm(
            messages=[{"role": "user", "content": "hi"}],
            response_schema=SchemaModel
        )

    # Verifies it exhausted the 429 budget and never fell back to chat.completions.create.
    assert mock_client.beta.chat.completions.parse.call_count == _RATE_LIMIT_ATTEMPTS
    assert mock_client.chat.completions.create.call_count == 0
    assert mock_sleep.call_count == _RATE_LIMIT_ATTEMPTS - 1  # backoff between attempts

