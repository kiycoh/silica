from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from silica.config import SilicaConfig
from silica.agent.providers import get_provider, OpenAICompatibleProvider
from silica.agent.llm import LLMResponse, ToolCall, call_llm
from silica.agent.loop import run_agent, _is_tool_failure
from silica.tools import TOOLS, Tool
from silica.kernel.prep_delegation import run_distiller


class TestAgentGuardsAndProvider(unittest.TestCase):
    def test_dynamic_provider_derivation(self):
        # Default scenario
        config = SilicaConfig()
        config.model = "openrouter/google/gemma-4-31b-it"
        config._provider = None
        self.assertEqual(config.provider, "openrouter")

        # Explicit override in env or field
        config._provider = "lmstudio"
        self.assertEqual(config.provider, "lmstudio")

        # Different model string prefix
        config._provider = None
        config.model = "lmstudio/anthropic/claude"
        self.assertEqual(config.provider, "lmstudio")

        # Non-matching prefix falls back to lmstudio
        config.model = "anthropic/claude"
        self.assertEqual(config.provider, "lmstudio")

    @patch("litellm.completion")
    def test_max_tokens_and_finish_reason_in_litellm(self, mock_completion):
        # Mock choice and message with finish_reason
        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message.content = "Truncated text"
        mock_choice.message.tool_calls = None
        
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = {}
        
        mock_completion.return_value = mock_resp

        # Call with max_tokens
        resp = call_llm(model="test_model", messages=[], max_tokens=100)

        # Assertions
        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args[1]
        self.assertEqual(kwargs.get("max_tokens"), 100)
        self.assertEqual(resp.finish_reason, "length")

    @patch("openai.OpenAI")
    def test_max_tokens_and_finish_reason_in_provider(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message.content = "Truncated structured"
        mock_choice.message.parsed = None
        mock_choice.message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = {}

        mock_client.chat.completions.create.return_value = mock_response

        # Instantiate provider
        provider = OpenAICompatibleProvider(base_url="http://dummy", api_key="dummy", model="test-model")
        resp = provider.call_llm(messages=[], max_tokens=4000)

        # Check call arguments
        mock_client.chat.completions.create.assert_called_once()
        kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs.get("max_tokens"), 4000)
        self.assertEqual(resp.finish_reason, "length")

    @patch("silica.agent.providers.get_provider")
    def test_run_distiller_fails_on_length(self, mock_get_provider):
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        # Setup length finish reason
        mock_response = MagicMock()
        mock_response.finish_reason = "length"
        mock_response.text = '{"updates": []}'
        mock_provider.call_llm.return_value = mock_response

        payload = {"schema_version": 1, "batches": []}
        res = run_distiller(payload=payload, target="TargetFolder")

        # Verify it returns error dictionary
        self.assertIn("error", res)
        self.assertIn("maximum tokens limit", res["error"])

    def test_is_tool_failure(self):
        # Dict representation
        self.assertTrue(_is_tool_failure({"error": "something"}))
        self.assertFalse(_is_tool_failure({"success": True}))

        # String representation
        self.assertTrue(_is_tool_failure('{"error": "bad stuff"}'))
        self.assertTrue(_is_tool_failure('An error occurred'))
        self.assertTrue(_is_tool_failure('Failed to read file'))
        self.assertTrue(_is_tool_failure('Exception raised'))
        self.assertFalse(_is_tool_failure('All systems operational'))

    @patch("silica.agent.loop.call_llm")
    def test_convergence_guard_trigger(self, mock_call_llm):
        from pydantic import BaseModel

        class DummyParams(BaseModel):
            param: str

        def failing_fn(param: str):
            return '{"error": "Execution failed"}'

        failing_tool = Tool(
            fn=failing_fn,
            name="failing_tool",
            description="Fails consistently",
            params_model=DummyParams,
            cls="atomic"
        )

        # Mock LLM to always call the same tool with same args
        resp1 = LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="tc1", name="failing_tool", args={"param": "value"})],
            assistant_message={"role": "assistant", "tool_calls": []},
            usage={}
        )
        mock_call_llm.return_value = resp1

        with patch.dict(TOOLS, {"failing_tool": failing_tool}):
            messages = [{"role": "user", "content": "run the tool"}]
            
            # Since N=3 causes RuntimeError, running the agent should raise RuntimeError
            with self.assertRaises(RuntimeError) as context:
                run_agent(messages, model="test_model")
            
            self.assertIn("ha fallito 3 volte consecutivamente", str(context.exception))

            # Inspect history to check that the warning message was injected at consecutive failure #2
            system_messages = [m for m in messages if m.get("role") == "system"]
            self.assertTrue(any("NON richiamare questo tool con gli stessi identici argomenti" in m["content"] for m in system_messages))
