from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable
import openai
import orjson
from pydantic import BaseModel

from silica.agent.llm import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

PROVIDER_PRESETS = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio"
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY"
    }
}


@runtime_checkable
class Provider(Protocol):
    def call_llm(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def call_llm(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if response_schema:
            try:
                response = self.client.beta.chat.completions.parse(
                    **kwargs,
                    response_format=response_schema
                )
                choice = response.choices[0]
                message = choice.message
                finish_reason = getattr(choice, "finish_reason", None)
                
                parsed_object = message.parsed
                content_str = message.content if message.content else ""
                if not content_str and parsed_object:
                    content_str = orjson.dumps(parsed_object.model_dump()).decode("utf-8")
                
                assistant_msg = {"role": "assistant"}
                if message.content:
                    assistant_msg["content"] = message.content
                if message.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ]
                
                parsed_calls = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        try:
                            args = orjson.loads(tc.function.arguments)
                        except Exception:
                            args = {}
                        parsed_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
                
                return LLMResponse(
                    text=content_str,
                    tool_calls=parsed_calls,
                    assistant_message=assistant_msg,
                    usage=dict(response.usage) if response.usage else {},
                    reasoning=getattr(message, "reasoning_content", None),
                    finish_reason=finish_reason,
                )
            except Exception as e:
                logger.warning("Constrained decoding failed, falling back to non-structured: %s", e)

        # Non-structured fallback
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", None)
        
        assistant_msg = {"role": "assistant"}
        if message.content:
            assistant_msg["content"] = message.content
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        
        parsed_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = orjson.loads(tc.function.arguments)
                except Exception:
                    args = {}
                parsed_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        
        return LLMResponse(
            text=message.content,
            tool_calls=parsed_calls,
            assistant_message=assistant_msg,
            usage=dict(response.usage) if response.usage else {},
            reasoning=getattr(message, "reasoning_content", None),
            finish_reason=finish_reason,
        )


def get_provider(config: Any) -> Provider:
    provider_name = getattr(config, "provider", "lmstudio")
    model_name = getattr(config, "model", "")
    
    preset = PROVIDER_PRESETS.get(provider_name)
    if not preset:
        preset = PROVIDER_PRESETS["lmstudio"]
        
    base_url = preset["base_url"]
    api_key = preset.get("api_key", "lm-studio")
    if "api_key_env" in preset:
        api_key = os.getenv(preset["api_key_env"], "dummy-key")
        
    if provider_name == "openrouter" and model_name.startswith("openrouter/"):
        model_name = model_name.removeprefix("openrouter/")
    elif provider_name == "lmstudio" and model_name.startswith("lmstudio/"):
        model_name = model_name.removeprefix("lmstudio/")
        
    return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model_name)
