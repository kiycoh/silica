from __future__ import annotations

import logging
import os
from typing import Any
import httpx
import openai
import orjson
from pydantic import BaseModel

from silica.agent.llm import LLMResponse, ToolCall, retry_transient

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


SILICA_CLI_OPEN = "<silica-cli>"
SILICA_CLI_CLOSE = "</silica-cli>"


def _to_wire(msg: dict) -> dict:
    """Strip internal provenance and render the CLI marker for the wire.

    `origin` is an internal-only field; the OpenAI message object rejects
    unknown fields, so it must never reach the SDK. When ``origin == "cli"``
    the content is wrapped in <silica-cli> markers so the model can tell a
    harness directive apart from a human turn. Messages without ``origin``
    (the common case) are returned unchanged.
    """
    if "origin" not in msg:
        return msg
    origin = msg["origin"]
    wire = {k: v for k, v in msg.items() if k != "origin"}
    if origin == "cli" and wire.get("content"):
        wire["content"] = f"{SILICA_CLI_OPEN}{wire['content']}{SILICA_CLI_CLOSE}"
    return wire


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, model: str):
        # Granular timeouts: connect=10s, read=45s per-chunk (streaming inactivity watchdog).
        # The read timeout applies to each received chunk, so a frozen stream that stops
        # producing tokens raises APITimeoutError after 45s — triggering the retry loop.
        _timeout = httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=5.0)
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=_timeout)
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
            "messages": [_to_wire(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
        kwargs["max_tokens"] = max_tokens if max_tokens is not None else int(os.getenv("MAX_TOKENS", "256000"))

        def _execute_call() -> LLMResponse:
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
                    
                    assistant_msg: dict[str, Any] = {"role": "assistant"}
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
                except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError):
                    # Re-raise transient/network/rate-limit errors to be retried in the outer loop
                    raise
                except Exception as e:
                    logger.warning("Constrained decoding failed: %s", e)
                    # If the error is due to truncation/parsing failure, try to salvage the partial content
                    completion = getattr(e, "completion", None)
                    if completion and completion.choices:
                        msg = completion.choices[0].message
                        content_str = msg.content or ""
                        finish_reason = getattr(completion.choices[0], "finish_reason", None)
                        if content_str:
                            logger.info("Extracted partial response text (len=%d) from parsing error.", len(content_str))
                            return LLMResponse(
                                text=content_str,
                                tool_calls=[],
                                assistant_message={"role": "assistant", "content": content_str},
                                usage=dict(completion.usage) if getattr(completion, "usage", None) else {},
                                finish_reason=finish_reason or "length",
                            )
                    logger.warning("No partial content salvageable, falling back to non-structured")

            # Non-structured path: stream so the httpx read-timeout acts as a
            # per-chunk inactivity watchdog rather than a total-body deadline.
            # stream_options is ignored by providers that don't support it.
            stream = self.client.chat.completions.create(
                **kwargs, stream=True, stream_options={"include_usage": True}
            )
            content_chunks: list[str] = []
            tc_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None
            usage_dict: dict[str, int] = {}

            for chunk in stream:
                if not chunk.choices:
                    continue
                _choice = chunk.choices[0]
                finish_reason = _choice.finish_reason or finish_reason
                delta = _choice.delta
                if delta.content:
                    content_chunks.append(delta.content)
                if delta.tool_calls:
                    for _tc in delta.tool_calls:
                        _i = _tc.index
                        if _i not in tc_acc:
                            tc_acc[_i] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if _tc.id:
                            tc_acc[_i]["id"] = _tc.id
                        if _tc.function:
                            if _tc.function.name:
                                tc_acc[_i]["function"]["name"] += _tc.function.name
                            if _tc.function.arguments:
                                tc_acc[_i]["function"]["arguments"] += _tc.function.arguments
                if getattr(chunk, "usage", None) is not None:
                    u = chunk.usage
                    usage_dict = {
                        "prompt_tokens": getattr(u, "prompt_tokens", 0),
                        "completion_tokens": getattr(u, "completion_tokens", 0),
                        "total_tokens": getattr(u, "total_tokens", 0),
                    }

            content = "".join(content_chunks) or None
            tool_calls_list = [tc_acc[k] for k in sorted(tc_acc)]

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if content:
                assistant_msg["content"] = content
            if tool_calls_list:
                assistant_msg["tool_calls"] = tool_calls_list

            parsed_calls = []
            for _tc in tool_calls_list:
                try:
                    args = orjson.loads(_tc["function"]["arguments"])
                except Exception:
                    args = {}
                parsed_calls.append(
                    ToolCall(id=_tc["id"], name=_tc["function"]["name"], args=args)
                )

            return LLMResponse(
                text=content,
                tool_calls=parsed_calls,
                assistant_message=assistant_msg,
                usage=usage_dict,
                finish_reason=finish_reason,
            )

        return retry_transient(
            _execute_call,
            (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError),
        )


class OpenAIEmbedder:
    """Thin wrapper for the OpenAI-compatible /v1/embeddings endpoint.

    Uses the same SDK already present in the project. Suitable for any
    provider that speaks the OpenAI API (LM Studio, OpenRouter, etc.).
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        # Mirror the LLM provider's hardening: a granular read-timeout turns a
        # frozen embedding server (e.g. a cold/contended local model) into a
        # fast failure instead of an indefinite hang, and max_retries=1 stops
        # the SDK's default 2 silent retries from stacking 60s waits. COLLISION
        # is best_effort, so a bounded failure degrades to "skip dedup" rather
        # than freezing the run.
        _timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)
        self.client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=_timeout, max_retries=1
        )
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        Vectors are normalised by most embedding models; cosine similarity is
        therefore equivalent to dot-product for those models.
        """
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        # The API guarantees ordering matches the input list
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def get_embedder(config: Any) -> OpenAIEmbedder:
    """Return an embedder configured from SilicaConfig."""
    return OpenAIEmbedder(
        base_url=getattr(config, "embedding_base_url", "http://localhost:1234/v1"),
        api_key=getattr(config, "embedding_api_key", "lm-studio"),
        model=getattr(config, "embedding_model", "qwen3-embedding-8b"),
    )


def get_provider(config: Any, role: str = "router") -> OpenAICompatibleProvider:
    """Return an LLM provider for the given role.

    role="router" (default) → uses config.provider / config.model (the main model).
    role="worker"            → uses config.worker_provider / config.worker_model so
                               leashed sub-agents can run on a separate small model.

    When the worker role specifies an explicit worker_api_key it wins over the
    preset; the endpoint always comes from the worker_provider preset.
    """
    if role == "worker":
        provider_name = getattr(config, "worker_provider", None)
        model_name = getattr(config, "worker_model", None)
        if not provider_name or not model_name:
            provider_name = getattr(config, "provider", "lmstudio")
            model_name = getattr(config, "model", "")
            role = "router"
    else:
        provider_name = getattr(config, "provider", "lmstudio")
        model_name = getattr(config, "model", "")

    preset = PROVIDER_PRESETS.get(provider_name)
    if not preset:
        preset = PROVIDER_PRESETS["lmstudio"]

    base_url = preset["base_url"]
    api_key = preset.get("api_key", "lm-studio")
    if "api_key_env" in preset:
        api_key = os.getenv(preset["api_key_env"], "dummy-key")

    # Worker role: explicit api-key override takes precedence over the preset.
    if role == "worker":
        api_key = getattr(config, "worker_api_key", None) or api_key

    if provider_name == "openrouter" and model_name.startswith("openrouter/"):
        model_name = model_name.removeprefix("openrouter/")
    elif provider_name == "lmstudio" and model_name.startswith("lmstudio/"):
        model_name = model_name.removeprefix("lmstudio/")

    return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model_name)
