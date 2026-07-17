# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from __future__ import annotations

import importlib.util
import logging
import os
from functools import lru_cache
from typing import Any
import httpx
import openai
import orjson
from pydantic import BaseModel

from silica.agent.llm import LLMResponse, ToolCall, openrouter_routing, retry_transient

logger = logging.getLogger(__name__)

PROVIDER_PRESETS = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"  # Ollama ignores it; the OpenAI SDK demands non-empty.
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY"
    }
}


SILICA_CLI_OPEN = "<silica-cli>"
SILICA_CLI_CLOSE = "</silica-cli>"


# ponytail: cached per process — reload the model in LM Studio with a different
# window and silica needs a restart to see it; TTL cache if that ever bites.
@lru_cache(maxsize=None)
def model_limits(provider: str, model: str) -> tuple[int, int]:
    """(context_window, max_output_tokens) as reported by the live provider.

    lmstudio   → GET {base}/api/v0/models: `loaded_context_length` (the window
                 the model is loaded with RIGHT NOW, often below its max) with
                 `max_context_length` as fallback. No output cap.
    ollama     → POST {base}/api/show: `num_ctx` from the Modelfile parameters
                 (the loaded window) if pinned, else the trained max from
                 `model_info["<arch>.context_length"]`. No output cap.
    openrouter → GET /api/v1/models: `context_length` plus the top provider's
                 `max_completion_tokens` (often far below the window — e.g.
                 qwen3-8b: 131k ctx, 8k out).

    (0, 0) means unknown/unreachable: callers keep their static defaults.
    """
    try:
        if provider == "ollama":
            base = PROVIDER_PRESETS["ollama"]["base_url"].removesuffix("/v1")
            wanted = model.removeprefix("ollama/")
            info = httpx.post(f"{base}/api/show", json={"model": wanted}, timeout=5.0).json()
            # Prefer num_ctx if the Modelfile pins it (the window actually loaded).
            # ponytail: Ollama's *default* num_ctx (~4096) isn't reported in /api/show,
            # so an unpinned model reports its trained max here — cap it with
            # SILICA_MAX_CONTEXT or `PARAMETER num_ctx` if that overshoots the load.
            params = info.get("parameters") or ""
            num_ctx = next((int(f[1]) for p in params.splitlines()
                            if (f := p.split())[:1] == ["num_ctx"] and len(f) > 1), 0)
            mi = info.get("model_info") or {}
            arch = mi.get("general.architecture", "")
            max_ctx = mi.get(f"{arch}.context_length") or next(
                (v for k, v in mi.items() if k.endswith(".context_length")), 0)
            return int(num_ctx or max_ctx or 0), 0
        if provider == "lmstudio":
            base = PROVIDER_PRESETS["lmstudio"]["base_url"].removesuffix("/v1")
            data = httpx.get(f"{base}/api/v0/models", timeout=5.0).json()["data"]
            wanted = model.removeprefix("lmstudio/")
            entry = next(m for m in data if m["id"] == wanted)
            window = entry.get("loaded_context_length") or entry.get("max_context_length") or 0
            return int(window), 0
        if provider == "openrouter":
            data = httpx.get("https://openrouter.ai/api/v1/models", timeout=5.0).json()["data"]
            wanted = model.removeprefix("openrouter/")
            entry = next(m for m in data if m["id"] == wanted)
            out_cap = (entry.get("top_provider") or {}).get("max_completion_tokens") or 0
            return int(entry.get("context_length") or 0), int(out_cap)
    except Exception as e:
        logger.debug("model_limits(%s, %s) unavailable: %s", provider, model, e)
    return (0, 0)


def clamp_max_tokens(provider: str, model: str, requested: int | None, input_chars: int = 0) -> int:
    """Output-token budget for a request: the caller's ask (or the MAX_TOKENS
    default), never above the provider's live max_completion_tokens, never
    above the window space left after the input.

    Providers validate input + max_tokens <= context window upfront; without
    this clamp a default above either limit makes them reject the request
    (e.g. claude-sonnet-5 on OpenRouter: 128k output cap, 262144 window).
    input_chars is the serialized request size; // 3 overestimates its token
    count (English runs ~4 chars/token, JSON/code closer to 3), which errs on
    the side of a smaller output budget.
    """
    want = requested if requested is not None else int(os.getenv("MAX_TOKENS", "65536"))
    window, out_cap = model_limits(provider, model)
    if out_cap:
        want = min(want, out_cap)
    if window:
        # ponytail: floor 1024 keeps the request well-formed when input nearly
        # fills the window; compaction is the real defense at that point.
        want = min(want, max(window - input_chars // 3, 1024))
    return want


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
        self.base_url = base_url
        self.model = model

    def call_llm(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        openrouter_provider: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_wire(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
        provider = "openrouter" if "openrouter.ai" in self.base_url else ""
        input_chars = len(str(kwargs["messages"])) + (len(str(tools)) if tools else 0)
        kwargs["max_tokens"] = clamp_max_tokens(provider, self.model, max_tokens, input_chars)

        if "openrouter.ai" in self.base_url and (rt := openrouter_routing(openrouter_provider)):
            kwargs["extra_body"] = rt

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


class Reranker:
    """Cross-encoder reranker over a served /rerank endpoint.

    Speaks the de-facto protocol (llama.cpp --rerank, Infinity, Jina, Cohere):
    ``POST {model, query, documents} -> {results: [{index, relevance_score}]}``.
    A cross-encoder scores query x document *jointly* — the biggest precision
    lever retrieval has after first-stage recall — so it is used to reorder an
    already-fused candidate pool, never to retrieve.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 5.0):
        self.url = base_url.rstrip("/") + "/rerank"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    def scores(self, query: str, documents: list[str]) -> list[float] | None:
        """Relevance score per document in input order, or None to abstain.

        Abstains (None) on any transport or response-shape failure so the caller
        keeps its prior ordering rather than dropping candidates. The short
        timeout keeps a slow reranker from stalling an interactive path.
        """
        if not query or not documents:
            return None
        try:
            resp = httpx.post(
                self.url,
                json={"model": self.model, "query": query, "documents": documents},
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results")
            if not isinstance(results, list):
                return None
            scored = [0.0] * len(documents)
            for r in results:
                i = r.get("index")
                if isinstance(i, int) and 0 <= i < len(documents):
                    scored[i] = float(r.get("relevance_score", r.get("score", 0.0)))
            return scored
        except Exception as e:
            logger.debug("rerank abstained: %s", e)
            return None


# Multilingual by design: a vault is whatever language its owner writes in
# (conventions.language), and bge-reranker-base is English/Chinese only.
LOCAL_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def has_local_rerank() -> bool:
    """Whether the optional [rerank] extra is installed. find_spec, not import:
    get_reranker runs per query and importing torch costs seconds."""
    return importlib.util.find_spec("sentence_transformers") is not None


@lru_cache(maxsize=1)
def _load_cross_encoder(model: str) -> Any:
    """Load the cross-encoder, cached for the process lifetime.

    Cached because get_reranker() is called per query: a fresh CrossEncoder per
    recall would reload ~2GB of weights every time. First call downloads to the
    HF cache (like the [pdf] extra's models).
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model)


class LocalReranker:
    """Cross-encoder reranker in-process, via the optional [rerank] extra.

    Exists because the reranker is the one leg of the stack with nowhere to run:
    LM Studio and Ollama serve generative and embedding models, and a cross-encoder
    is neither — it scores a [query, document] PAIR jointly rather than embedding
    texts independently, so those runtimes either refuse it or (LM Studio) coerce it
    into an embedding model whose output is meaningless for ranking. Without this
    class the only path is a llama-server the user starts and maintains by hand,
    which is why rerank silently never ran for anyone but the eval harness.

    Duck-types Reranker: same .scores() contract, same abstention, so every call
    site and the reorder in kernel/rerank stay untouched.
    """

    def __init__(self, model: str = LOCAL_RERANK_MODEL):
        self.model = model

    def scores(self, query: str, documents: list[str]) -> list[float] | None:
        """Relevance score per document in input order, or None to abstain.

        Abstains on any failure (missing weights, no disk, OOM) so a broken local
        reranker degrades to the fused pool's order, exactly as a down endpoint does.
        """
        if not query or not documents:
            return None
        try:
            encoder = _load_cross_encoder(self.model)
            return [float(s) for s in encoder.predict([[query, d] for d in documents])]
        except Exception as e:
            logger.debug("local rerank abstained: %s", e)
            return None


def get_reranker(config: Any) -> Reranker | LocalReranker | None:
    """Return a reranker: a served /rerank endpoint if configured, else in-process
    if the [rerank] extra is installed, else None (disabled).

    The endpoint wins when set so the eval harness keeps pinning its own llama-server.
    """
    base_url = getattr(config, "rerank_base_url", "")
    model = getattr(config, "rerank_model", "")
    if base_url and model:
        return Reranker(
            base_url=base_url,
            model=model,
            api_key=getattr(config, "rerank_api_key", ""),
        )
    if not has_local_rerank():
        return None
    return LocalReranker(model=model or LOCAL_RERANK_MODEL)


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

    # Strip the preset prefix (openrouter/lmstudio/ollama) — the OpenAI-compatible
    # endpoint wants the bare model id. No-op when the model carries no prefix.
    # For openrouter this drops only the leading "openrouter/", keeping "vendor/model".
    model_name = model_name.removeprefix(f"{provider_name}/")

    return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model_name)
