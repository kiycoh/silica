"""Silica configuration — model, vault, provider settings.

Configuration is loaded from (in order of precedence):
  1. Environment variables (SILICA_MODEL, SILICA_VAULT, etc.)
  2. .env file in the project root
  3. Hardcoded defaults

The config module is imported early and provides a singleton CONFIG object
that the rest of the codebase reads from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env from the working directory (or project root)
_dotenv_path = Path.cwd() / ".env"
if not _dotenv_path.exists():
    _dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_dotenv_path, override=False)


@dataclass
class SilicaConfig:
    """Runtime configuration singleton."""

    # LLM provider — litellm model string.
    # Examples: "openrouter/anthropic/claude-sonnet-4-20250514",
    #           "anthropic/claude-sonnet-4-20250514",
    #           "openai/gpt-4o"
    model: str = field(
        default_factory=lambda: os.getenv(
            "SILICA_MODEL", "openrouter/google/gemma-4-31b-it"
        )
    )

    # Provider preset name (derived from model prefix by default, or overridden)
    _provider: str | None = field(
        default_factory=lambda: os.getenv("SILICA_PROVIDER", None)
    )

    @property
    def provider(self) -> str:
        if self._provider is not None:
            return self._provider
        if self.model and "/" in self.model:
            prefix = self.model.split("/", 1)[0]
            if prefix in ("openrouter", "lmstudio"):
                return prefix
        return "lmstudio"

    @provider.setter
    def provider(self, val: str) -> None:
        self._provider = val

    # --- Sub-agent worker model (leashed sub-agents run on a separate, smaller model) ---
    # The router (agent loop) uses `model`/`provider` above; sub-agents (dedup, refiner)
    # use these worker_* fields so they can run concurrently on a small local model.
    worker_model: str | None = field(
        default_factory=lambda: os.getenv("SILICA_WORKER_MODEL", None)
    )
    # Worker provider preset name; falls back to "lmstudio" when unset.
    worker_provider: str | None = field(
        default_factory=lambda: os.getenv("SILICA_WORKER_PROVIDER", None)
    )
    # Explicit endpoint overrides for the worker model (default → local LM Studio).
    worker_base_url: str | None = field(
        default_factory=lambda: os.getenv("SILICA_WORKER_BASE_URL", None)
    )
    worker_api_key: str | None = field(
        default_factory=lambda: os.getenv("SILICA_WORKER_API_KEY", None)
    )


    # Execution caps — bound how far a sub-agent can move before the framework stops it.
    subagent_max_turns: int = field(
        default_factory=lambda: int(os.getenv("SILICA_SUBAGENT_MAX_TURNS", "6"))
    )
    subagent_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SUBAGENT_TIMEOUT_S", "120"))
    )
    subagent_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("SILICA_SUBAGENT_MAX_CONCURRENT", "3"))
    )
    # Global ceiling on concurrent worker-model LLM calls (the one true
    # concurrency budget; see ADR-0004). Sized to the worker backend
    # (API rate limit or local GPU slots).
    worker_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("SILICA_WORKER_MAX_CONCURRENT", "4"))
    )
    # Master switch: when False, silica_inject runs the legacy single-FSM path.
    subagents_enabled: bool = field(
        default_factory=lambda: os.getenv("SILICA_SUBAGENTS_ENABLED", "True").lower() in ("true", "1", "t")
    )

    # Vault path — used by the fs backend and for context.
    vault_path: str = field(
        default_factory=lambda: os.getenv("SILICA_VAULT", "")
    )

    # Obsidian CLI vault name (for multi-vault setups).
    vault_name: str = field(
        default_factory=lambda: os.getenv("SILICA_VAULT_NAME", "")
    )

    # Driver backend: "cli" (default, requires Obsidian desktop) or "fs" (headless).
    backend: str = field(
        default_factory=lambda: os.getenv("SILICA_BACKEND", "cli")
    )

    # Inbox folder inside the vault — used to archive and blacklist staging files.
    inbox_dir: str = field(
        default_factory=lambda: os.getenv("SILICA_INBOX_DIR", "Inbox")
    )

    # Maximum context tokens before the agent warns.
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("SILICA_MAX_CONTEXT", "60000"))
    )

    # Tool progress display level (REPL-runtime, ciclabile con /verbose)
    # off     — silenzio totale, solo risposta finale
    # new     — mostra il nome del tool solo quando cambia
    # all     — ogni tool call con preview degli args (default)
    # verbose — args completi, risultato troncato, durata
    tool_progress: Literal["off", "new", "all", "verbose"] = field(
        default_factory=lambda: os.getenv("SILICA_TOOL_PROGRESS", "all")  # type: ignore
    )

    # Debug logging su stderr (--verbose / -v flag CLI, non ciclabile)
    debug_logging: bool = field(
        default_factory=lambda: os.getenv("SILICA_VERBOSE", "False").lower() in ("true", "1", "t")
    )

    # Mostra i blocchi di reasoning del modello (toggle a runtime con /thinking)
    show_thinking: bool = field(
        default_factory=lambda: os.getenv("SILICA_SHOW_THINKING", "True").lower() in ("true", "1", "t")
    )

    # Font pyfiglet del banner di avvio
    banner_font: str = field(
        default_factory=lambda: os.getenv("SILICA_BANNER_FONT", "tarty1")
    )

    # Runtime session state — updated by cli.py after each agent turn
    context_tokens: int = 0

    # Stile del banner di avvio (wordmark, minimal)
    banner_style: Literal["wordmark", "minimal"] = field(
        default_factory=lambda: os.getenv("SILICA_BANNER_STYLE", "wordmark")  # type: ignore
    )

    # Embedding model — used by silica/kernel/embed.py (Phase 3)
    # Example: "qwen3-embedding-8b" for LM Studio, "text-embedding-3-small" for OpenAI
    embedding_model: str = field(
        default_factory=lambda: os.getenv("SILICA_EMBEDDING_MODEL", "qwen3-embedding-4b")
    )

    # Base URL for the embeddings endpoint (defaults to the same LM Studio endpoint)
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv("SILICA_EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    )

    # API key for embeddings endpoint (usually same as chat, or "lm-studio" for local)
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("SILICA_EMBEDDING_API_KEY", "lm-studio")
    )

    # Cosine similarity thresholds for dedup routing (Phase 5)
    # score >= sim_threshold_high → strong duplicate → patch existing note
    # score <= sim_threshold_low  → clearly new concept → write new note
    # between the two → ambiguous → deferred store
    sim_threshold_high: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SIM_THRESHOLD_HIGH", "0.85"))
    )
    sim_threshold_low: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SIM_THRESHOLD_LOW", "0.65"))
    )

    # Number of candidates to retrieve per note during dedup scan.
    # Higher values increase recall at negligible BLAS cost (search is a single
    # matrix-vector product). k=1 misses borderline secondary matches when the
    # primary match lands above τ_high and is discarded.
    dedup_scan_k: int = field(
        default_factory=lambda: int(os.getenv("SILICA_DEDUP_SCAN_K", "5"))
    )

    # Minimum title-only cosine similarity to promote a pair into the dedup
    # borderline window, regardless of the full-note score.
    # Set higher than sim_threshold_low (0.65) to avoid spurious matches between
    # generically related titles (e.g. "Python" / "Python async").
    sim_title_threshold: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SIM_TITLE_THRESHOLD", "0.80"))
    )

    # Language for the co-occurrence graph stemmer + stopwords (kernel/cooccurrence.py).
    # Default English; "italian" and ~25 Snowball languages are supported.
    cooccurrence_lang: str = field(
        default_factory=lambda: os.getenv("SILICA_COOCCURRENCE_LANG", "english")
    )

    # Salience gate (Phase 2.05): concept kept only if cosine(concept, doc_centroid) >= threshold
    sim_threshold_theme: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SIM_THRESHOLD_THEME", "0.35"))
    )
    salience_gate_enabled: bool = field(
        default_factory=lambda: os.getenv("SILICA_SALIENCE_GATE", "True").lower() in ("true", "1", "t")
    )

    # Image handling mode:
    #   strip (default) — remove image embeds from text before embedding / LLM context
    #   vlm             — replace embeds with VLM-generated descriptions (requires vlm_model)
    image_mode: Literal["strip", "vlm"] = field(
        default_factory=lambda: os.getenv("SILICA_IMAGE_MODE", "strip")  # type: ignore
    )

    # VLM model used when image_mode="vlm" (litellm model string).
    # Example: "openai/gpt-4o-mini", "openrouter/google/gemini-flash-1.5"
    vlm_model: str = field(
        default_factory=lambda: os.getenv("SILICA_VLM_MODEL", "")
    )

    # Hard timeout (seconds) for each individual Obsidian CLI subprocess call.
    # The CDP bridge should respond in < 1 s normally; 8 s gives headroom for
    # slow machines and large notes without allowing 88-second hangs.
    # Override via SILICA_OBSIDIAN_CLI_TIMEOUT if you hit false-positive timeouts.
    obsidian_cli_timeout: float = field(
        default_factory=lambda: float(os.getenv("SILICA_OBSIDIAN_CLI_TIMEOUT", "8"))
    )

    domain: str | None = field(
        default_factory=lambda: os.getenv("SILICA_DOMAIN") or None
    )

    # Git commit safety net for docs/ writes. "off" (default) → never commit;
    # "auto" → after each write batch, commit the touched docs/ paths with a
    # structured message. Additive to the undo journal (ADR-0002), never a
    # replacement. Only takes effect when the vault sits inside a git repo.
    git_commit: Literal["off", "auto"] = field(
        default_factory=lambda: os.getenv("SILICA_GIT_COMMIT", "off")  # type: ignore
    )

    @property
    def verbose(self) -> bool:
        return self.debug_logging

    @verbose.setter
    def verbose(self, v: bool) -> None:
        self.debug_logging = v



CONFIG = SilicaConfig()
