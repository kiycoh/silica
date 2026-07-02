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

    # LLM provider — litellm model string. Empty by default (fail-fast):
    # the REPL points the user to `silica init` instead of assuming a
    # hosted model whose API key was never mentioned.
    # Examples: "openrouter/anthropic/claude-sonnet-4-20250514", "qwen3-30b"
    model: str = field(
        default_factory=lambda: os.getenv("SILICA_MODEL", "")
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

    # Driver backend: "fs" (default, filesystem-native, headless) or "cli" (Obsidian
    # desktop via CDP — adds version-history rollback and live metadata-cache reads).
    backend: str = field(
        default_factory=lambda: os.getenv("SILICA_BACKEND", "fs")
    )

    # Inbox folder inside the vault — used to archive and blacklist staging files.
    inbox_dir: str = field(
        default_factory=lambda: os.getenv("SILICA_INBOX_DIR", "Inbox")
    )

    # PDF→Markdown converter (ADR-0011 provider seam): "pymupdf4llm" (default,
    # pure-Python, fast on text PDFs) or "mineru" (heavyweight OCR/layout CLI,
    # shelled out as a subprocess).
    pdf_provider: str = field(
        default_factory=lambda: os.getenv("SILICA_PDF_PROVIDER", "pymupdf4llm")
    )

    # Tavily API key for /web-search. Empty → /web-search errors clearly and
    # writes no note. The only new config this feature adds.
    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("SILICA_TAVILY_API_KEY", "")
        or os.getenv("TAVILY_API_KEY", "")
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
    # "auto" (default) detects the vault language from its own text at build time
    # and freezes it into the index; set an explicit Snowball language to override.
    cooccurrence_lang: str = field(
        default_factory=lambda: os.getenv("SILICA_COOCCURRENCE_LANG", "auto")
    )

    # Salience gate (Phase 2.05): concept kept only if cosine(concept, doc_centroid) >= threshold
    sim_threshold_theme: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SIM_THRESHOLD_THEME", "0.35"))
    )
    salience_gate_enabled: bool = field(
        default_factory=lambda: os.getenv("SILICA_SALIENCE_GATE", "True").lower() in ("true", "1", "t")
    )

    # Defer uncorroborated concepts on degraded (embedder-down) extraction.
    # When the embedder is configured but unavailable, the salience gate can't run;
    # with this ON, single-signal (INFERRED) concepts are held back for a later
    # embedder-up pass instead of admitted ungated. Only structurally-corroborated
    # (EXTRACTED) concepts pass — author markup needs no embedder.
    # OFF by default: an embedder-free vault has no "later pass", so deferral there
    # would lose concepts permanently. Turn ON only with a real, occasionally-flaky embedder.
    defer_uncorroborated_concepts: bool = field(
        default_factory=lambda: os.getenv("SILICA_DEFER_UNCORROBORATED", "False").lower() in ("true", "1", "t")
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

    # Recurrence-gated note creation (llm-wiki rule: "create a page when a
    # concept appears in 2+ sources OR is central; never for passing
    # mentions"). PAYLOAD's classify_action requires a no-collision concept to
    # either be structurally corroborated (author markup) or recur (cross-file
    # via CROSSDEDUP, or 2+ times intra-file) before hinting "create"; below
    # the bar it hints "likely_skip" instead. Default 1 = today's behavior
    # (every no-collision concept creates) — the gate only activates above 1.
    min_recurrence_for_create: int = field(
        default_factory=lambda: int(os.getenv("SILICA_MIN_RECURRENCE_FOR_CREATE", "1"))
    )

    @property
    def verbose(self) -> bool:
        return self.debug_logging

    @verbose.setter
    def verbose(self, v: bool) -> None:
        self.debug_logging = v



CONFIG = SilicaConfig()
