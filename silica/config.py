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
            "SILICA_MODEL", "openrouter/anthropic/claude-sonnet-4-20250514"
        )
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
        default_factory=lambda: os.getenv("SILICA_BANNER_FONT", "slant")
    )

    @property
    def verbose(self) -> bool:
        return self.debug_logging

    @verbose.setter
    def verbose(self, v: bool) -> None:
        self.debug_logging = v


CONFIG = SilicaConfig()
