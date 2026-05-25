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

    # Maximum context tokens before the agent warns.
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("SILICA_MAX_CONTEXT", "60000"))
    )

    # Verbose / debug mode active
    verbose: bool = field(
        default_factory=lambda: os.getenv("SILICA_VERBOSE", "False").lower() in ("true", "1", "t")
    )


CONFIG = SilicaConfig()
