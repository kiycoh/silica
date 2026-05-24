"""Silica CLI — the entry point REPL.

From SILICA.md §8.4:
  After `uv pip install -e .`, the command `silica` is in PATH.
  Opens a REPL with prompt_toolkit, runs the agentic loop.
"""
from __future__ import annotations

import logging
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from silica.agent.loop import run_agent
from silica.config import CONFIG
from silica.prompts import SYSTEM_PROMPT

# Import tools to trigger registration via @tool decorator
import silica.tools.atomic  # noqa: F401
import silica.tools.composed  # noqa: F401

logger = logging.getLogger(__name__)

BANNER = """\
\033[1;36m╭─────────────────────────────────────────╮
│  silica v0.1.0 — agente Obsidian-nativo │
│  /exit per uscire · /model per cambiare │
╰─────────────────────────────────────────╯\033[0m"""


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI session."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _handle_slash_command(cmd: str, messages: list[dict]) -> bool:
    """Handle slash commands. Returns True if the command was handled."""
    cmd = cmd.strip().lower()

    if cmd in ("/exit", "/quit", "/q"):
        return False  # Signal to exit

    if cmd == "/model":
        print(f"  Modello attuale: \033[1m{CONFIG.model}\033[0m")
        return True

    if cmd == "/tools":
        from silica.tools import TOOLS
        if not TOOLS:
            print("  Nessun tool registrato.")
        else:
            print(f"  \033[1m{len(TOOLS)} tool registrati:\033[0m")
            for name, t in sorted(TOOLS.items()):
                print(f"    [{t.cls}] {name}")
        return True

    if cmd == "/clear":
        messages.clear()
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        print("  Conversazione resettata.")
        return True

    if cmd == "/help":
        print("  /exit    — esci da silica")
        print("  /model   — mostra il modello LLM attuale")
        print("  /tools   — elenca i tool registrati")
        print("  /clear   — resetta la conversazione")
        print("  /verbose — attiva logging dettagliato")
        print("  /help    — mostra questo messaggio")
        return True

    if cmd == "/verbose":
        _setup_logging(verbose=True)
        print("  Logging verbose attivato.")
        return True

    print(f"  Comando sconosciuto: {cmd}. Usa /help per la lista.")
    return True


def main():
    """Entry point for the `silica` CLI command."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    _setup_logging(verbose=verbose)

    print(BANNER)
    print(f"  Modello: \033[1m{CONFIG.model}\033[0m")
    if CONFIG.vault_name:
        print(f"  Vault:   \033[1m{CONFIG.vault_name}\033[0m")
    print()

    session = PromptSession(history=InMemoryHistory())
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = session.prompt("silica> ")
        except (EOFError, KeyboardInterrupt):
            print("\n  Arrivederci.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            should_continue = _handle_slash_command(user_input, messages)
            if not should_continue:
                print("  Arrivederci.")
                break
            continue

        # Normal user message → agentic loop
        messages.append({"role": "user", "content": user_input})

        try:
            answer = run_agent(messages, model=CONFIG.model)
            if answer:
                print(f"\n{answer}\n")
            messages.append({"role": "assistant", "content": answer or ""})
        except KeyboardInterrupt:
            print("\n  (interrotto)")
        except Exception as e:
            logger.exception("Agent error")
            print(f"\n  \033[1;31mErrore: {e}\033[0m\n")


if __name__ == "__main__":
    main()
