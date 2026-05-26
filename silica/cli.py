"""Silica CLI — the entry point REPL.

From SILICA.md §8.4:
  After `uv pip install -e .`, the command `silica` is in PATH.
  Opens a REPL with prompt_toolkit, runs the agentic loop.
"""
from __future__ import annotations

import logging
import sys

from silica.agent.loop import run_agent
from silica.config import CONFIG
from silica.prompts import SYSTEM_PROMPT
from silica.ui.banner import print_banner
from silica.ui.console import CONSOLE
from silica.ui.prompt import build_session, bottom_toolbar, prompt_text

# Import tools to trigger registration via @tool decorator
import silica.tools.atomic  # noqa: F401
import silica.tools.composed  # noqa: F401
import silica.tools.wrapped  # noqa: F401

logger = logging.getLogger(__name__)


def _setup_logging(debug: bool = False) -> None:
    """Configure logging for the CLI session."""
    CONFIG.debug_logging = debug
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy libraries unless debug logging is requested
    logging.getLogger("httpx").setLevel(level)
    logging.getLogger("litellm").setLevel(level)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR if level == logging.WARNING else level)
    logging.getLogger("openai").setLevel(level)


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
        print(f"  /verbose — cicla tool progress: off → new → all → verbose  (attuale: {CONFIG.tool_progress})")
        print("  /thinking — toggle display dei blocchi reasoning")
        print("  /help    — mostra questo messaggio")
        return True

    if cmd == "/thinking":
        CONFIG.show_thinking = not CONFIG.show_thinking
        state = "on" if CONFIG.show_thinking else "off"
        CONSOLE.print(f"  Thinking display: [bold]{state}[/]")
        return True

    if cmd == "/verbose":
        from typing import Literal
        modes: tuple[Literal["off", "new", "all", "verbose"], ...] = ("off", "new", "all", "verbose")
        current = CONFIG.tool_progress
        next_mode = modes[(modes.index(current) + 1) % len(modes)]
        CONFIG.tool_progress = next_mode
        print(f"  Tool progress: \033[1m{next_mode}\033[0m")
        return True

    print(f"  Comando sconosciuto: {cmd}. Usa /help per la lista.")
    return True


def main():
    """Entry point for the `silica` CLI command."""
    debug_mode = "--verbose" in sys.argv or "-v" in sys.argv or CONFIG.debug_logging
    _setup_logging(debug=debug_mode)

    print_banner()
    CONSOLE.print(f"  Modello: [bold]{CONFIG.model}[/]")
    if CONFIG.vault_name:
        CONSOLE.print(f"  Vault:   [bold]{CONFIG.vault_name}[/]")
    CONSOLE.print()

    session = build_session()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    from silica.agent.progress import make_progress_callback
    callback = make_progress_callback()

    while True:
        try:
            user_input = session.prompt(prompt_text(), bottom_toolbar=bottom_toolbar)
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
            answer = run_agent(messages, model=CONFIG.model, tool_progress_callback=callback)
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
