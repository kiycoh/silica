"""Silica CLI — the entry point REPL.

From SILICA.md §8.4:
  After `uv pip install -e .`, the command `silica` is in PATH.
  Opens a REPL with prompt_toolkit, runs the agentic loop.
"""
from __future__ import annotations

import json
import logging
import sys

from rich.markdown import Markdown

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
    
    # Reset existing handlers to configure cleanly
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    if debug:
        from silica.ui.logging import HumanFriendlyFormatter
        formatter = HumanFriendlyFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)

    # LiteLLM/httpx/openai are always silenced — their DEBUG is raw HTTP/request dumps
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)


def _handle_direct_shortcut(raw_input: str, messages: list[dict]) -> bool:
    """Execute read-only commands directly without an LLM round-trip.

    Operates on the raw (case-preserved) input so that query strings and file
    paths reach the tool with their original casing intact.  Returns True if
    the command was handled, False to fall through to the normal dispatch.

    Handled commands (immediate, synchronous):
        /status [run_id]
        /embed [folder] [--force]
        /graph [output.html] [folder]
        /find <query> [--k=N]
    """
    from silica.tools import TOOLS

    parts = raw_input.strip().split()
    if not parts:
        return False
    cmd = parts[0].lower()

    if cmd == "/status":
        run_id = parts[1] if len(parts) > 1 else ""
        result = TOOLS["silica_ledger_digest"].run(run_id=run_id)
        try:
            parsed = json.loads(result)
            digest = parsed.get("digest", result)
            CONSOLE.print(Markdown(str(digest)))
        except Exception:
            CONSOLE.print(result)
        return True

    if cmd == "/embed":
        folder = ""
        force = False
        for part in parts[1:]:
            if part == "--force":
                force = True
            elif part.startswith("--folder="):
                folder = part[len("--folder="):]
            elif not part.startswith("-"):
                folder = part
        result = TOOLS["silica_embed_refresh"].run(folder=folder, force=force)
        try:
            parsed = json.loads(result)
            CONSOLE.print(f"  Indexed: [bold]{parsed.get('indexed', '?')}[/] / {parsed.get('total_notes', '?')} notes")
            if parsed.get("read_errors"):
                CONSOLE.print(f"  [yellow]Read errors:[/] {parsed['read_errors']}")
        except Exception:
            CONSOLE.print(result)
        return True

    if cmd == "/graph":
        output_path = "graph.html"
        folder = ""
        positional = [p for p in parts[1:] if not p.startswith("-")]
        if positional:
            output_path = positional[0]
        if len(positional) > 1:
            folder = positional[1]
        result = TOOLS["silica_graph_export"].run(output_path=output_path, folder=folder)
        try:
            parsed = json.loads(result)
            CONSOLE.print(f"  Graph written to: [bold]{parsed.get('output_path', output_path)}[/]")
        except Exception:
            CONSOLE.print(result)
        return True

    if cmd == "/find":
        k = 5
        tokens: list[str] = []
        for part in parts[1:]:
            if part.startswith("--k="):
                try:
                    k = int(part[4:])
                except ValueError:
                    pass
            else:
                tokens.append(part)  # preserve original case
        query = " ".join(tokens)
        if not query:
            CONSOLE.print("  Usage: /find <query> [--k=N]")
            return True
        result = TOOLS["silica_semantic_search"].run(query=query, k=k)
        try:
            parsed = json.loads(result)
            results = parsed.get("results", [])
            if results:
                CONSOLE.print(f"  Results for [bold]{query}[/] (top {len(results)}):")
                for r in results:
                    score = r.get("score", 0.0)
                    path = r.get("path", r.get("name", "?"))
                    CONSOLE.print(f"    [{score:.3f}] {path}")
            elif "error" in parsed:
                CONSOLE.print(f"  [yellow]{parsed['error']}[/]")
            else:
                CONSOLE.print(f"  No results for '{query}'.")
        except Exception:
            CONSOLE.print(result)
        return True

    return False


def _expand_workflow_shortcut(user_input: str) -> str | None:
    """Expand workflow shortcuts (e.g. /report, /inject) into agent-directed messages.

    Returns the expanded message string, or None if the input is not a
    recognised shortcut. Expanded messages flow through the normal agentic
    loop so the agent calls the tools and follows the steering protocol.

    Syntax:
        /report [folder] [--top-k=N] [--embeddings]
        /inject <file...> --target=DIR [--hub=H]

    Examples:
        /report
        /report Concepts/ML
        /report --top-k=15 --embeddings
        /report Inbox --embeddings
        /inject Inbox/notes.md --target=Concepts/AI
        /inject Inbox/a.md Inbox/b.md --target=Concepts/AI --hub=AI
    """
    parts = user_input.strip().split()
    if not parts:
        return None

    cmd = parts[0].lower()

    if cmd == "/inject":
        args = parts[1:]
        inbox_files: list[str] = []
        target_dir = ""
        hub = ""
        for arg in args:
            if arg.startswith("--target="):
                target_dir = arg[len("--target="):]
            elif arg.startswith("--hub="):
                hub = arg[len("--hub="):]
            elif not arg.startswith("-"):
                inbox_files.append(arg)  # preserve original case
        if not inbox_files:
            return "Error: /inject requires at least one file path. Usage: /inject <file...> --target=DIR"
        if not target_dir:
            return "Error: /inject requires --target=DIR. Usage: /inject <file...> --target=DIR"
        files_json = json.dumps(inbox_files)
        msg = (
            f"Run the Injector pipeline for {len(inbox_files)} file(s).\n"
            f"Call `silica_run_injector` with "
            f"inbox_files={files_json}, target_dir={json.dumps(target_dir)}"
        )
        if hub:
            msg += f", hub={json.dumps(hub)}"
        msg += "."
        return msg

    if cmd == "/report":
        args = parts[1:]
        folder = ""
        top_k = 10
        with_embeddings = False

        for arg in args:
            if arg.startswith("--folder="):
                folder = arg[len("--folder="):]
            elif arg.startswith("--top-k="):
                try:
                    top_k = int(arg[len("--top-k="):])
                except ValueError:
                    pass
            elif arg in ("--embeddings", "--with-embeddings"):
                with_embeddings = True
            elif not arg.startswith("-"):
                folder = arg  # positional: /report Concepts/ML

        scope_desc = f"scoped to `{folder}`" if folder else "on the whole vault"
        embed_note = " Also propose missing links via the embedding index." if with_embeddings else ""

        return (
            f"Run a structural vault audit {scope_desc}.{embed_note}\n"
            f"Call `silica_vault_report` with "
            f"folder={json.dumps(folder)}, top_k={top_k}, "
            f"with_embeddings={'true' if with_embeddings else 'false'}, seed_ledger=true. "
            f"Then follow the steering loop exactly as described in your instructions: "
            f"call `silica_ledger_next` repeatedly, execute each capability with its payload, "
            f"call `silica_ledger_update` after each one, and stop when the plan returns done."
        )

    return None


def _handle_slash_command(cmd: str, messages: list[dict]) -> bool:
    """Handle slash commands. Returns True if the command was handled."""
    cmd = cmd.strip().lower()

    if cmd in ("/exit", "/quit", "/q"):
        return False  # Signal to exit

    if cmd == "/model":
        CONSOLE.print(f"  Current model: [bold]{CONFIG.model}[/]")
        return True

    if cmd == "/tools":
        from silica.tools import TOOLS
        if not TOOLS:
            CONSOLE.print("  No tools registered.")
        else:
            CONSOLE.print(f"  [bold]{len(TOOLS)} registered tools:[/]")
            for name, t in sorted(TOOLS.items()):
                CONSOLE.print(f"    [dim]\\[{t.cls}][/] {name}")
        return True

    if cmd == "/help":
        CONSOLE.print("  [bold cyan]/exit[/]            — exit silica")
        CONSOLE.print("  [bold cyan]/model[/]           — show current LLM model")
        CONSOLE.print("  [bold cyan]/tools[/]           — list registered tools")
        CONSOLE.print("  [bold cyan]/clear[/]           — reset conversation history")
        CONSOLE.print(f"  [bold cyan]/verbose[/]         — cycle tool progress: off → new → all → verbose  [dim](current: {CONFIG.tool_progress})[/]")
        CONSOLE.print("  [bold cyan]/thinking[/]        — toggle reasoning block display")
        CONSOLE.print("  [bold cyan]/help[/]            — show this help message")
        CONSOLE.print()
        CONSOLE.print("  [bold yellow]Workflow shortcuts[/]  [dim](agent-directed)[/]")
        CONSOLE.print("  [bold cyan]/report[/] [dim][[folder] [--top-k=N] [--embeddings]][/]")
        CONSOLE.print("     Structural vault audit → steering loop (auto-fix orphans, surface bridges, escalate dangling links)")
        CONSOLE.print("     Examples: [dim]/report[/]  [dim]/report Concepts/ML[/]  [dim]/report --embeddings[/]")
        CONSOLE.print()
        CONSOLE.print("  [bold cyan]/inject[/] [dim]<file...> --target=DIR [--hub=H][/]")
        CONSOLE.print("     Ingest one or more inbox files via the Injector FSM (multi-file, per-chunk containment)")
        CONSOLE.print("     Examples: [dim]/inject Inbox/notes.md --target=Concepts/AI[/]")
        CONSOLE.print("               [dim]/inject Inbox/a.md Inbox/b.md --target=Concepts/AI --hub=AI[/]")
        CONSOLE.print()
        CONSOLE.print("  [bold yellow]Direct commands[/]  [dim](immediate, no LLM round-trip)[/]")
        CONSOLE.print("  [bold cyan]/status[/] [dim][[run_id]][/]")
        CONSOLE.print("     Show progress digest for the given run (latest run if omitted)")
        CONSOLE.print("  [bold cyan]/embed[/] [dim][[folder] [--force]][/]")
        CONSOLE.print("     Build or refresh the embedding index  [dim](--force: re-embed all)[/]")
        CONSOLE.print("  [bold cyan]/graph[/] [dim][[output.html] [folder]][/]")
        CONSOLE.print("     Export a vis.js knowledge graph to an HTML file")
        CONSOLE.print("     Example: [dim]/graph Out.html Concepts/AI[/]")
        CONSOLE.print("  [bold cyan]/find[/] [dim]<query> [--k=N][/]")
        CONSOLE.print("     Semantic search — find vault notes similar to the query text")
        CONSOLE.print("     Example: [dim]/find Neural Networks --k=10[/]")
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
        CONSOLE.print(f"  Tool progress: [bold]{next_mode}[/]")

        if next_mode == "verbose":
            _setup_logging(debug=True)
            CONSOLE.print("  System log level: [bold]DEBUG[/]")
        else:
            _setup_logging(debug=False)
            CONSOLE.print("  System log level: [bold]WARNING[/]")

        return True

    CONSOLE.print(f"  Unknown command: {cmd}. Use [bold cyan]/help[/] to see options.")
    return True


def main():
    """Entry point for the `silica` CLI command."""
    debug_mode = "--verbose" in sys.argv or "-v" in sys.argv or CONFIG.debug_logging
    _setup_logging(debug=debug_mode)

    print_banner()
    CONSOLE.print(f"  Model: [bold]{CONFIG.model}[/]")
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
            print("\n  Goodbye.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Direct shortcuts bypass the LLM entirely (case-sensitive args preserved)
        if user_input.startswith("/") and _handle_direct_shortcut(user_input, messages):
            continue

        # Expand workflow shortcuts (/report, /inject etc.) into agent-directed messages
        expanded = _expand_workflow_shortcut(user_input)
        if expanded is not None:
            user_input = expanded

        # Handle slash commands
        if user_input.startswith("/"):
            cmd = user_input.strip().lower()
            if cmd == "/clear":
                CONSOLE.clear()
                print_banner()
                CONSOLE.print(f"  Model: [bold]{CONFIG.model}[/]")
                if CONFIG.vault_name:
                    CONSOLE.print(f"  Vault:   [bold]{CONFIG.vault_name}[/]")
                CONSOLE.print()

                messages.clear()
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
                session = build_session()
                continue

            should_continue = _handle_slash_command(user_input, messages)
            if not should_continue:
                print("  Goodbye.")
                break
            continue

        # Normal user message → agentic loop
        messages.append({"role": "user", "content": user_input})

        try:
            answer = run_agent(messages, model=CONFIG.model, tool_progress_callback=callback)
            if answer:
                CONSOLE.print()
                CONSOLE.print(Markdown(answer))
                CONSOLE.print()
            messages.append({"role": "assistant", "content": answer or ""})
        except KeyboardInterrupt:
            CONSOLE.print("\n  [dim](interrupted)[/]")
        except Exception as e:
            logger.exception("Agent error")
            CONSOLE.print(f"\n  [bold red]Error:[/] {e}\n")


if __name__ == "__main__":
    main()
