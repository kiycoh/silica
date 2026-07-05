"""Silica CLI — the entry point REPL.

From SILICA.md §8.4:
  After `uv pip install -e .`, the command `silica` is in PATH.
  Opens a REPL with prompt_toolkit, runs the agentic loop.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import sys
from typing import NamedTuple

from silica.ui.style import FlatMarkdown

from silica.agent.loop import run_agent
from silica.config import CONFIG
from silica.prompts import SYSTEM_PROMPT
from silica.ui.console import CONSOLE
from silica.ui.home import print_home
from silica.ui.prompt import build_session, bottom_toolbar, prompt_text

# Import tools to trigger registration via @tool decorator
import silica.tools.atomic  # noqa: F401
import silica.tools.composed  # noqa: F401
import silica.tools.wrapped  # noqa: F401
import silica.tools.codedocs_tool  # noqa: F401
import silica.tools.delegate_tool  # noqa: F401
import silica.sources.web_research  # noqa: F401  (registers the web_search tool)

logger = logging.getLogger(__name__)


def _update_context_tokens(messages: list[dict]) -> None:
    try:
        import litellm
        CONFIG.context_tokens = litellm.token_counter(model=CONFIG.model, messages=messages)
    except Exception:
        CONFIG.context_tokens = sum(len(m.get("content") or "") for m in messages) // 4


def _inject_vault_map(messages: list[dict]) -> None:
    """Appends the vault map as a system message (best-effort).

    CoALA recall: loads the corpus self-model into working memory at session
    start so the agent doesn't rediscover the vault via tools. The map is a
    startup snapshot; this session's writes already live in working memory.
    # ponytail: recomputed once per session; no storage/refresh.
    """
    try:
        from silica.kernel.vault_map import build_vault_map

        vault_map = build_vault_map()
        if vault_map:
            messages.append({"role": "system", "content": vault_map})
    except Exception as exc:
        logger.debug("vault map injection skipped: %s", exc)


def _fresh_messages() -> list[dict]:
    """Seed a fresh conversation: system prompt + vault map + token count.

    Single source of truth for the initial state, shared by session start and
    /clear so the two can't drift.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    _inject_vault_map(messages)
    _update_context_tokens(messages)
    return messages


def _setup_logging(debug: bool = False) -> None:
    """Configure logging for the CLI session."""
    import threading
    CONFIG.debug_logging = debug
    level = logging.DEBUG if debug else logging.WARNING

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler: logging.Handler
    if debug:
        from rich.logging import RichHandler
        from silica.ui.logging import HumanFriendlyFormatter, LiveAwareStreamHandler
        handler = RichHandler(
            console=CONSOLE,
            markup=True,
            show_path=False,
            show_level=False,
            show_time=False,
        )
        handler.setFormatter(HumanFriendlyFormatter())
        # Rich's Live display is driven from the main thread; worker threads logging
        # through RichHandler concurrently corrupt the terminal render state.
        # Restrict RichHandler to the main thread only.
        main_thread = threading.main_thread()
        handler.addFilter(lambda r: threading.current_thread() is main_thread)

        # Worker-thread records fall back to a live-aware stderr handler: resolving
        # sys.stderr at emit time follows rich.Live's redirect, so they print above
        # an active live region instead of tearing it (stale-frame duplication).
        bg_handler = LiveAwareStreamHandler()
        bg_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        bg_handler.addFilter(lambda r: threading.current_thread() is not main_thread)
        root.addHandler(bg_handler)
    else:
        from silica.ui.logging import LiveAwareStreamHandler
        # Live-aware: follows rich.Live's stderr redirect so warnings during the
        # injector/batch live region print above it instead of tearing the panel.
        handler = LiveAwareStreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(handler)
    root.setLevel(level)

    # LiteLLM/httpx/openai/httpcore are always silenced — their DEBUG is raw HTTP/request dumps
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)


class VaultTarget(NamedTuple):
    """Outcome of resolving a runtime ``/vault <arg>`` switch.

    ``vault`` is the absolute path to adopt (None on error). ``created`` is
    True when the codebase has no ``.silica/`` yet and the caller must mkdir it.
    ``error`` carries a human-readable message when the arg cannot be adopted.
    """
    vault: str | None
    created: bool
    error: str | None


def resolve_vault_switch(arg: str) -> VaultTarget:
    """Resolve a ``/vault <arg>`` target, codebase-aware (pure, read-only I/O).

    Mirrors the startup repo-mode rule for the runtime command: if ``arg`` lives
    inside a git repo, the vault is that repo's ``.silica/`` (memory node) rather
    than the repo root itself — created on demand. A path that is already a
    ``.silica`` dir, or any plain (non-codebase) directory, is adopted verbatim.
    """
    from pathlib import Path
    from silica.kernel import gitstate

    target = Path(arg).expanduser()
    # An explicit .silica dir is already a vault root — adopt it verbatim.
    if target.is_dir() and target.name == ".silica":
        return VaultTarget(str(target.resolve()), False, None)
    # Codebase? Adopt <repo>/.silica, creating it on demand.
    repo = gitstate.find_repo_root(target) if target.exists() else None
    if repo is not None:
        silica_dir = repo / ".silica"
        return VaultTarget(str(silica_dir), not silica_dir.is_dir(), None)
    # Plain directory → literal vault (preserves pre-codebase behaviour).
    if target.is_dir():
        return VaultTarget(str(target.resolve()), False, None)
    return VaultTarget(None, False, f"Not a directory: {arg}")


def default_user_vault(home=None):
    """Stable per-user vault used when no explicit SILICA_VAULT and no repo
    mode applies. Sits alongside ~/.silica/{ledger,undo_journal,checkpoints}.db.
    """
    from pathlib import Path

    return (home or Path.home()) / ".silica" / "vault"


def resolve_repo_mode_vault(cwd, vault_env: str, docs_exists_ok: bool, self_repo=None):
    """Pure resolver for repo-mode vault selection (testable, no I/O prompts).

    Returns the vault path string to adopt, or None to leave config unchanged.
    - Explicit SILICA_VAULT (vault_env truthy) always wins → None.
    - Not inside a git repo → None.
    - Inside Silica's own source repo (root == self_repo) → None: that's dev
      mode, not a vault. Caller falls back to the home default.
    - .silica/ exists → return it.
    - .silica/ missing → return it only if docs_exists_ok (caller already
      confirmed creation); otherwise None.
    """
    from pathlib import Path
    from silica.kernel import gitstate

    if vault_env.strip():
        return None
    root = gitstate.find_repo_root(cwd)
    if root is None:
        return None
    if self_repo is not None and root == Path(self_repo):
        return None
    vault_dir = Path(root) / ".silica"
    if vault_dir.is_dir():
        return str(vault_dir)
    if docs_exists_ok:
        return str(vault_dir)
    return None


def _activate_repo_mode() -> None:
    """Side-effecting startup vault selection. Explicit SILICA_VAULT wins; else
    a *user* project repo → its .silica/ (prompted if absent); else — including
    inside Silica's own source repo (dev mode) — a stable ~/.silica/vault."""
    from pathlib import Path
    from silica.kernel import gitstate
    import silica

    if CONFIG.vault_path.strip():
        # Explicit SILICA_VAULT wins — but still codebase-aware, like /vault:
        # a repo root becomes <repo>/.silica (created on demand); a .silica dir
        # or plain (non-repo) directory is adopted verbatim; a not-yet-existing
        # path stays as-is for the backend to create.
        t = resolve_vault_switch(CONFIG.vault_path)
        if t.vault:
            if t.created:
                Path(t.vault).mkdir(parents=True, exist_ok=True)
            CONFIG.vault_path = t.vault
        return
    cwd = Path.cwd()
    self_repo = gitstate.find_repo_root(Path(silica.__file__).resolve())
    existing = resolve_repo_mode_vault(cwd, "", docs_exists_ok=False, self_repo=self_repo)
    if existing:  # user repo already carries a .silica/
        CONFIG.vault_path = existing
        CONSOLE.print(f"  Repo mode: vault = [bold]{existing}[/]")
        return
    root = gitstate.find_repo_root(cwd)
    if root is not None and root != self_repo:  # user repo, .silica/ missing → ask
        CONSOLE.print(f"  Git repo detected at [bold]{root}[/] but no [bold].silica/[/] folder.")
        answer = input("  Create .silica/ and manage it as the Silica vault? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            vault_dir = Path(root) / ".silica"
            vault_dir.mkdir(parents=True, exist_ok=True)
            CONFIG.vault_path = str(vault_dir)
            CONSOLE.print(f"  Repo mode: vault = [bold]{vault_dir}[/]")
            return
    # No user repo, Silica's own repo, or declined → stable home vault.
    home_vault = default_user_vault()
    home_vault.mkdir(parents=True, exist_ok=True)
    CONFIG.vault_path = str(home_vault)
    CONSOLE.print(f"  Vault: [bold]{home_vault}[/]")


def _handle_direct_shortcut(raw_input: str, messages: list[dict]) -> bool:
    """Execute read-only commands directly without an LLM round-trip.

    Operates on the raw (case-preserved) input so that query strings and file
    paths reach the tool with their original casing intact.  Returns True if
    the command was handled, False to fall through to the normal dispatch.

    Handled commands (immediate, synchronous):
        /status [run_id]
        /embed [folder] [--force]
        /cooccur [folder] [--force]
        /graph [output.html] [folder]
        /find <query> [--k=N]
        /undo [note-path]
    """
    from silica.tools import TOOLS

    parts = raw_input.strip().split()
    if not parts:
        return False
    cmd = parts[0].lower()

    if cmd == "/vault":
        from pathlib import Path
        from silica.driver import reset_driver

        arg = " ".join(parts[1:]).strip()
        if arg:
            target = resolve_vault_switch(arg)
            if target.error:
                CONSOLE.print(f"  [red]{target.error}[/]")
                return True
            if target.created:
                Path(target.vault).mkdir(parents=True, exist_ok=True)
                CONSOLE.print(
                    f"  Codebase detected — created [bold]{target.vault}[/] "
                    "as the session vault (memory node)."
                )
            resolved = target.vault
            CONFIG.vault_path = resolved
            reset_driver()
            from silica.kernel.overlay import reset_overlay_cache
            reset_overlay_cache()  # overlay is vault-scoped; don't serve the old vault's
            from silica.kernel.vault_manifest import apply_manifest_to_config, reset_manifest_cache
            reset_manifest_cache()  # manifest is vault-scoped too
            apply_manifest_to_config()
            CONSOLE.print(f"  Vault → [bold]{resolved}[/] (backend: {CONFIG.backend})")
            CONSOLE.print(
                "  [dim]Index namespace follows the vault — run /embed and /cooccur "
                "if this vault has not been indexed yet.[/]"
            )
            return True
        vault = CONFIG.vault_path or "(not configured)"
        CONSOLE.print(f"  Vault:   [bold]{vault}[/]")
        CONSOLE.print(f"  Backend: {CONFIG.backend}")
        if CONFIG.vault_path:
            count = len(list(Path(CONFIG.vault_path).rglob("*.md")))
            CONSOLE.print(f"  Notes:   {count}")
            from silica.onboarding.checks import detect_vault_language, frozen_store_language

            detected = detect_vault_language(CONFIG.vault_path)
            if detected:
                store_lang = frozen_store_language(CONFIG.vault_path)
                if store_lang and store_lang != detected:
                    CONSOLE.print(
                        f"  Language: {detected} (store frozen: {store_lang} "
                        "⚠ — run /cooccur --force to rebuild)"
                    )
                elif store_lang:
                    CONSOLE.print(f"  Language: {detected} (store: {store_lang})")
                else:
                    CONSOLE.print(f"  Language: {detected}")
        return True

    if cmd == "/status":
        run_id = parts[1] if len(parts) > 1 else ""
        result = TOOLS["silica_ledger_digest"].run(run_id=run_id)
        try:
            parsed = json.loads(result)
            digest = parsed.get("digest", result)
            CONSOLE.print(FlatMarkdown(str(digest)))
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
            if "error" in parsed:
                CONSOLE.print(f"  [red]Error:[/] {parsed['error']}")
            else:
                CONSOLE.print(f"  Indexed: [bold]{parsed.get('indexed', '?')}[/] / {parsed.get('total_notes', '?')} notes")
            if parsed.get("read_errors"):
                CONSOLE.print(f"  [yellow]Read errors:[/] {parsed['read_errors']}")
        except Exception:
            CONSOLE.print(result)
        return True

    if cmd == "/cooccur":
        folder = ""
        force = False
        for part in parts[1:]:
            if part == "--force":
                force = True
            elif part.startswith("--folder="):
                folder = part[len("--folder="):]
            elif not part.startswith("-"):
                folder = part
        result = TOOLS["silica_cooccurrence_refresh"].run(folder=folder, force=force)
        try:
            parsed = json.loads(result)
            if "error" in parsed:
                CONSOLE.print(f"  [red]Error:[/] {parsed['error']}")
            else:
                CONSOLE.print(f"  Indexed: [bold]{parsed.get('indexed', '?')}[/] / {parsed.get('total_notes', '?')} notes (co-occurrence)")
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

    if cmd == "/stale":
        from pathlib import Path
        from silica.kernel import codedocs
        vault = CONFIG.vault_path
        if not vault:
            CONSOLE.print("  No vault configured; /stale needs a .silica vault in a git repo.")
            return True
        stale = codedocs.stale_docs(Path(vault))
        if not stale:
            CONSOLE.print("  No stale docs — every documents: note matches its code_ref.")
            return True
        CONSOLE.print(f"  [bold]Stale docs — {len(stale)} note/path pair(s):[/]")
        for sd in stale:
            n = len(sd.intervening)
            CONSOLE.print(
                f"  · [bold]{sd.note_path}[/] documents [bold]{sd.code_path}[/] "
                f"— {n} new commit(s) since {sd.recorded_ref[:8]}"
            )
            for c in sd.intervening[:3]:
                CONSOLE.print(f"      {c.sha[:8]}  {c.subject}")
        CONSOLE.print("  Run [bold]/ingest <path>[/] to regenerate, or edit and re-badge.")
        return True

    if cmd == "/plans":
        from pathlib import Path

        from rich.markup import escape

        from silica.kernel import plans as plans_mod
        if not CONFIG.vault_path:
            CONSOLE.print("  No vault configured; /plans needs a .silica vault.")
            return True
        vault = Path(CONFIG.vault_path)
        counts = plans_mod.status_counts(vault)
        if not counts:
            CONSOLE.print("  No plans found under plans/.")
            return True
        summary = ", ".join(f"[bold]{n}[/] {s}" for s, n in sorted(counts.items()))
        CONSOLE.print(f"  Plans: {summary}")
        for note_path, data in plans_mod.iter_plan_notes(vault):
            status = str(data.get("status") or "?").strip()
            # escape() keeps the literal [status] bracket from being parsed as
            # rich markup (otherwise [todo] is swallowed as an unknown tag).
            CONSOLE.print(f"    {escape(f'[{status}] {note_path.stem}')}")
        return True

    if cmd == "/undo":
        from silica.driver import DRIVER
        from silica.kernel.checkpoints import get_checkpoint_store

        store = get_checkpoint_store()
        note_path = parts[1] if len(parts) > 1 else store.most_recent_path()
        if not note_path:
            CONSOLE.print("  Nothing to undo — no patches recorded in this session.")
            return True

        content = store.undo(note_path)
        if content is None:
            CONSOLE.print(f"  [yellow]Nothing to undo for[/] {note_path} (already at original).")
            return True

        try:
            DRIVER.overwrite(note_path, content)
            depth = store.depth(note_path)
            remaining = max(0, depth - 1)
            CONSOLE.print(f"  Undone: [bold]{note_path}[/]  [dim]({remaining} undo step(s) remaining)[/]")
        except Exception as exc:
            CONSOLE.print(f"  [red]Undo failed:[/] {exc}")
        return True

    if cmd == "/revert":
        from silica.kernel.undo_journal import get_undo_journal, revert_run
        parts_split = raw_input.strip().split(maxsplit=1)
        run_id = parts_split[1].strip() if len(parts_split) > 1 else get_undo_journal().last_active_run()
        if not run_id:
            CONSOLE.print("  Nothing to undo — no runs recorded in this session.")
            return True
        res = revert_run(run_id)
        CONSOLE.print(
            f"  Revert {run_id[:8]}…: {len(res['reverted'])} reverted, "
            f"{len(res['skipped'])} skipped (modified), {len(res['errors'])} errors."
        )
        return True

    if cmd == "/review":
        from silica.kernel.deferred import get_deferred_store
        store = get_deferred_store()
        flush_hash = next((p[len("--flush="):] for p in parts[1:] if p.startswith("--flush=")), None)
        if flush_hash:
            removed = store.remove(flush_hash)
            if removed:
                CONSOLE.print(f"  Flushed bundle [bold]{flush_hash[:12]}[/] from review queue.")
            else:
                CONSOLE.print(f"  [yellow]No bundle with hash {flush_hash[:12]} found.[/]")
            return True
        items = store.list_all()
        if not items:
            CONSOLE.print("  Review queue is empty.")
        else:
            CONSOLE.print(f"  [bold]Review queue — {len(items)} bundle(s):[/]")
            for item in items:
                import datetime as _dt
                ts = _dt.datetime.fromtimestamp(item["timestamp"]).strftime("%Y-%m-%d %H:%M")
                CONSOLE.print(
                    f"  · [bold]{item['content_hash'][:12]}[/]  {item['source_path']}  "
                    f"({item['rejected_count']} op(s))  {ts}"
                )
            CONSOLE.print("  Use [bold]/review --flush=<hash>[/] to discard a bundle.")
        return True

    if cmd == "/dedup":
        positional = [p for p in parts[1:] if not p.startswith("-")]
        folder = " ".join(positional)
        tool_name = "silica_dedup"
        label = "Dedup"
        CONSOLE.print(f"  {label} on [bold]{folder or '(vault)'}[/] — sub-agents on the worker model…")
        result = TOOLS[tool_name].run(folder=folder)
        try:
            parsed = json.loads(result)
            if "error" in parsed:
                CONSOLE.print(f"  [yellow]{parsed['error']}[/]")
            else:
                scope = parsed.get("pairs_found", parsed.get("notes", parsed.get("items", 0)))
                CONSOLE.print(f"  Processed [bold]{scope}[/] pair(s) — outcomes: {parsed.get('summary', {})}")
        except Exception:
            CONSOLE.print(result)
        return True

    if cmd == "/curate":
        apply = any(p == "--apply" for p in parts[1:])
        positional = [p for p in parts[1:] if not p.startswith("-")]
        folder = " ".join(positional)
        scope = folder or "(vault)"
        if apply:
            CONSOLE.print(f"  Curate on [bold]{scope}[/] — applying via the worker seam…")
        else:
            CONSOLE.print(f"  Curate on [bold]{scope}[/] — dry-run (nothing is written)…")
        res = TOOLS["silica_curate"].run(apply=apply, folder=folder)

        total = res.get("total", 0)
        counts = res.get("counts", {})
        if total == 0:
            CONSOLE.print("  Nothing to do — the vault is coherent.")
            return True

        breakdown = ", ".join(f"{v} {k}" for k, v in counts.items())
        if apply:
            # Real outcomes (execution["outcome_counts"], derived from the
            # dispatch batch's per-item status + the mechanical autolink's
            # actual links-added count) — NOT the planned counts above, which
            # would report "Applied N" even when e.g. every dedup came back
            # a distinct verdict and nothing was actually merged.
            outcome = res.get("execution", {}).get("outcome_counts", {})
            dispatched = sum(outcome.values())
            outcome_breakdown = ", ".join(f"{v} {k}" for k, v in outcome.items()) or "no changes"
            CONSOLE.print(f"  Applied — dispatched [bold]{dispatched}[/] → outcomes: {outcome_breakdown}")
        else:
            CONSOLE.print(f"  Plan — [bold]{total}[/] item(s): {breakdown}")
            for it in res.get("items", []):
                pair = f" ↔ {it['partner']}" if it.get("partner") else ""
                CONSOLE.print(f"  · [bold]{it['kind']}[/]  {it['target']}{pair}")
            CONSOLE.print("  Run [bold]/curate --apply[/] to execute.")
        return True

    return False


def _expand_workflow_shortcut(user_input: str) -> str | None:
    """Expand workflow shortcuts (e.g. /report, /ingest) into agent-directed messages.

    Returns the expanded message string, or None if the input is not a
    recognised shortcut. Expanded messages flow through the normal agentic
    loop so the agent calls the tools and follows the steering protocol.

    Syntax:
        /report [folder] [--top-k=N] [--embeddings]
        /ingest <file...> [--target=DIR] [--hub=H]
        /convert <file...> [--target=DIR]

    Examples:
        /report
        /report Concepts/ML
        /report --top-k=15 --embeddings
        /report Inbox --embeddings
        /ingest Inbox/notes.md --target=Concepts/AI
        /ingest silica/cli.py
        /ingest paper.pdf --target=Concepts/AI
        /ingest "Inbox/papers/With Spaces.pdf" --target=Concepts/AI
        /convert paper.pdf
    """
    try:
        parts = shlex.split(user_input.strip())  # honours quoted paths with spaces
    except ValueError:
        return "Error: unbalanced quotes in command. Wrap paths with spaces in \"...\"."
    if not parts:
        return None

    cmd = parts[0].lower()

    if cmd == "/ingest":
        args = parts[1:]
        files: list[str] = []
        target_dir = ""
        hub = ""
        for arg in args:
            if arg.startswith("--target="):
                target_dir = arg[len("--target="):]
            elif arg.startswith("--hub="):
                hub = arg[len("--hub="):]
            elif not arg.startswith("-"):
                files.append(arg)  # preserve original case
        if not files:
            return "Error: /ingest requires at least one file path. Usage: /ingest <file...> [--target=DIR]"

        from silica.kernel.vault_manifest import get_active_manifest
        from silica.sources.convert import convert
        from silica.sources.registry import adapter_for, stage

        enabled = get_active_manifest().sources
        md_files: list[str] = []
        for f in files:
            adapter = adapter_for(f, enabled=enabled)
            if adapter is None:
                # No source claims this file type → try the converter fallback
                # (PDF today). The CONVERTED .md is what the FSM re-reads.
                try:
                    md_files.append(convert(f, dest_dir=target_dir))
                except ValueError as e:
                    CONSOLE.print(f"  [yellow]Skipped {f}: {e}[/]")
                continue
            result = stage(adapter, f)
            if result["status"] == "distill":
                md_files.append(f)
            elif result["status"] == "ok":
                code_ref = result["meta"].get("code_ref", "")
                CONSOLE.print(
                    f"  Staged [bold]{result['note_path']}[/] "
                    f"(code_ref {code_ref[:8]}). Refine via the inbox."
                )
            else:
                CONSOLE.print(f"  [yellow]{f}: {result.get('message', '')}[/]")

        if not md_files:
            return ""  # fully handled inline — sentinel: nothing for the agent

        if not target_dir:
            return "Error: /ingest of notes requires --target=DIR. Usage: /ingest <file...> --target=DIR"

        from pathlib import Path as _Path
        from silica.kernel.provenance import check_reingest, content_sha256

        for mf in md_files:
            try:
                incoming_sha = content_sha256(mf)
                if not incoming_sha:
                    continue
                modified, prior_notes = check_reingest(_Path(mf).name, incoming_sha)
                if modified:
                    CONSOLE.print(
                        f"  [yellow]re-ingest of a modified source: {prior_notes} note(s) "
                        f"derived from the previous version[/]"
                    )
            except Exception as exc:
                logger.debug("/ingest: re-ingest provenance check skipped for %s (non-fatal): %s", mf, exc)

        files_json = json.dumps(md_files)
        msg = (
            f"Run the Injector pipeline for {len(md_files)} file(s).\n"
            f"Call `silica_run_injector` with "
            f"inbox_files={files_json}, target_dir={json.dumps(target_dir)}"
        )
        if hub:
            msg += f", hub={json.dumps(hub)}"
        msg += "."
        return msg

    if cmd == "/convert":
        args = parts[1:]
        files = [a for a in args if not a.startswith("-")]
        target_dir = next((a[len("--target="):] for a in args if a.startswith("--target=")), "")
        if not files:
            return "Error: /convert requires at least one file path. Usage: /convert <file...> [--target=DIR]"
        from silica.sources.convert import convert
        for f in files:
            try:
                CONSOLE.print(f"  Converted {f} → [bold]{convert(f, dest_dir=target_dir)}[/]")
            except ValueError as e:
                CONSOLE.print(f"  [yellow]Skipped {f}: {e}[/]")
        return ""  # fully handled inline — sentinel: nothing for the agent

    if cmd == "/web-search":
        from silica.sources.web_research import web_research, _DEFAULT_MAX_SEARCHES
        from silica.ui.renderer import make_progress_callback
        args = parts[1:]
        max_searches = _DEFAULT_MAX_SEARCHES
        positional = []
        for arg in args:
            if arg.startswith("--max-searches="):
                try:
                    max_searches = int(arg[len("--max-searches="):])
                except ValueError:
                    pass
            elif not arg.startswith("-"):
                positional.append(arg)
        concept = " ".join(positional).strip()
        if not concept:
            return 'Error: /web-search requires a concept. Usage: /web-search "<concept>" [--max-searches=N]'

        try:
            note_rel = web_research(
                concept, max_searches=max_searches,
                tool_progress_callback=make_progress_callback(),
            )
            CONSOLE.print(f"  Findings → [bold]{note_rel}[/]  (review, then /ingest to bring it in)")
        except Exception as e:  # missing key, no findings, convergence guard, network
            CONSOLE.print(f"  [yellow]web-search failed: {e}[/]")
        return ""  # fully handled inline — sentinel: nothing for the agent

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
            f"with_embeddings={'true' if with_embeddings else 'false'}, seed_ledger=true.\n"
            f"Then STOP. Write a short, human-readable brief in chat from the returned `digest` "
            f"(totals, top hubs, and how many fixes are available: auto / propose / issues), and "
            f"point the user to the GRAPH_REPORT.md that was written.\n"
            f"Do NOT run the steering loop, do NOT call `silica_ledger_next`, and do NOT apply any "
            f"autolinks, corrections, renames, or deletions. Instead, ask the user whether they want "
            f"to apply the changes. Only if they explicitly say yes, resume the run (`run_id`) and "
            f"follow the steering loop."
        )

    if cmd in ("/refine", "/enrich"):
        args = parts[1:]
        folder = next((p for p in args if not p.startswith("-")), "")

        from silica.driver import DRIVER
        from silica.kernel.progress import PlanStep, Run
        from pathlib import Path
        import orjson

        refs = DRIVER.list_files(folder=folder)
        paths = [r.path for r in refs if r.path.startswith(folder) or r.path == folder]
        if not paths:
            return f"Error: no files found in '{folder}'."

        chunks = []
        cur_chunk = []
        cur_size = 0
        for p in paths:
            s = len(json.dumps(p))
            if cur_size + s > 4000 and cur_chunk:
                chunks.append(cur_chunk)
                cur_chunk = [p]
                cur_size = s
            else:
                cur_chunk.append(p)
                cur_size += s
        if cur_chunk:
            chunks.append(cur_chunk)

        cap = "silica_refine_batch" if cmd == "/refine" else "silica_enrich_batch"

        run = Run.new(
            mode="analyst",
            user_request=f"{cmd.strip('/')} {folder or 'vault'}",
            checkpoints=[PlanStep(id="remediate", kind="gate", objective=cap)],
            inputs={"scope": folder or "vault"},
        )
        payloads_dir = run.payloads_dir

        for i, chunk in enumerate(chunks):
            task = run.progress.add_task(cap)
            payload = {"note_paths": chunk, "_reason": f"Batch {i+1} of {len(chunks)}"}
            payload_path = str(payloads_dir / f"{task.id}.json")
            Path(payload_path).write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
            task.input_ref = payload_path

        run.save()

        from silica.ui.renderer import emit_batch_event
        from silica.agent.events import BatchRunStartEvent
        emit_batch_event(BatchRunStartEvent(run_id=run.run_id, kind=cmd.strip("/"), label=folder or "vault", total=len(chunks)))

        return f"A ledger for {cmd} has been created with {len(chunks)} chunk(s) across {len(paths)} note(s). Use `silica_ledger_next` to execute them."

    if cmd == "/organize":
        args = parts[1:]
        intent_parts: list[str] = []
        scope = ""
        taxonomy_file = ""
        apply_now = False
        merge = False
        move_uncat = False

        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--scope="):
                scope = arg[len("--scope="):]
            elif arg.startswith("--file="):
                taxonomy_file = arg[len("--file="):]
            elif arg in ("--apply",):
                apply_now = True
            elif arg in ("--merge",):
                merge = True
            elif arg in ("--move-uncategorized",):
                move_uncat = True
            elif not arg.startswith("-"):
                intent_parts.append(arg)
            i += 1

        # Re-join intent (handles both quoted and unquoted multi-word)
        intent = " ".join(intent_parts).strip('"\'')
        run_extra = ", move_uncategorized=true" if move_uncat else ""

        if taxonomy_file:
            # Skip taxonomy generation — use existing file
            dry = "false" if apply_now else "true"
            scope_str = f", scope={json.dumps(scope)}" if scope else ""
            msg = (
                f"Run the vault organizer using the existing taxonomy file {json.dumps(taxonomy_file)}.\n"
                f"Call `silica_run_organizer` with taxonomy_path={json.dumps(taxonomy_file)}{scope_str}, "
                f"dry_run={dry}{run_extra}.\n"
            )
            if not apply_now:
                msg += (
                    "Show the move plan to the user and ask for confirmation. "
                    "If confirmed, call `silica_run_organizer` again with dry_run=false."
                )
        elif intent:
            scope_str = f", scope={json.dumps(scope)}" if scope else ""
            merge_str = ", merge=true" if merge else ""
            dry_note = (
                f"Then call `silica_run_organizer` with dry_run=true{run_extra} to preview the moves. "
                "Show the plan to the user and ask for confirmation before executing."
            ) if not apply_now else (
                f"Then call `silica_run_organizer` with dry_run=false{run_extra} to execute the moves."
            )
            msg = (
                f"Organize the vault based on the user's intent: {json.dumps(intent)}.\n"
                f"Step 1: Call `silica_generate_taxonomy` with user_intent={json.dumps(intent)}{scope_str}{merge_str}.\n"
                f"Step 2: Show the generated taxonomy to the user and ask if it looks correct.\n"
                f"Step 3: {dry_note}"
            )
        else:
            msg = (
                "Help me organize my vault. "
                "Ask me to describe how I want to group my notes, "
                "then call `silica_generate_taxonomy` with my answer, "
                "show me the taxonomy, and run `silica_run_organizer` with dry_run=true to preview."
            )
        return msg

    return None


def _handle_slash_command(cmd: str, messages: list[dict]) -> bool:
    """Handle slash commands. Returns True if the command was handled."""
    cmd = cmd.strip().lower()

    if cmd in ("/exit", "/quit", "/q"):
        return False  # Signal to exit

    if cmd == "/model":
        if not CONFIG.model:
            CONSOLE.print("  Current model: [bold](not configured)[/]")
            return True
        from silica.agent.providers import model_limits
        window, out_cap = model_limits(CONFIG.provider, CONFIG.model)
        extra = ""
        if window:
            extra = f"  [dim]ctx {window:,}[/]"
            if out_cap:
                extra += f" [dim]· max out {out_cap:,}[/]"
        CONSOLE.print(f"  Current model: [bold]{CONFIG.model}[/]{extra}")
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
        from silica.ui.commands import render_help
        render_help()
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


_NO_MODEL_HINT = (
    "  [yellow]No chat model configured.[/] Run [bold]silica init[/] to set one — "
    "direct commands (/find, /status, /cooccur, …) still work."
)


def _model_configured() -> bool:
    return bool(CONFIG.model.strip())


def _resolve_context_budget() -> None:
    """Size the REPL context meter to the live model's window.

    SILICA_MAX_CONTEXT, when set, is an explicit operator pin and wins;
    otherwise ask the provider (LM Studio reports the loaded window, OpenRouter
    the model's context_length) and fall back to the static default when
    unreachable.
    """
    if os.getenv("SILICA_MAX_CONTEXT") or not _model_configured():
        return
    from silica.agent.providers import model_limits
    window, _ = model_limits(CONFIG.provider, CONFIG.model)
    if window:
        CONFIG.max_context_tokens = window


def _dispatch_subcommand(args: list[str]) -> int | None:
    """Handle `silica doctor` / `silica init`.

    Returns an exit code, or None when no subcommand matched (→ REPL).
    Lazy imports keep REPL startup unchanged. Module attributes (not `from`
    imports) so tests can monkeypatch run_checks / run_wizard.
    """
    if args[:1] == ["doctor"]:
        import silica.onboarding.checks as checks
        results = checks.run_checks(CONFIG)
        checks.render_report(results)
        return 1 if checks.has_failures(results) else 0
    if args[:1] == ["init"]:
        import silica.onboarding.wizard as wizard_mod
        return wizard_mod.run_wizard()
    return None


def main():
    """Entry point for the `silica` CLI command."""
    _args = [a for a in sys.argv[1:] if a not in ("--verbose", "-v")]
    code = _dispatch_subcommand(_args)
    if code is not None:
        sys.exit(code)
    _activate_repo_mode()
    from silica.kernel.vault_manifest import apply_manifest_to_config
    apply_manifest_to_config()
    _resolve_context_budget()
    debug_mode = "--verbose" in sys.argv or "-v" in sys.argv or CONFIG.debug_logging
    _setup_logging(debug=debug_mode)

    print_home()
    if not _model_configured():
        CONSOLE.print(_NO_MODEL_HINT)

    session = build_session()
    messages = _fresh_messages()

    from silica.ui.renderer import make_progress_callback
    callback = make_progress_callback()

    while True:
        try:
            user_input = session.prompt(prompt_text(), bottom_toolbar=bottom_toolbar)
        except (EOFError, KeyboardInterrupt):
            print("\n  (_  _)。˚")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Direct shortcuts bypass the LLM entirely (case-sensitive args preserved)
        if user_input.startswith("/") and _handle_direct_shortcut(user_input, messages):
            continue

        # Expand workflow shortcuts (/report, /ingest etc.) into agent-directed messages
        is_directive = False
        expanded = _expand_workflow_shortcut(user_input)
        if expanded is not None:
            if not expanded:
                continue  # shortcut fully handled inline (e.g. /ingest of code files)
            user_input = expanded
            is_directive = True

        # Handle slash commands
        if user_input.startswith("/"):
            cmd = user_input.strip().lower()
            if cmd == "/clear":
                CONSOLE.clear()
                print_home()
                messages[:] = _fresh_messages()
                continue

            should_continue = _handle_slash_command(user_input, messages)
            if not should_continue:
                print("  (_  _)。˚")
                break
            continue

        # Fail-fast guard: a chat turn without a model would only surface a
        # provider stack trace — point at `silica init` instead.
        if not _model_configured():
            CONSOLE.print(_NO_MODEL_HINT)
            continue

        # Normal user message → agentic loop. CLI-expanded shortcuts carry an
        # `origin` so the wire boundary (and our own bookkeeping) can tell a
        # harness directive apart from a human turn.
        msg: dict = {"role": "user", "content": user_input}
        if is_directive:
            msg["origin"] = "cli"
        messages.append(msg)

        try:
            answer = run_agent(messages, model=CONFIG.model, tool_progress_callback=callback)
            if answer:
                CONSOLE.print()
                CONSOLE.print("[role.assistant]⏺ silica[/]")
                CONSOLE.print(FlatMarkdown(answer))
                CONSOLE.print()
            messages.append({"role": "assistant", "content": answer or ""})
            _update_context_tokens(messages)
        except KeyboardInterrupt:
            callback.close()
            CONSOLE.print("\n  [dim](interrupted)[/]")
        except Exception as e:
            callback.close()
            logger.exception("Agent error")
            CONSOLE.print(f"\n  [bold red]Error:[/] {e}\n")


if __name__ == "__main__":
    main()
