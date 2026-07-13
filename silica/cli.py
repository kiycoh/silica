# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

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
from silica.prompts import system_prompt
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


def _count_context_tokens(messages: list[dict]) -> int:
    """Pure counter — lets callers (e.g. the web seed prewarm) count a candidate
    message list without clobbering the live session's CONFIG.context_tokens."""
    try:
        import litellm
        return litellm.token_counter(model=CONFIG.model, messages=messages)
    except Exception:
        return sum(len(m.get("content") or "") for m in messages) // 4


def _update_context_tokens(messages: list[dict]) -> None:
    CONFIG.context_tokens = _count_context_tokens(messages)


# ponytail: fixed knobs, promote to Config only if someone actually needs to tune them
_COMPACT_FRACTION = 0.6   # collapse old reads once history crosses 60% of the window
_COMPACT_FLOOR_TURNS = 3  # the last N assistant turns are never collapsed


def _compact_context(messages: list[dict], collapsed: set[int]) -> set[int]:
    """Collapse old read-tool results once the context meter crosses the budget.

    Runs after _update_context_tokens (which feeds prompt_tokens); when
    anything collapsed, recounts so the toolbar meter reflects the slimmer
    history. Loss is recoverable: each stub names the call to re-issue.
    """
    from silica.agent.compaction import compact_read_history
    from silica.tools import TOOLS

    updated = compact_read_history(
        messages,
        collapsed,
        prompt_tokens=CONFIG.context_tokens,
        budget=int(_COMPACT_FRACTION * CONFIG.max_context_tokens),
        floor_turns=_COMPACT_FLOOR_TURNS,
        tools=TOOLS,
    )
    if updated != collapsed:
        _update_context_tokens(messages)
    return updated


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
    from silica.kernel.vault_manifest import get_active_manifest

    conv = get_active_manifest().conventions
    reply = conv.reply_language or conv.language
    messages: list[dict] = [{"role": "system", "content": system_prompt(reply)}]
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
        from silica.ui.logging import (
            AnsiHumanFriendlyFormatter,
            HumanFriendlyFormatter,
            LiveAwareStreamHandler,
        )
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
        # Same human-friendly seam as the main thread — rendered to ANSI in the
        # formatter (throwaway Console) so worker logs (dedup, refine, enrich,
        # expand, orphan…) read like the main-thread ones instead of raw dumps.
        bg_handler.setFormatter(AnsiHumanFriendlyFormatter())
        bg_handler.addFilter(lambda r: threading.current_thread() is not main_thread)
        root.addHandler(bg_handler)
    else:
        from silica.ui.logging import AnsiHumanFriendlyFormatter, LiveAwareStreamHandler
        # Live-aware: follows rich.Live's stderr redirect so warnings during the
        # injector/batch live region print above it instead of tearing the panel.
        # Same human-friendly ANSI seam as debug mode's worker handler, so
        # warnings/errors (incl. worker threads like dedup) render coloured instead
        # of raw dumps. Level stays WARNING here — only warn/error surface.
        handler = LiveAwareStreamHandler()
        handler.setFormatter(AnsiHumanFriendlyFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # LiteLLM/httpx/openai/httpcore are always silenced — their DEBUG is raw HTTP/request dumps
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)
    # asyncio DEBUG is one "Using selector" line per event loop — litellm's sync
    # streaming path creates a fresh loop PER CHUNK, so --verbose drowns in them.
    logging.getLogger("asyncio").setLevel(logging.WARNING)


class VaultTarget(NamedTuple):
    """Outcome of resolving a runtime ``/vault <arg>`` switch.

    ``vault`` is the absolute path to adopt. ``created`` is True when repo mode
    has no ``docs/silica`` yet and the caller must mkdir it. Every path now
    resolves to a valid target — an Obsidian vault verbatim, everything else
    repo mode — so there is no error outcome.
    """
    vault: str
    created: bool


def resolve_vault_switch(arg: str) -> VaultTarget:
    """Resolve a ``/vault <arg>`` (or explicit ``SILICA_VAULT``) target.

    ``.obsidian/`` is the sole layout signal, not git: an Obsidian vault is
    adopted verbatim (notes in its root); anything else — a code repo, a plain
    or not-yet-existing directory — is Silica repo mode, notes under
    ``<arg>/docs/silica`` (created on demand). Pure, read-only I/O.
    """
    from pathlib import Path
    from silica.kernel.paths import is_obsidian_vault, repo_mode_vault

    target = Path(arg).expanduser().resolve()
    if is_obsidian_vault(target):
        return VaultTarget(str(target), False)
    vault = repo_mode_vault(target)
    return VaultTarget(str(vault), not vault.is_dir())


def default_user_vault(home=None):
    """Stable per-user vault used when no explicit SILICA_VAULT and no repo
    mode applies. Sits alongside ~/.silica/{ledger,undo_journal,checkpoints}.db.
    """
    from pathlib import Path

    return (home or Path.home()) / ".silica" / "vault"


def resolve_repo_mode_vault(cwd, vault_env: str, docs_exists_ok: bool, self_repo=None):
    """Pure resolver for repo-mode vault selection (testable, no I/O prompts).

    Returns the vault path string to adopt, or None to leave config unchanged.
    Git still *discovers* the project root from cwd; ``.obsidian`` then decides
    the layout on that root.
    - Explicit SILICA_VAULT (vault_env truthy) always wins → None.
    - Not inside a git repo → None.
    - Inside Silica's own source repo (root == self_repo) → None: that's dev
      mode, not a vault. Caller falls back to the home default.
    - root is an Obsidian vault (.obsidian/) → adopt it verbatim.
    - else docs/silica exists → return it; missing → only if docs_exists_ok
      (caller confirmed creation); otherwise None.
    """
    from pathlib import Path
    from silica.kernel import gitstate
    from silica.kernel.paths import is_obsidian_vault, repo_mode_vault

    if vault_env.strip():
        return None
    root = gitstate.find_repo_root(cwd)
    if root is None:
        return None
    if self_repo is not None and root == Path(self_repo):
        return None
    if is_obsidian_vault(root):
        return str(Path(root).resolve())
    vault_dir = repo_mode_vault(root)
    if vault_dir.is_dir():
        return str(vault_dir)
    if docs_exists_ok:
        return str(vault_dir)
    return None


def _activate_repo_mode() -> None:
    """Side-effecting startup vault selection. Explicit SILICA_VAULT wins; else
    a *user* project repo → its docs/silica (prompted if absent), unless the
    repo is an Obsidian vault (adopted verbatim); else — including inside
    Silica's own source repo (dev mode) — a stable ~/.silica/vault."""
    from pathlib import Path
    from silica.kernel import gitstate
    import silica

    if CONFIG.vault_path.strip():
        # Explicit SILICA_VAULT wins, resolved like /vault: an Obsidian vault
        # (.obsidian/) is adopted verbatim; anything else → <path>/docs/silica,
        # created on demand. Git is never consulted for a named path.
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
    if root is not None and root != self_repo:  # user repo, no docs/silica → ask
        from silica.kernel.paths import repo_mode_vault

        vault_dir = repo_mode_vault(root)
        CONSOLE.print(f"  Git repo detected at [bold]{root}[/] but no [bold]docs/silica/[/] folder.")
        answer = input("  Create docs/silica/ and manage it as the Silica vault? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            vault_dir.mkdir(parents=True, exist_ok=True)
            CONFIG.vault_path = str(vault_dir)
            CONSOLE.print(f"  Repo mode: vault = [bold]{vault_dir}[/]")
            return
    # No user repo, Silica's own repo, or declined → stable home vault.
    home_vault = default_user_vault()
    home_vault.mkdir(parents=True, exist_ok=True)
    CONFIG.vault_path = str(home_vault)
    CONSOLE.print(f"  Vault: [bold]{home_vault}[/]")


def _announce_code_lane() -> None:
    """Eager repo-root resolution (ADR-0019): validate the vault⊂repo invariant
    once at startup / vault switch and surface a violation loudly."""
    from silica.kernel.paths import repo_root_warning

    warn = repo_root_warning(CONFIG.vault_path)
    if warn:
        CONSOLE.print(f"  [yellow]⚠ {warn}[/]")


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
        /map <nota> [--force]
        /find <query> [--k=N]
        /impact [<git-range>]
        /path <noteA> <noteB>
        /contested
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
            _announce_code_lane()
            # Surface the frozen-language drift here, not only in `/vault` info:
            # a switch is exactly when a wrong-frozen store (english on an IT
            # vault) would otherwise stay silent. Reuses the doctor's check.
            from silica.onboarding.checks import language_status

            lang, store_lang, drift = language_status(resolved)
            if drift:
                CONSOLE.print(
                    f"  [yellow]⚠ Language: {lang}, co-occurrence store "
                    f"frozen {store_lang} — run /cooccur --force to rebuild.[/]"
                )
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
            from silica.onboarding.checks import language_status

            lang, store_lang, drift = language_status(CONFIG.vault_path)
            if lang and drift:
                CONSOLE.print(
                    f"  Language: {lang} (store frozen: {store_lang} "
                    "⚠ — run /cooccur --force to rebuild)"
                )
            elif lang and store_lang:
                CONSOLE.print(f"  Language: {lang} (store: {store_lang})")
            elif lang:
                CONSOLE.print(f"  Language: {lang}")
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

    if cmd == "/wiki":
        args = parts[1:]
        folder = next((a for a in args if not a.startswith("-")), "") or None
        overview_only = "--overview-only" in args
        force = "--force" in args
        from silica.capabilities.codewiki import run_wiki
        result = run_wiki(CONFIG.vault_path, CONFIG, folder=folder,
                          overview_only=overview_only, force=force)
        if result["status"] == "no_repo":
            CONSOLE.print("  [yellow]wiki: vault is not inside a git repo, nothing to describe.[/]")
        elif result["status"] == "empty":
            CONSOLE.print("  [yellow]wiki: no supported source files found "
                          "(code lane parses Python/TypeScript/JavaScript only).[/]")
        elif result["status"] == "error":
            CONSOLE.print(f"  [yellow]wiki: {result.get('reason', 'error')}[/]")
        else:
            CONSOLE.print(
                f"  wiki: {len(result['written'])} note(s) written, "
                f"{len(result['skipped'])} up-to-date"
                + (f", {result['parse_errors']} file(s) not analyzable" if result["parse_errors"] else "")
            )
            for fail in result.get("failed", []):
                CONSOLE.print(f"  [red]wiki: write failed:[/] {fail['path']}: {fail['reason']}")
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

    if cmd == "/map":
        force = "--force" in parts[1:]
        positional = [p for p in parts[1:] if not p.startswith("-")]
        note = " ".join(positional).strip()
        if not note:
            CONSOLE.print("  Usage: /map <nota> [--force]")
            return True
        result = TOOLS["silica_mindmap"].run(note_path=note, force=force)
        try:
            parsed = json.loads(result)
            if parsed.get("skipped"):
                CONSOLE.print(
                    f"  [yellow]Mappa già presente[/] ({parsed['skipped']}) — "
                    "non sovrascritta. Usa [bold]/map <nota> --force[/] per rigenerare."
                )
            elif "error" in parsed:
                CONSOLE.print(f"  [red]{parsed['error']}[/]")
            else:
                CONSOLE.print(
                    f"  Mappa scritta: [bold]{parsed.get('path', '?')}[/] "
                    f"({parsed.get('nodes', '?')} nodi, {parsed.get('edges', '?')} archi)"
                )
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
        show_all = "--all" in parts[1:]
        stale = codedocs.stale_docs(Path(vault))
        by_note: dict[str, list] = {}
        for sd in stale:
            by_note.setdefault(sd.note_path, []).append(sd)
        shown = 0
        for note_path, docs in sorted(by_note.items()):
            level, details = codedocs.note_verdict(docs)
            if level != codedocs.CHANGE_STRUCTURAL and not show_all:
                continue
            shown += 1
            CONSOLE.print(f"  · [bold]{note_path}[/] — {level}")
            for sd in docs:
                n = len(sd.intervening)
                CONSOLE.print(
                    f"      documents [bold]{sd.code_path}[/] — {n} new commit(s) "
                    f"since {sd.recorded_ref[:8]}"
                )
            for d in details[:6]:
                CONSOLE.print(f"      {d}")
        if not shown:
            hidden = len(by_note)
            if hidden and not show_all:
                CONSOLE.print(
                    f"  No structural staleness. {hidden} note(s) have cosmetic-only "
                    "changes — use [bold]/stale --all[/] to list them."
                )
            else:
                CONSOLE.print("  No stale docs — every documents: note matches its code_ref.")
            return True
        CONSOLE.print("  Run [bold]/ingest <path>[/] to regenerate, or edit and re-badge.")
        return True

    if cmd == "/impact":
        from pathlib import Path
        from silica.kernel.codegraph import compute_impact
        vault = CONFIG.vault_path
        if not vault:
            CONSOLE.print("  No vault configured; /impact needs a vault inside a git repo.")
            return True
        range_spec = parts[1] if len(parts) > 1 else None
        entries = compute_impact(Path(vault), range_spec)
        if entries is None:
            CONSOLE.print("  No git repo — impact analysis unavailable.")
            return True
        if not entries:
            scope = range_spec or "working tree vs HEAD"
            CONSOLE.print(f"  No supported source files changed ({scope}).")
            return True
        for e in entries:
            CONSOLE.print(f"  · [bold]{e.path}[/] — {e.change_level} (fan-in {e.fan_in})")
            for d in e.details[:4]:
                CONSOLE.print(f"      {d}")
            if e.notes:
                CONSOLE.print(f"      documents: {', '.join(e.notes)}")
            if e.neighbor_notes:
                CONSOLE.print(f"      1-hop neighbors documented by: {', '.join(e.neighbor_notes)}")
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

    if cmd == "/path":
        from silica.kernel.mindmap import note_resolver, reading_path
        try:
            toks = shlex.split(raw_input.strip())[1:]  # honours quoted titles with spaces
        except ValueError:
            CONSOLE.print('  Unbalanced quotes. Usage: /path "<note A>" "<note B>"')
            return True
        endpoints = [t for t in toks if not t.startswith("-")]
        if len(endpoints) != 2:
            CONSOLE.print("  Usage: /path <noteA> <noteB>")
            return True
        resolve = note_resolver()
        src, dst = resolve(endpoints[0]), resolve(endpoints[1])
        for given, got in zip(endpoints, (src, dst)):
            if got is None:
                CONSOLE.print(f"  Note not found: '{given}'")
        if src is None or dst is None:
            return True
        if src == dst:
            CONSOLE.print("  Both resolve to the same note — nothing to walk.")
            return True
        path = reading_path(src, dst)
        if path is None:
            CONSOLE.print(
                f"  No path between [bold]{src}[/] and [bold]{dst}[/] — "
                "not connected (try /map on each to see its neighborhood)."
            )
            return True
        CONSOLE.print(f"  Reading path — {len(path) - 1} step(s):")
        for i, (node, leg) in enumerate(path):
            if leg != "start":
                CONSOLE.print(f"        [dim]↓ {leg}[/]")
            CONSOLE.print(f"    {i + 1}. [bold]{node}[/]")
        return True

    if cmd == "/contested":
        from silica.driver import DRIVER
        from silica.kernel.contested import CONTESTED_KEY, CONTRADICTIONS_KEY
        # ponytail: frontmatter scan of every note per call; index it if a
        # 10k-note vault ever makes this command feel slow.
        found: list[tuple[str, list[str]]] = []
        for ref in DRIVER.list_files(""):
            try:
                props = DRIVER.props_of(ref.path)
            except Exception:
                continue  # attachments / unreadable frontmatter — not contested
            if props and props.get(CONTESTED_KEY):
                contras = [str(c) for c in (props.get(CONTRADICTIONS_KEY) or [])]
                found.append((ref.path, contras))
        if not found:
            CONSOLE.print("  No contested notes — no unresolved contradictions.")
            return True
        CONSOLE.print(f"  {len(found)} contested note(s):")
        for note_path, contras in sorted(found):
            CONSOLE.print(f"  · [bold]{note_path}[/]")
            for c in contras:
                CONSOLE.print(f"      conflicts with: {c}")
        CONSOLE.print(
            "  Resolve by editing the note, then remove `contested: true` and its callout."
        )
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
        res = json.loads(TOOLS["silica_curate"].run(apply=apply, folder=folder))
        if "error" in res:
            CONSOLE.print(f"  [yellow]{res['error']}[/]")
            return True

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
            CONSOLE.print('  Run [bold]/curate --apply[/] to execute, or ask e.g. "apply only dedup".')
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
        /summarize <note|folder...>
        /explain "<concept>" [--level=intro|expert]
        /compare "<A>" "<B>" [...]
        /quiz <note|folder> [--n=10]
        /relate <note> [--n=8]

    Examples:
        /report
        /report Concepts/ML
        /report --top-k=15 --embeddings
        /report Inbox --embeddings
        /ingest Inbox/notes.md --target=Concepts/AI
        /ingest Inbox/notes.md
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
                    md_files.extend(convert(f, dest_dir=target_dir))
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
        if target_dir:
            msg = (
                f"Run the Injector pipeline for {len(md_files)} file(s).\n"
                f"Call `silica_run_injector` with "
                f"inbox_files={files_json}, target_dir={json.dumps(target_dir)}"
            )
        else:
            # ponytail: auto-target = one folder per run chosen by the chat
            # agent (it already holds the vault map); per-note thematic
            # placement stays /organize's job.
            msg = (
                f"Run the Injector pipeline for {len(md_files)} file(s).\n"
                f"No target folder was given. Skim the inbox file(s) {files_json}, "
                f"then pick the single most relevant existing vault folder for "
                f"this content (use the vault map; list folders if unsure). If "
                f"nothing fits, pick a sensible new folder name. State the chosen "
                f"folder in one line, then call `silica_run_injector` with "
                f"inbox_files={files_json}, target_dir=<chosen folder>"
            )
        if hub:
            msg += f", hub={json.dumps(hub)}"
        msg += "."
        return msg

    if cmd == "/settings":
        # View/edit vault.yaml without the wizard. ponytail: safe_dump rewrite —
        # YAML comments are not preserved; hand-edit the file to keep them.
        import yaml as _yaml
        from pathlib import Path as _P
        from silica.kernel.vault_manifest import MANIFEST_REL, reset_manifest_cache
        _KEYS = {"cooccurrence_lang", "conventions.language",
                 "conventions.reply_language", "conventions.max_tags"}
        mf = _P(CONFIG.vault_path) / MANIFEST_REL
        data = _yaml.safe_load(mf.read_text(encoding="utf-8")) if mf.exists() else {}
        if not isinstance(data, dict):
            data = {}
        args = parts[1:]
        if not args:
            CONSOLE.print(f"  [bold]{mf}[/]")
            body = _yaml.safe_dump(data, allow_unicode=True, sort_keys=False).rstrip() \
                if data else "(defaults — no vault.yaml yet)"
            CONSOLE.print(f"  {body}")
            CONSOLE.print(f"  Keys: {', '.join(sorted(_KEYS))} — /settings <key> <value|none>")
            return ""
        if len(args) != 2 or args[0] not in _KEYS:
            return f"Error: usage /settings <key> <value|none>. Keys: {', '.join(sorted(_KEYS))}"
        key, raw = args
        val = None if raw.lower() in ("none", "null") else (int(raw) if raw.isdigit() else raw)
        node = data
        *heads, leaf = key.split(".")
        for h in heads:
            if not isinstance(node.get(h), dict):
                node[h] = {}
            node = node[h]
        if val is None:
            node.pop(leaf, None)
        else:
            node[leaf] = val
        mf.write_text(_yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        reset_manifest_cache()
        CONSOLE.print(f"  {key} = {raw} → {mf.name} (comments in the file are not preserved)")
        return ""

    if cmd == "/convert":
        args = parts[1:]
        files = [a for a in args if not a.startswith("-")]
        target_dir = next((a[len("--target="):] for a in args if a.startswith("--target=")), "")
        if not files:
            return "Error: /convert requires at least one file path. Usage: /convert <file...> [--target=DIR]"
        from silica.sources.convert import convert
        for f in files:
            try:
                paths = convert(f, dest_dir=target_dir)
                CONSOLE.print(
                    f"  Converted {f} → [bold]{len(paths)}[/] note(s): {', '.join(paths)}"
                )
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

    # --- reader commands: agent-directed, strictly read-only ---------------

    if cmd == "/summarize":
        targets = [a for a in parts[1:] if not a.startswith("-")]
        if not targets:
            return "Error: /summarize requires a note or folder. Usage: /summarize <note|folder...>"
        listing = ", ".join(f"`{t}`" for t in targets)
        return (
            f"Summarize {listing} from the vault.\n"
            f"Resolve each target (note path, note title, or folder — list a folder's notes and "
            f"read them). Then write a digest in chat: lead with the core ideas, use tables for "
            f"anything enumerable (comparisons, parameters, timelines), keep it scannable.\n"
            f"READ-ONLY: do not create, edit, patch, or move any note."
        )

    if cmd == "/explain":
        level = ""
        words: list[str] = []
        for arg in parts[1:]:
            if arg.startswith("--level="):
                level = arg[len("--level="):]
            elif not arg.startswith("-"):
                words.append(arg)
        concept = " ".join(words).strip()
        if not concept:
            return 'Error: /explain requires a concept. Usage: /explain "<concept>" [--level=intro|expert]'
        register = {
            "intro": "for a newcomer: plain language, concrete analogies, no unexplained jargon",
            "expert": "for an expert: precise and technical, no hand-holding",
        }.get(level, "for a practitioner: clear, correct, minimal jargon")
        return (
            f"Explain {json.dumps(concept)} grounded in this vault, {register}.\n"
            f"Search the vault (semantic search + related notes), read the top matches, and explain "
            f"the concept in chat, citing every note you drew on as a [[wikilink]]. If the vault has "
            f"nothing relevant, say so plainly — do not silently answer from general knowledge alone.\n"
            f"READ-ONLY: do not create, edit, patch, or move any note."
        )

    if cmd == "/compare":
        subjects = [a for a in parts[1:] if not a.startswith("-")]
        if len(subjects) < 2:
            return 'Error: /compare requires at least two subjects. Usage: /compare "<A>" "<B>"'
        listing = ", ".join(f"`{s}`" for s in subjects)
        return (
            f"Compare {listing} using the vault.\n"
            f"Each subject is a note (path or title) or a concept — locate and read the matching "
            f"note(s) for each. Output in chat: a comparison table (one column per subject, "
            f"dimensions as rows), then a short similarities/differences rundown. If any involved "
            f"note carries `contested: true`, or the notes contradict each other, call that out "
            f"explicitly.\n"
            f"READ-ONLY: do not create, edit, patch, or move any note."
        )

    if cmd == "/quiz":
        n = 10
        targets = []
        for arg in parts[1:]:
            if arg.startswith("--n="):
                try:
                    n = int(arg[len("--n="):])
                except ValueError:
                    pass
            elif not arg.startswith("-"):
                targets.append(arg)
        if not targets:
            return "Error: /quiz requires a note or folder. Usage: /quiz <note|folder> [--n=10]"
        listing = ", ".join(f"`{t}`" for t in targets)
        return (
            f"Create a {n}-question active-recall quiz from {listing}.\n"
            f"Read the note(s) (list a folder's notes first). Mix recall, comprehension, and "
            f"application questions; ask only what the notes actually support. Output in chat: "
            f"numbered questions first, then an 'Answers' section keyed by number, each answer "
            f"citing its source note as a [[wikilink]].\n"
            f"READ-ONLY: do not create, edit, patch, or move any note."
        )

    if cmd == "/relate":
        n = 8
        targets = []
        for arg in parts[1:]:
            if arg.startswith("--n="):
                try:
                    n = int(arg[len("--n="):])
                except ValueError:
                    pass
            elif not arg.startswith("-"):
                targets.append(arg)
        if not targets:
            return "Error: /relate requires a note. Usage: /relate <note> [--n=8]"
        target = targets[0]
        return (
            f"Map how and why `{target}` relates to its most relevant neighbors in the vault.\n"
            f"Resolve the note, then pull its top {n} related notes via silica's relatedness "
            f"(the fusion of embeddings + co-occurrence). Read the target and each neighbor enough "
            f"to judge the link, and note which neighbors the target already [[wikilinks]].\n"
            f"Output in chat a Markdown table: | Neighbor | Relation | Why | Link |. "
            f"For Relation pick the type that fits — common ones: prerequisite, elaborates, "
            f"contradicts, sibling, example-of, depends-on, alternative-to. Why is one line grounded "
            f"in the notes. Link is [[the neighbor]] if already linked, else 'latent'. Cite every "
            f"neighbor as a [[wikilink]]. If a neighbor is `contested: true` or contradicts the "
            f"target, flag it.\n"
            f"READ-ONLY: do not create, edit, patch, or move any note."
        )

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

    SILICA_MAX_CONTEXT pins the window for LOCAL providers (LM Studio, Ollama),
    whose served window silica can't reliably introspect. Hosted providers
    (OpenRouter) report their own context_length, so the pin is ignored there
    and the provider's value always wins. Falls back to the static default when
    the provider is unreachable.
    """
    if not _model_configured():
        return
    if os.getenv("SILICA_MAX_CONTEXT") and CONFIG.provider != "openrouter":
        return
    from silica.agent.providers import model_limits
    window, _ = model_limits(CONFIG.provider, CONFIG.model)
    if window:
        CONFIG.max_context_tokens = window


def _dispatch_subcommand(args: list[str]) -> int | None:
    """Handle `silica doctor` / `silica init` / `silica connect` / `silica mcp` / `silica update`.

    Returns an exit code, or None when no subcommand matched (→ REPL).
    Lazy imports keep REPL startup unchanged. Module attributes (not `from`
    imports) so tests can monkeypatch run_checks / run_wizard / run_connect.
    """
    if args[:1] == ["update"]:
        import silica.update as update_mod
        return update_mod.update(check_only="--check" in args[1:])
    if args[:1] == ["doctor"]:
        import silica.onboarding.checks as checks
        results = checks.run_checks(CONFIG)
        checks.render_report(results)
        return 1 if checks.has_failures(results) else 0
    if args[:1] == ["init"]:
        import silica.onboarding.wizard as wizard_mod
        return wizard_mod.run_wizard()
    if args[:1] == ["connect"]:
        # Dispatch runs before main()'s setup (unlike --gui) — do it here.
        _activate_repo_mode()
        _announce_code_lane()
        from silica.kernel.vault_manifest import apply_manifest_to_config
        apply_manifest_to_config()
        _resolve_context_budget()
        _setup_logging(debug="--verbose" in sys.argv or "-v" in sys.argv or CONFIG.debug_logging)
        import silica.ui.connect as connect_mod
        return connect_mod.run_connect()
    if args[:1] == ["mcp"]:
        # Same bootstrap as connect, minus the REPL context meter (no agent
        # loop behind MCP tools). stdio transport: stdout is the protocol
        # channel, so plain stderr logging instead of _setup_logging's console.
        _activate_repo_mode()
        from silica.kernel.vault_manifest import apply_manifest_to_config
        apply_manifest_to_config()
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
        import silica.ui.mcp as mcp_mod
        return mcp_mod.run_mcp(all_tools="--all" in args[1:])
    return None


def _gui_port() -> int:
    """Parse `--port N` / `--port=N` from argv (default 8765)."""
    for i, a in enumerate(sys.argv):
        raw = a.split("=", 1)[1] if a.startswith("--port=") else (
            sys.argv[i + 1] if a == "--port" and i + 1 < len(sys.argv) else None
        )
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                pass
    return 8765


def main():
    """Entry point for the `silica` CLI command."""
    _args = [a for a in sys.argv[1:] if a not in ("--verbose", "-v")]
    code = _dispatch_subcommand(_args)
    if code is not None:
        sys.exit(code)
    _activate_repo_mode()
    _announce_code_lane()
    from silica.kernel.vault_manifest import apply_manifest_to_config
    apply_manifest_to_config()
    _resolve_context_budget()
    debug_mode = "--verbose" in sys.argv or "-v" in sys.argv or CONFIG.debug_logging
    _setup_logging(debug=debug_mode)

    # --gui: serve the localhost web GUI instead of the REPL (config/model/logging
    # already applied above). Blocks on uvicorn until Ctrl-C. Needs the [gui] extra.
    if "--gui" in sys.argv:
        try:
            from silica.ui.web import serve
        except ImportError:
            CONSOLE.print("  [red]La GUI richiede l'extra:[/] pip install 'silica\\[gui]'")
            sys.exit(1)
        serve(port=_gui_port())
        return

    # Obsidian bridge: host the rpc channel so the plugin can dial in and the
    # driver hot-swaps to ws (writes land through Obsidian's vault API while
    # the app is open). Silent no-op without [connect] or .obsidian/.
    from silica.ui.connect import start_bridge_thread
    _bridge = start_bridge_thread()

    print_home()
    if _bridge is not None:
        CONSOLE.print(f"  [dim]Obsidian bridge on ws://127.0.0.1:{_bridge.port}[/]\n")
    if not _model_configured():
        CONSOLE.print(_NO_MODEL_HINT)

    session = build_session()
    messages = _fresh_messages()
    collapsed: set[int] = set()  # message indices already elided by compaction

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
                collapsed = set()  # indices reset with the history
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
            # run_agent already appended the final assistant message to the
            # history — re-appending `answer` here would store it twice.
            _update_context_tokens(messages)
            collapsed = _compact_context(messages, collapsed)
        except KeyboardInterrupt:
            callback.close()
            CONSOLE.print("\n  [dim](interrupted)[/]")
        except Exception as e:
            callback.close()
            logger.exception("Agent error")
            CONSOLE.print(f"\n  [bold red]Error:[/] {e}\n")


if __name__ == "__main__":
    main()
