# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica init` — interactive setup wizard. Writes .env, then runs the doctor checks."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

from silica.config import SilicaConfig
from silica.kernel import gitstate
from silica.kernel.vault_manifest import MANIFEST_REL
from silica.onboarding.checks import has_failures, render_report, run_checks
from silica.ui.banner import print_banner
from silica.ui.console import CONSOLE
from silica.ui.style import GLYPHS

_STEPS = 4

_KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")

_LANG_PROMPT = (
    "Force a language for distilled notes? "
    "[Enter = no, follow the source language]"
)
# Bare language names only: letters and spaces. Rejects punctuation (a colon
# above all — see _ask_language) that would corrupt the raw YAML the answer
# is embedded into.
_LANG_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z ]*$")
# YAML 1.1 boolean literals: they'd pass the letters-only regex above but
# parse as `True`/`False`, which `_parse_conventions` folds to None — the
# user would believe they forced a language but silently didn't.
_LANG_ANSWER_REJECT = {"y", "n", "yes", "no", "true", "false", "on", "off"}


def merge_env(existing: str, updates: dict[str, str]) -> str:
    """Update KEY=VALUE lines in place, preserve every other line untouched,
    append keys that were not present. Never deletes a line it did not write."""
    pending = dict(updates)
    out: list[str] = []
    for line in existing.splitlines():
        m = _KEY_RE.match(line)
        if m and m.group(1) in pending:
            key = m.group(1)
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)
    for key, value in pending.items():
        out.append(f"{key}={value}")
    text = "\n".join(out)
    return text + "\n" if text else ""


def _ask(
    input_fn: Callable[[str], str],
    prompt: str,
    default: str = "",
    *,
    secret: bool = False,
) -> str:
    shown = f"…{default[-4:]}" if (secret and default) else default
    suffix = f" [{shown}]" if default else ""
    try:
        # `→` gutter marks every question with the TUI's arrow glyph (same one
        # render_report uses for hints). Plain text: input() ignores markup.
        raw = input_fn(f"  {GLYPHS['arrow']} {prompt}{suffix}: ").strip()
    except (EOFError, StopIteration):
        # EOF (Ctrl+D) or an exhausted scripted input — treat like Ctrl+C.
        raise KeyboardInterrupt
    return raw or default


def _section(glyph_key: str, title: str, n: int) -> None:
    """Flat-gutter step header in the TUI's brand vocabulary: glyph + title in
    bold brand cyan, a dim `· n/N` counter riding after it."""
    CONSOLE.print()
    CONSOLE.print(f"  [bold brand.cyan]{GLYPHS[glyph_key]} {title}[/]  [dim]· {n}/{_STEPS}[/]")


def _ask_language(input_fn: Callable[[str], str]) -> str:
    """Ask the "force a language" question and return an answer safe to embed
    raw into vault.yaml: either a plausible bare language name or "" (no
    language forced — same as Enter).

    Both call sites below splice the answer directly into unquoted YAML.
    Left unvalidated: "yes"/"no"/"true" etc. parse as YAML booleans that
    `_parse_conventions` folds to None (the user believes they forced a
    language but silently didn't), and any other stray punctuation — a colon
    above all — can break the surrounding YAML, degrading the WHOLE manifest
    (in repo mode this silently drops sources/overlay too). Anything that
    isn't a bare name is treated as no answer rather than risking either
    failure mode.
    """
    raw = _ask(input_fn, _LANG_PROMPT).strip()
    if not raw:
        return ""
    if not _LANG_NAME_RE.match(raw) or raw.lower() in _LANG_ANSWER_REJECT:
        CONSOLE.print(
            f"  [yellow]'{raw}' doesn't look like a language name — skipping "
            "(no language forced; distiller follows the source language).[/]"
        )
        return ""
    return raw


def _run_wizard_inner(
    input_fn: Callable[[str], str],
    env_path: Path,
) -> int:
    updates: dict[str, str] = {}
    repo_root = gitstate.find_repo_root(env_path.parent)

    print_banner()
    CONSOLE.print()
    CONSOLE.print(
        "  [bold]Interactive setup[/]  [dim]· press Enter to accept a shown default[/]"
    )

    # 1. Vault — repo mode (docs/silica/) when inside a git repo, else explicit
    # path. An Obsidian-vault repo (.obsidian/) is adopted verbatim instead.
    _section("vault", "Vault", 1)
    use_repo_mode = False
    if repo_root is not None:
        from silica.kernel.paths import is_obsidian_vault, repo_mode_vault

        repo_vault = Path(repo_root) if is_obsidian_vault(repo_root) else repo_mode_vault(repo_root)
        state = "exists" if repo_vault.is_dir() else "will be created"
        answer = _ask(
            input_fn,
            f"Git repo detected — use repo mode? vault = {repo_vault} ({state}) [y/n]",
            "y",
        )
        if answer.lower() in ("y", "yes"):
            use_repo_mode = True
            repo_vault.mkdir(parents=True, exist_ok=True)
            manifest = repo_vault / MANIFEST_REL
            if not manifest.exists():
                # Declared capabilities (ADR-0014): repo-mode vault wants the
                # codebase overlay and the code source active.
                lang_answer = _ask_language(input_fn)
                content = "sources: [prose, code]\noverlay: codebase\n"
                if lang_answer:
                    # cooccurrence_lang (stemmer/stopwords) is separate from
                    # conventions.language (distiller translation intent). Pin
                    # both from the one answer so the co-occurrence store never
                    # falls back to fragile auto-detection.
                    content += f"cooccurrence_lang: {lang_answer.lower()}\n"
                    content += f"conventions:\n  language: {lang_answer}\n"
                manifest.write_text(content, encoding="utf-8")
    if not use_repo_mode:
        while True:
            path = _ask(input_fn, "Vault path (existing directory)")
            resolved = Path(path).expanduser() if path else None
            if resolved is not None and resolved.is_dir():
                updates["SILICA_VAULT"] = str(resolved)
                break
            CONSOLE.print(f"  [red]{GLYPHS['err']} Not a directory — try again.[/]")
        # The design's language question is unscoped to repo mode ("init asks
        # whether to force a language"): an explicit-path vault with no
        # vault.yaml yet must be asked too. Unlike repo mode there is no other
        # content due to be written for this vault, so Enter writes nothing —
        # a vault.yaml wouldn't otherwise exist, and conventions is the only
        # thing this question could ever put in it. An existing manifest is
        # never touched, and the question is skipped entirely in that case.
        manifest = resolved / MANIFEST_REL
        if not manifest.exists():
            lang_answer = _ask_language(input_fn)
            if lang_answer:
                # Pin cooccurrence_lang (stemmer) alongside conventions.language
                # (distiller) — two separate axes, one answer. See repo-mode note.
                manifest.write_text(
                    f"cooccurrence_lang: {lang_answer.lower()}\n"
                    f"conventions:\n  language: {lang_answer}\n",
                    encoding="utf-8",
                )

    # 2. Chat provider — only the two PROVIDER_PRESETS entries exist.
    _section("model", "Chat provider", 2)
    provider = ""
    while provider not in ("lmstudio", "openrouter"):
        provider = _ask(
            input_fn,
            "Chat provider — lmstudio (local, no key) or openrouter (hosted)",
            "lmstudio",
        )
    updates["SILICA_PROVIDER"] = provider
    if provider == "openrouter":
        model = _ask(input_fn, "Model id", "openrouter/anthropic/claude-sonnet-5")
        key = ""
        while not key:
            key = _ask(
                input_fn, "OpenRouter API key",
                os.getenv("OPENROUTER_API_KEY", ""),
                secret=True,
            )
        updates["OPENROUTER_API_KEY"] = key
    else:
        model = ""
        while not model:
            model = _ask(input_fn, "Model id as loaded in LM Studio (e.g. qwen3-30b)")
    updates["SILICA_MODEL"] = model

    # 3. Embeddings — optional; skipping degrades gracefully.
    _section("think", "Embeddings", 3)
    defaults = SilicaConfig()
    answer = _ask(
        input_fn,
        "Configure embeddings? `skip` degrades dedup//find to co-occurrence [y/skip]",
        "y",
    )
    if answer.lower() in ("skip", "s", "n", "no"):
        CONSOLE.print(
            "  [yellow]Embeddings skipped — dedup routing and /find need them; "
            "relatedness falls back to co-occurrence.[/]"
        )
    else:
        updates["SILICA_EMBEDDING_MODEL"] = _ask(
            input_fn, "Embedding model", defaults.embedding_model
        )
        updates["SILICA_EMBEDDING_BASE_URL"] = _ask(
            input_fn, "Embedding base URL", defaults.embedding_base_url
        )
        updates["SILICA_EMBEDDING_API_KEY"] = _ask(
            input_fn, "Embedding API key", defaults.embedding_api_key
        )

    # 4. Confirm and write.
    _section("arrow", "Write configuration", 4)
    CONSOLE.print(
        f"  {len(updates)} key(s) → [bold]{env_path}[/]: "
        f"[dim]{', '.join(sorted(updates))}[/]"
    )
    answer = _ask(input_fn, "Write? [y/n]", "y")
    if answer.lower() not in ("y", "yes"):
        CONSOLE.print(f"  [dim]{GLYPHS['err']} Aborted — nothing written.[/]")
        return 1
    existing = env_path.read_text() if env_path.exists() else ""
    env_path.write_text(merge_env(existing, updates))
    CONSOLE.print(f"  [green]{GLYPHS['ok']} Wrote {env_path}[/]")

    # 5. Doctor checks against the values just chosen.
    CONSOLE.print()
    CONSOLE.print(f"  [bold brand.cyan]{GLYPHS['run']} Checking your setup[/]")
    os.environ.update(updates)
    results = run_checks(SilicaConfig())
    render_report(results)
    return 1 if has_failures(results) else 0


def run_wizard(
    input_fn: Callable[[str], str] = input,
    env_path: Path | None = None,
) -> int:
    cwd = Path.cwd()
    if env_path is None:
        repo_root = gitstate.find_repo_root(cwd)
        env_path = (Path(repo_root) if repo_root else cwd) / ".env"
    try:
        return _run_wizard_inner(input_fn, env_path)
    except KeyboardInterrupt:
        CONSOLE.print(
            f"\n  [dim]{GLYPHS['err']} Aborted — nothing written beyond what was already confirmed.[/]"
        )
        return 1
