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
from silica.ui.console import CONSOLE

_KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


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
        raw = input_fn(f"  {prompt}{suffix}: ").strip()
    except (EOFError, StopIteration):
        # EOF (Ctrl+D) or an exhausted scripted input — treat like Ctrl+C.
        raise KeyboardInterrupt
    return raw or default


def _run_wizard_inner(
    input_fn: Callable[[str], str],
    env_path: Path,
) -> int:
    updates: dict[str, str] = {}
    repo_root = gitstate.find_repo_root(env_path.parent)

    CONSOLE.print("\n  [bold]silica init[/] — interactive setup\n")

    # 1. Vault — repo mode (.silica/) when inside a git repo, else explicit path.
    use_repo_mode = False
    if repo_root is not None:
        repo_vault = Path(repo_root) / ".silica"
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
                manifest.write_text(
                    "sources: [prose, code]\noverlay: codebase\n", encoding="utf-8"
                )
    if not use_repo_mode:
        while True:
            path = _ask(input_fn, "Vault path (existing directory)")
            resolved = Path(path).expanduser() if path else None
            if resolved is not None and resolved.is_dir():
                updates["SILICA_VAULT"] = str(resolved)
                break
            CONSOLE.print("  [red]Not a directory — try again.[/]")

    # 2. Backend — fs is the default (filesystem-native, headless, no Obsidian required).
    # cli is an opt-in enhancement: adds version-history rollback, live metadata-cache
    # reads, and user link-format preference in autolink (requires Obsidian desktop).
    backend = ""
    while backend not in ("cli", "fs"):
        backend = _ask(
            input_fn,
            "Backend — fs (default, headless) or cli (Obsidian desktop, adds rollback + live cache)",
            "fs",
        )
    updates["SILICA_BACKEND"] = backend

    # 3. Chat provider — only the two PROVIDER_PRESETS entries exist.
    provider = ""
    while provider not in ("lmstudio", "openrouter"):
        provider = _ask(
            input_fn,
            "Chat provider — lmstudio (local, no key) or openrouter (hosted)",
            "lmstudio",
        )
    updates["SILICA_PROVIDER"] = provider
    if provider == "openrouter":
        model = _ask(input_fn, "Model id", "openrouter/openai/gpt-4o-mini")
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

    # 4. Embeddings — optional; skipping degrades gracefully.
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

    # 5. Confirm and write.
    CONSOLE.print(
        f"\n  Will write {len(updates)} key(s) to [bold]{env_path}[/]: "
        f"{', '.join(sorted(updates))}"
    )
    answer = _ask(input_fn, "Write? [y/n]", "y")
    if answer.lower() not in ("y", "yes"):
        CONSOLE.print("  Aborted — nothing written.")
        return 1
    existing = env_path.read_text() if env_path.exists() else ""
    env_path.write_text(merge_env(existing, updates))
    CONSOLE.print(f"  [green]Wrote {env_path}[/]")

    # 6. Doctor checks against the values just chosen.
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
        CONSOLE.print("\n  Aborted — nothing written beyond what was already confirmed.")
        return 1
