# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Onboarding checks — pure diagnostics shared by `silica doctor` and `silica init`.

Each check reads config / env / filesystem / HTTP and returns a CheckResult.
No check mutates state and none makes a paid LLM completion call — key
presence and HTTP reachability only.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from silica.agent.providers import PROVIDER_PRESETS
from silica.config import SilicaConfig
from silica.kernel import gitstate

_HTTP_TIMEOUT = 3.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str
    hint: str = ""


def check_chat_model(config: SilicaConfig) -> CheckResult:
    if not config.model.strip():
        return CheckResult(
            "chat model", "fail",
            "SILICA_MODEL is not set",
            "run `silica init`",
        )
    if config.provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        return CheckResult(
            "chat model", "fail",
            f"{config.model} — provider openrouter but OPENROUTER_API_KEY is unset",
            "export OPENROUTER_API_KEY or run `silica init`",
        )
    return CheckResult("chat model", "ok", f"{config.model} via {config.provider}")


def check_chat_endpoint(config: SilicaConfig) -> CheckResult:
    if not config.model.strip():
        return CheckResult("chat endpoint", "warn", "skipped — no model configured")
    if config.provider != "lmstudio":
        return CheckResult(
            "chat endpoint", "ok", f"{config.provider} (hosted, not probed)"
        )
    base_url = PROVIDER_PRESETS["lmstudio"]["base_url"]
    try:
        httpx.get(f"{base_url}/models", timeout=_HTTP_TIMEOUT)
    except Exception:
        return CheckResult(
            "chat endpoint", "fail",
            f"{base_url} unreachable",
            "start LM Studio, or switch provider with `silica init`",
        )
    return CheckResult("chat endpoint", "ok", f"{base_url} reachable")


def check_vault(config: SilicaConfig) -> CheckResult:
    vault = config.vault_path.strip()
    if vault:
        p = Path(vault)
        if not p.is_dir():
            return CheckResult(
                "vault", "fail", f"{vault} does not exist",
                "fix SILICA_VAULT or run `silica init`",
            )
        if not os.access(p, os.W_OK):
            return CheckResult("vault", "fail", f"{vault} is not writable", "fix permissions")
        if not (p / config.inbox_dir).is_dir():
            return CheckResult(
                "vault", "warn",
                f"{vault} ok, but inbox folder `{config.inbox_dir}/` is missing",
                f"create `{config.inbox_dir}/` inside the vault for ingestion",
            )
        return CheckResult("vault", "ok", vault)
    root = gitstate.find_repo_root(Path.cwd())
    if root is not None:
        from silica.kernel.paths import is_obsidian_vault, repo_mode_vault

        vault_dir = Path(root) if is_obsidian_vault(root) else repo_mode_vault(root)
        if vault_dir.is_dir():
            return CheckResult("vault", "ok", f"repo mode → {vault_dir}")
    if config.backend == "fs":
        return CheckResult(
            "vault", "fail",
            "SILICA_VAULT not set and no docs/silica/ in this repo",
            "set SILICA_VAULT=/path/to/vault in .env, or run `silica init`",
        )
    return CheckResult(
        "vault", "warn",
        "SILICA_VAULT not set and no docs/silica/ in this repo",
        "run `silica init`",
    )


def check_obsidian_backend(config: SilicaConfig) -> CheckResult:
    if config.backend == "fs":
        # fs is the default: filesystem-native, no Obsidian required.
        # Vault configuration is check_vault's responsibility — report ok here.
        return CheckResult(
            "obsidian backend", "ok",
            "filesystem-native (headless — Obsidian not required)",
        )
    # backend == "cli": Obsidian desktop is an opt-in enhancement.
    if shutil.which("obsidian") is None:
        return CheckResult(
            "obsidian backend", "fail",
            "`obsidian` binary not on PATH",
            "install the Obsidian CLI, or set SILICA_BACKEND=fs for headless use",
        )
    # Responsiveness probe: the CLI is a CDP bridge — when the desktop app is
    # closed it hangs instead of erroring, so only a timeout (or a missing
    # binary) counts as failure; any completed process means the bridge is up.
    try:
        subprocess.run(
            ["obsidian", "version"],
            capture_output=True, text=True,
            timeout=config.obsidian_cli_timeout,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            "obsidian backend", "fail",
            f"`obsidian` did not respond within {config.obsidian_cli_timeout:.0f}s",
            "open the Obsidian desktop app, or set SILICA_BACKEND=fs",
        )
    return CheckResult("obsidian backend", "ok", "`obsidian` CLI responds")


def check_embeddings(config: SilicaConfig) -> CheckResult:
    # Never "fail": relatedness degrades to the co-occurrence leg by design.
    url = f"{config.embedding_base_url.rstrip('/')}/models"
    try:
        data = httpx.get(url, timeout=_HTTP_TIMEOUT).json()
    except Exception:
        return CheckResult(
            "embeddings", "warn",
            f"{config.embedding_base_url} unreachable",
            "dedup routing and /find need embeddings; relatedness falls back to co-occurrence",
        )
    ids = {m.get("id", "") for m in data.get("data", [])}
    if config.embedding_model not in ids:
        return CheckResult(
            "embeddings", "warn",
            f"model `{config.embedding_model}` not listed at {config.embedding_base_url}",
            "load the embedding model, or update SILICA_EMBEDDING_MODEL",
        )
    return CheckResult(
        "embeddings", "ok",
        f"{config.embedding_model} @ {config.embedding_base_url}",
    )


_LANG_SAMPLE_MAX_FILES = 30
_LANG_SAMPLE_PER_FILE_CHARS = 150
_LANG_SAMPLE_TOTAL_CHARS = 4000


def sample_vault_text(vault: str) -> str:
    """Deterministic, cheap sample of a vault's prose for language detection.

    Up to `_LANG_SAMPLE_MAX_FILES` `.md` files (sorted rglob — deterministic
    across runs/platforms), the first `_LANG_SAMPLE_PER_FILE_CHARS` characters
    of each, concatenated and capped at `_LANG_SAMPLE_TOTAL_CHARS`. The
    per-file cap is kept small (well under total/max_files) so the budget is
    actually SPREAD across the file cap rather than exhausted by the first
    handful of alphabetically-sorted files — a minority-language head (e.g.
    a lone "AAA notes.md") must not drown out the vault's real majority
    language, which only shows up once later files get sampled too. Returns
    "" when the vault has no readable `.md` files. Degrades on any
    filesystem error instead of raising — matches this module's
    pure-diagnostic contract.

    Single seam for this sampling logic: both `check_language` (doctor) and
    the `/vault` info block in cli.py go through `detect_vault_language`
    below, which calls this — no duplicated sampling.
    """
    try:
        files = sorted(Path(vault).rglob("*.md"))[:_LANG_SAMPLE_MAX_FILES]
    except Exception:
        return ""
    parts: list[str] = []
    total = 0
    for f in files:
        if total >= _LANG_SAMPLE_TOTAL_CHARS:
            break
        try:
            chunk = f.read_text(encoding="utf-8", errors="ignore")[:_LANG_SAMPLE_PER_FILE_CHARS]
        except Exception:
            continue
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)[:_LANG_SAMPLE_TOTAL_CHARS]


def detect_vault_language(vault: str) -> str | None:
    """Cheap, deterministic dominant-language detection for `vault`.

    None when there is nothing to sample (no `.md` files, or all unreadable)
    — callers treat that as "no notes yet". Never raises.
    """
    if not vault:
        return None
    sample = sample_vault_text(vault)
    if not sample.strip():
        return None
    from silica.kernel import language

    return language.detect(sample)


def frozen_store_language(vault: str) -> str | None:
    """Read `vault`'s persisted cooccurrence store's frozen `lang` field, if
    a store exists on disk for THIS vault.

    Thin pass-through to `kernel.cooccurrence.frozen_lang` — this module
    owns no on-disk store schema knowledge; the store's own module does.
    Resolved from the `vault` argument, never from the global CONFIG
    singleton, so a caller comparing a specific (possibly non-active) vault
    never cross-checks a different vault's store. None when no store file
    exists yet, or on any read/parse error (degrade, never raise — inherited
    from the accessor this delegates to).

    Direct leg import — allowlisted in tests/test_relatedness_boundary.py:
    metadata-only read via the public accessor, no store construction.
    """
    from silica.kernel.cooccurrence import frozen_lang

    return frozen_lang(vault)


def declared_language(vault: str) -> str | None:
    """The language `vault` DECLARES in its `vault.yaml` (`cooccurrence_lang`),
    or None when it declares none — or declares the `auto` sentinel, meaning
    "detect me". A declaration is authority: it is the language the
    co-occurrence store is (or should be) frozen to, so it SUPERSEDES the
    stopword heuristic. A frontmatter-heavy sample fools `detect` into
    "english" (the bundled english stoplist matches `last:`/`related:`/`null`
    scaffolding), but a vault that declares italian is italian.

    Resolved from the `vault` argument, never the global CONFIG singleton —
    same contract as frozen_store_language above.
    """
    from silica.kernel.vault_manifest import load_manifest

    lang = load_manifest(vault).cooccurrence_lang
    return lang if lang and lang != "auto" else None


def language_status(vault: str) -> tuple[str | None, str | None, bool]:
    """`(authority, store, drift)` for `vault` — the single seam behind both
    the doctor's `check_language` and the `/vault` info block.

    authority = the declared language if any, else the heuristically detected
    dominant language (None when there is nothing to sample). store = the
    frozen co-occurrence store language (None if no store on disk yet). drift =
    both known and differing — the signal that `/cooccur --force` is needed to
    rebuild the store in the authoritative language.
    """
    authority = declared_language(vault) or detect_vault_language(vault)
    store = frozen_store_language(vault) if authority else None
    return authority, store, bool(authority and store and authority != store)


def check_language(config: SilicaConfig) -> CheckResult:
    """The vault's authoritative language (declared in vault.yaml, else
    detected) vs. the cooccurrence store's frozen language. A divergence is
    the signature of the historic bug that froze stores to "english" on
    non-English vaults — this is how existing users discover a store needs a
    `/cooccur` rebuild.

    Resolved from `config.vault_path` — never from the global CONFIG
    singleton — so a caller that just reconfigured (e.g. the init wizard
    building a fresh `SilicaConfig()` right after a vault switch) never
    compares the newly-chosen vault against a *different*, still-active
    vault's frozen store.
    """
    vault = config.vault_path.strip()
    if not vault:
        return CheckResult("language", "ok", "no vault — skipped")

    authority, store_lang, drift = language_status(vault)
    if authority is None:
        return CheckResult("language", "ok", "no notes yet")
    if store_lang is None:
        return CheckResult("language", "ok", f"language={authority}, no store frozen yet")
    if not drift:
        return CheckResult("language", "ok", f"language={authority}, store={store_lang}")
    return CheckResult(
        "language", "warn",
        f"language={authority}, store frozen={store_lang} — mismatch",
        "run `/cooccur --force` to rebuild the co-occurrence store in the vault's language",
    )


def check_manifest(config: SilicaConfig) -> CheckResult:
    from silica.kernel.vault_manifest import MANIFEST_REL, load_manifest
    from silica.sources.registry import ALL_ADAPTERS

    vault = config.vault_path.strip()
    if not vault:
        return CheckResult("vault manifest", "ok", "no vault — defaults apply")
    path = Path(vault) / MANIFEST_REL
    if not path.is_file():
        return CheckResult("vault manifest", "ok", "absent — retro-compatible defaults")
    m = load_manifest(vault)
    known = {a.name for a in ALL_ADAPTERS}
    unknown = [s for s in m.sources if s not in known]
    if unknown:
        return CheckResult(
            "vault manifest", "warn",
            f"unknown source(s) {unknown} in {MANIFEST_REL}",
            f"known sources: {sorted(known)}",
        )
    detail = f"sources={list(m.sources)}"
    if m.overlay:
        detail += f", overlay={m.overlay}"
    return CheckResult("vault manifest", "ok", detail)


def run_checks(config: SilicaConfig) -> list[CheckResult]:
    return [
        check_chat_model(config),
        check_chat_endpoint(config),
        check_vault(config),
        check_manifest(config),
        check_language(config),
        check_obsidian_backend(config),
        check_embeddings(config),
    ]


def has_failures(results: list[CheckResult]) -> bool:
    return any(r.status == "fail" for r in results)


_STATUS_GLYPH = {"ok": ("✓", "green"), "warn": ("⚠", "yellow"), "fail": ("✗", "red")}


def render_report(results: list[CheckResult]) -> None:
    from rich.table import Table

    from silica.ui.console import CONSOLE

    table = Table(show_header=False, box=None, padding=(0, 1))
    for r in results:
        glyph, style = _STATUS_GLYPH[r.status]
        hint = f"[dim]→ {r.hint}[/]" if r.hint else ""
        table.add_row(f"[{style}]{glyph}[/]", f"[bold]{r.name}[/]", r.detail, hint)
    CONSOLE.print()
    CONSOLE.print(table)
    CONSOLE.print()
