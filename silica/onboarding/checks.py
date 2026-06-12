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
    if root is not None and (Path(root) / ".silica").is_dir():
        return CheckResult("vault", "ok", f"repo mode → {Path(root) / '.silica'}")
    if config.backend == "fs":
        return CheckResult(
            "vault", "fail",
            "SILICA_VAULT not set and no .silica/ in this repo",
            "set SILICA_VAULT=/path/to/vault in .env, or run `silica init`",
        )
    return CheckResult(
        "vault", "warn",
        "SILICA_VAULT not set and no .silica/ in this repo",
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
