"""Vault manifest — declared capabilities per vault (ADR-0014).

`<vault>/vault.yaml` declares which source adapters participate, the active
domain overlay (ADR-0005 pack name) and the co-occurrence language. This is
composition, not taxonomy: there is no vault *type*. Absence of the file ⇒
retro-compatible defaults (prose always on; code on iff the vault sits
inside a git repo) — no migration required. Cached like kernel/overlay.py;
reset on /vault switch.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from silica.kernel import gitstate

logger = logging.getLogger(__name__)

MANIFEST_REL = "vault.yaml"


@dataclass(frozen=True)
class VaultConventions:
    """Per-vault authoring conventions — single source for prompt + linter.

    Consumed by `prep_delegation.render_prompt` ({LANGUAGE}/{MAX_TAGS}
    placeholders) and `ofm.ofm_lint` (LIMITS/CALLOUT_TYPES resolution).
    Defaults equal today's hardcoded values, so a vault without a
    `conventions:` block behaves bit-identically to before this existed.
    """

    language: str = "Italian"
    max_tags: int = 3
    extra_callouts: tuple[str, ...] = ()
    max_lines: int = 400
    max_chars: int = 20000


DEFAULT_CONVENTIONS = VaultConventions()


@dataclass(frozen=True)
class VaultManifest:
    sources: tuple[str, ...]
    overlay: str | None = None
    cooccurrence_lang: str | None = None
    conventions: VaultConventions = DEFAULT_CONVENTIONS


def default_sources(vault: str | Path) -> tuple[str, ...]:
    out = ["prose"]
    try:
        if vault and gitstate.find_repo_root(Path(vault)) is not None:
            out.append("code")
    except Exception:
        pass
    return tuple(out)


def _parse_conventions(raw: dict) -> VaultConventions:
    """Parse the optional `conventions:` block; malformed/missing ⇒ defaults (soft)."""
    conv_raw = raw.get("conventions")
    if conv_raw is None:
        return DEFAULT_CONVENTIONS
    if not isinstance(conv_raw, dict):
        logger.warning("vault.yaml: `conventions` must be a mapping — using defaults")
        return DEFAULT_CONVENTIONS

    language = conv_raw.get("language")
    if not (isinstance(language, str) and language):
        language = DEFAULT_CONVENTIONS.language

    max_tags = conv_raw.get("max_tags")
    if not (isinstance(max_tags, int) and not isinstance(max_tags, bool) and max_tags > 0):
        max_tags = DEFAULT_CONVENTIONS.max_tags

    extra_callouts = conv_raw.get("extra_callouts")
    if isinstance(extra_callouts, list) and all(isinstance(c, str) for c in extra_callouts):
        extra_callouts = tuple(c.lower() for c in extra_callouts)
    else:
        extra_callouts = DEFAULT_CONVENTIONS.extra_callouts

    max_lines = conv_raw.get("max_lines")
    if not (isinstance(max_lines, int) and not isinstance(max_lines, bool) and max_lines > 0):
        max_lines = DEFAULT_CONVENTIONS.max_lines

    max_chars = conv_raw.get("max_chars")
    if not (isinstance(max_chars, int) and not isinstance(max_chars, bool) and max_chars > 0):
        max_chars = DEFAULT_CONVENTIONS.max_chars

    return VaultConventions(
        language=language,
        max_tags=max_tags,
        extra_callouts=extra_callouts,
        max_lines=max_lines,
        max_chars=max_chars,
    )


def load_manifest(vault: str | Path) -> VaultManifest:
    """Parse <vault>/vault.yaml; absent or malformed ⇒ defaults (soft)."""
    defaults = VaultManifest(sources=default_sources(vault))
    if not vault:
        return defaults
    path = Path(vault) / MANIFEST_REL
    if not path.is_file():
        return defaults
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("vault.yaml: parse failed (%s) — using defaults", exc)
        return defaults
    if not isinstance(raw, dict):
        logger.warning("vault.yaml: expected a mapping — using defaults")
        return defaults

    sources = raw.get("sources")
    if isinstance(sources, list) and sources and all(isinstance(s, str) for s in sources):
        src = tuple(sources)
    else:
        if sources is not None:
            logger.warning("vault.yaml: `sources` must be a non-empty string list — using defaults")
        src = defaults.sources

    overlay = raw.get("overlay")
    lang = raw.get("cooccurrence_lang")
    return VaultManifest(
        sources=src,
        overlay=overlay if isinstance(overlay, str) and overlay else None,
        cooccurrence_lang=lang if isinstance(lang, str) and lang else None,
        conventions=_parse_conventions(raw),
    )


_cached: VaultManifest | None = None


def reset_manifest_cache() -> None:
    """Invalidate the cache. Use in tests and after /vault switch."""
    global _cached
    _cached = None


def get_active_manifest() -> VaultManifest:
    global _cached
    if _cached is None:
        from silica.config import CONFIG

        _cached = load_manifest((getattr(CONFIG, "vault_path", "") or "").strip())
    return _cached


def apply_manifest_to_config() -> None:
    """Manifest determines CONFIG fields the environment did not set (env
    wins). Symmetric on purpose: a vault that declares no overlay clears a
    previous vault's overlay on /vault switch instead of leaking it."""
    from silica.config import CONFIG

    m = get_active_manifest()
    if os.getenv("SILICA_DOMAIN") is None:
        CONFIG.domain = m.overlay
    if os.getenv("SILICA_COOCCURRENCE_LANG") is None:
        # "english" mirrors the config-level default for this field
        CONFIG.cooccurrence_lang = m.cooccurrence_lang or "english"
