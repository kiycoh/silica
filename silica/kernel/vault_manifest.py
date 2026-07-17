# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

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

from silica.kernel import paths

logger = logging.getLogger(__name__)

MANIFEST_REL = "vault.yaml"


@dataclass(frozen=True)
class VaultConventions:
    """Per-vault authoring conventions — single source for prompt + linter.

    Consumed by `prep_delegation.render_prompt` ({LANGUAGE}/{MAX_TAGS}
    placeholders) and `ofm.ofm_lint` (LIMITS/CALLOUT_TYPES resolution).
    max_tags/extra_callouts/max_lines/max_chars default to today's hardcoded
    values, so a vault without a `conventions:` block behaves bit-identically
    to before this existed for those fields.

    `language: None` (the default) means "follow the source document's
    language" — resolved per-note downstream via `kernel.language.detect`.
    A declared non-empty string means "force/translate everything into this
    language" — an explicit declaration is translation intent.

    `reply_language` is a *different* axis: the language Silica speaks in chat
    (button/slash-command turns included), independent of note content. None
    ⇒ the call site falls back to `language`, then to follow-the-user.
    """

    language: str | None = None
    reply_language: str | None = None
    max_tags: int = 3
    extra_callouts: tuple[str, ...] = ()
    wiki_dir: str = ""  # landing dir for /wiki notes; "" ⇒ vault root
    # Frontmatter templates (2026-07-17 spec): None ⇒ built-in template_spoke
    # layout — a vault with no config behaves bit-identically to before.
    default_template: str | None = None
    templates_dir: str = "templates"
    # ADR-0021: None ⇒ no episodic key enforcement (bit-identical to today).
    # Only meaningful on the MEMORY vault's manifest; other vaults ignore it.
    episodic_keys: "EpisodicKeySchema | None" = None


@dataclass(frozen=True)
class EpisodicKeySchema:
    """Declared grammar of episodic keys (ADR-0021).

    Owned by the MEMORY vault's manifest (the episodic store's home), never
    by the vault active at capture: one store, one schema. Enforcement is
    structural and write-time (see `episodic.enforce_key_schema`).
    """

    prefixes: tuple[str, ...] = ("user", "assistant")
    default_prefix: str = "user"
    max_depth: int = 3


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
        if vault and paths.repo_root_for(vault) is not None:
            out += ["code", "notebook"]
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

    # Absent/malformed (non-string, empty or whitespace-only) -> None ("follow
    # the source"). A declared non-blank string passes through unchanged
    # (translation intent) — {LANGUAGE} must always get a concrete name.
    language = conv_raw.get("language")
    if isinstance(language, str) and language.strip():
        language = language.strip()
    else:
        language = None

    reply_language = conv_raw.get("reply_language")
    if isinstance(reply_language, str) and reply_language.strip():
        reply_language = reply_language.strip()
    else:
        reply_language = None

    max_tags = conv_raw.get("max_tags")
    if not (isinstance(max_tags, int) and not isinstance(max_tags, bool) and max_tags > 0):
        max_tags = DEFAULT_CONVENTIONS.max_tags

    extra_callouts = conv_raw.get("extra_callouts")
    if isinstance(extra_callouts, list) and all(isinstance(c, str) for c in extra_callouts):
        extra_callouts = tuple(c.lower() for c in extra_callouts)
    else:
        extra_callouts = DEFAULT_CONVENTIONS.extra_callouts

    wiki_dir = conv_raw.get("wiki_dir")
    wiki_dir = wiki_dir.strip() if isinstance(wiki_dir, str) else ""
    if wiki_dir:
        # trust boundary: vault.yaml is user-authored, and wiki_dir reaches the
        # write path — a traversal or absolute path would scatter derived notes
        # outside the vault, invisible to the index, /undo and snapshots
        parts = wiki_dir.replace("\\", "/").split("/")
        if wiki_dir.startswith(("/", "\\")) or ".." in parts or ":" in parts[0]:
            logger.warning("vault.yaml: conventions.wiki_dir must be a relative "
                           "path inside the vault — ignoring %r", wiki_dir)
            wiki_dir = ""

    default_template = conv_raw.get("default_template")
    if isinstance(default_template, str) and default_template.strip():
        default_template = default_template.strip()
    else:
        default_template = None

    templates_dir = conv_raw.get("templates_dir")
    templates_dir = templates_dir.strip() if isinstance(templates_dir, str) else ""
    if templates_dir:
        # trust boundary: same rule as wiki_dir — user-authored path that
        # reaches the read path must stay inside the vault
        parts = templates_dir.replace("\\", "/").split("/")
        if templates_dir.startswith(("/", "\\")) or ".." in parts or ":" in parts[0]:
            logger.warning("vault.yaml: conventions.templates_dir must be a relative "
                           "path inside the vault — ignoring %r", templates_dir)
            templates_dir = ""
    if not templates_dir:
        templates_dir = "templates"

    episodic_keys = None
    ek_raw = conv_raw.get("episodic_keys")
    if isinstance(ek_raw, dict):
        defaults = EpisodicKeySchema()
        prefixes = ek_raw.get("prefixes")
        if not (isinstance(prefixes, list) and prefixes
                and all(isinstance(p, str) and p.strip() for p in prefixes)):
            prefixes = list(defaults.prefixes)
        default_prefix = ek_raw.get("default_prefix")
        if not (isinstance(default_prefix, str) and default_prefix.strip()):
            default_prefix = defaults.default_prefix
        max_depth = ek_raw.get("max_depth")
        if not (isinstance(max_depth, int) and not isinstance(max_depth, bool)
                and max_depth > 0):
            max_depth = defaults.max_depth
        episodic_keys = EpisodicKeySchema(
            prefixes=tuple(p.strip() for p in prefixes),
            default_prefix=default_prefix.strip(),
            max_depth=max_depth,
        )
    elif ek_raw is not None:
        logger.warning("vault.yaml: `episodic_keys` must be a mapping — "
                       "no key schema (enforcement off)")

    return VaultConventions(
        language=language,
        reply_language=reply_language,
        max_tags=max_tags,
        extra_callouts=extra_callouts,
        wiki_dir=wiki_dir,
        default_template=default_template,
        templates_dir=templates_dir,
        episodic_keys=episodic_keys,
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
        # "auto" mirrors the config-level default for this field (per-store
        # detection, frozen at build — see kernel/cooccurrence.py). A vault
        # without a declared cooccurrence_lang must NOT be silently pinned to
        # english.
        CONFIG.cooccurrence_lang = m.cooccurrence_lang or "auto"
