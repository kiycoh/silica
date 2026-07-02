"""Per-vault `conventions:` contract (spec-hermes-coherence §2).

Language, max_tags, callout whitelist and size limits become a single-source
contract read from vault.yaml, instead of being hardcoded/duplicated across
the distiller prompt (`{LANGUAGE}`/`{MAX_TAGS}`) and `ofm.LIMITS`/
`ofm.CALLOUT_TYPES`. Absence of a `conventions:` block (or of a manifest at
all) must reproduce today's hardcoded values bit-for-bit.
"""
from __future__ import annotations

from silica.config import CONFIG
from silica.kernel.vault_manifest import (
    VaultConventions,
    load_manifest,
    reset_manifest_cache,
)

# NB: the conftest.py `_reset_manifest_cache` autouse fixture clears the
# module-level manifest cache before every test; `reset_manifest_cache()` is
# still called explicitly after writing a vault.yaml mid-test to force a
# fresh read against the file we just wrote.


# ---------------------------------------------------------------------------
# load_manifest: conventions block parsing
# ---------------------------------------------------------------------------

def test_conventions_default_when_no_manifest(tmp_path):
    m = load_manifest(tmp_path)
    assert m.conventions == VaultConventions(
        language="Italian", max_tags=3, extra_callouts=(), max_lines=400, max_chars=20000,
    )


def test_conventions_parsed_from_vault_yaml(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n"
        "  language: english\n"
        "  max_tags: 5\n"
        "  extra_callouts: [clinica]\n"
        "  max_lines: 300\n"
        "  max_chars: 15000\n",
        encoding="utf-8",
    )
    m = load_manifest(tmp_path)
    assert m.conventions.language == "english"
    assert m.conventions.max_tags == 5
    assert m.conventions.extra_callouts == ("clinica",)
    assert m.conventions.max_lines == 300
    assert m.conventions.max_chars == 15000


def test_conventions_partial_block_defaults_missing_keys(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  language: english\n", encoding="utf-8"
    )
    m = load_manifest(tmp_path)
    assert m.conventions.language == "english"
    assert m.conventions.max_tags == 3          # default, unset
    assert m.conventions.extra_callouts == ()    # default, unset


def test_conventions_non_mapping_block_degrades_to_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text("conventions: not-a-mapping\n", encoding="utf-8")
    m = load_manifest(tmp_path)
    assert m.conventions == VaultConventions()


def test_conventions_bad_field_types_degrade_to_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n"
        "  max_tags: not-a-number\n"
        "  extra_callouts: also-not-a-list\n"
        "  max_lines: -1\n",
        encoding="utf-8",
    )
    m = load_manifest(tmp_path)
    assert m.conventions.max_tags == 3
    assert m.conventions.extra_callouts == ()
    assert m.conventions.max_lines == 400


def test_conventions_extra_callouts_normalized_lowercase(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  extra_callouts: [Clinica, TRIAGE]\n", encoding="utf-8"
    )
    m = load_manifest(tmp_path)
    assert m.conventions.extra_callouts == ("clinica", "triage")


# ---------------------------------------------------------------------------
# render_prompt: {LANGUAGE} / {MAX_TAGS} placeholder substitution
# ---------------------------------------------------------------------------

def test_render_prompt_defaults_match_today_hardcoded_values(monkeypatch):
    """No manifest ⇒ bit-identical to the previously hardcoded prompt text."""
    monkeypatch.setattr(CONFIG, "vault_path", "")
    from silica.kernel.prep_delegation import render_prompt

    rendered = render_prompt(target="Concepts/AI")
    assert "written in Italian" in rendered
    assert "at most **3 tags**" in rendered
    assert "{LANGUAGE}" not in rendered
    assert "{MAX_TAGS}" not in rendered


def test_render_prompt_uses_vault_conventions(tmp_path, monkeypatch):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  language: english\n  max_tags: 5\n", encoding="utf-8"
    )
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    reset_manifest_cache()
    from silica.kernel.prep_delegation import render_prompt

    rendered = render_prompt(target="Concepts/AI")
    assert "written in english" in rendered
    assert "at most **5 tags**" in rendered
    assert "{LANGUAGE}" not in rendered
    assert "{MAX_TAGS}" not in rendered


# ---------------------------------------------------------------------------
# ofm_lint: LIMITS (max_tags) + CALLOUT_TYPES resolved from the active manifest
# ---------------------------------------------------------------------------

_NOTE_TMPL = """---
parent note: "[[Hub]]"
tags:
{tags}
last modified: 2026, 07, 02
AI: true
---

# Title

Body text with [[Hub]].
"""


def _note_with_n_tags(n: int) -> str:
    tags = "\n".join(f"  - tag{i}" for i in range(n))
    return _NOTE_TMPL.format(tags=tags)


def test_ofm_lint_default_max_tags_unchanged(monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", "")
    from silica.kernel.ofm import ofm_lint

    flags = ofm_lint(_note_with_n_tags(4))["flags"]
    assert any("too many tags (4); max 3" in f for f in flags)


def test_ofm_lint_accepts_max_tags_from_manifest(tmp_path, monkeypatch):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  max_tags: 5\n", encoding="utf-8"
    )
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    reset_manifest_cache()
    from silica.kernel.ofm import ofm_lint

    flags = ofm_lint(_note_with_n_tags(5))["flags"]
    assert not any("too many tags" in f for f in flags)


def test_ofm_lint_rejects_unknown_callout_by_default(monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", "")
    from silica.kernel.ofm import ofm_lint

    note = _note_with_n_tags(1) + "\n> [!clinica] some clinical note\n"
    violations = ofm_lint(note)["violations"]
    assert any("unknown callout type" in v for v in violations)


def test_ofm_lint_extra_callouts_whitelisted_from_manifest(tmp_path, monkeypatch):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  extra_callouts: [clinica]\n", encoding="utf-8"
    )
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    reset_manifest_cache()
    from silica.kernel.ofm import ofm_lint

    note = _note_with_n_tags(1) + "\n> [!clinica] some clinical note\n"
    violations = ofm_lint(note)["violations"]
    assert not any("unknown callout type" in v for v in violations)
