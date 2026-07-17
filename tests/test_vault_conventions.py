"""Per-vault `conventions:` contract (spec-hermes-coherence §2).

Language, max_tags, callout whitelist and size limits become a single-source
contract read from vault.yaml, instead of being hardcoded/duplicated across
the distiller prompt (`{LANGUAGE}`/`{MAX_TAGS}`) and `ofm.LIMITS`/
`ofm.CALLOUT_TYPES`. Absence of a `conventions:` block (or of a manifest at
all) must reproduce today's hardcoded values bit-for-bit.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        language=None, max_tags=3, extra_callouts=(),
    )


def test_conventions_parsed_from_vault_yaml(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n"
        "  language: english\n"
        "  max_tags: 5\n"
        "  extra_callouts: [clinica]\n",
        encoding="utf-8",
    )
    m = load_manifest(tmp_path)
    assert m.conventions.language == "english"
    assert m.conventions.max_tags == 5
    assert m.conventions.extra_callouts == ("clinica",)


def test_conventions_partial_block_defaults_missing_keys(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  language: english\n", encoding="utf-8"
    )
    m = load_manifest(tmp_path)
    assert m.conventions.language == "english"
    assert m.conventions.max_tags == 3          # default, unset
    assert m.conventions.extra_callouts == ()    # default, unset


def test_conventions_whitespace_only_language_folds_to_none(tmp_path):
    """A whitespace-only `language: '   '` is not a concrete language name —
    it must fold to None (follow the source), never leak into {LANGUAGE}."""
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  language: '   '\n", encoding="utf-8"
    )
    m = load_manifest(tmp_path)
    assert m.conventions.language is None


def test_conventions_declared_language_is_stripped(tmp_path):
    """Finding 6 (final multilingua review): `_parse_conventions` checks
    `.strip()` truthiness to accept the field but must STORE the stripped
    value — " Italian " must reach {LANGUAGE} as "Italian", not with
    leading/trailing whitespace baked in."""
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  language: ' Italian '\n", encoding="utf-8"
    )
    m = load_manifest(tmp_path)
    assert m.conventions.language == "Italian"


def test_conventions_non_mapping_block_degrades_to_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text("conventions: not-a-mapping\n", encoding="utf-8")
    m = load_manifest(tmp_path)
    assert m.conventions == VaultConventions()


def test_conventions_bad_field_types_degrade_to_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n"
        "  max_tags: not-a-number\n"
        "  extra_callouts: also-not-a-list\n",
        encoding="utf-8",
    )
    m = load_manifest(tmp_path)
    assert m.conventions.max_tags == 3
    assert m.conventions.extra_callouts == ()


def test_conventions_extra_callouts_normalized_lowercase(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  extra_callouts: [Clinica, TRIAGE]\n", encoding="utf-8"
    )
    m = load_manifest(tmp_path)
    assert m.conventions.extra_callouts == ("clinica", "triage")


# ---------------------------------------------------------------------------
# render_prompt: {LANGUAGE} / {MAX_TAGS} placeholder substitution
# ---------------------------------------------------------------------------

def test_render_prompt_no_manifest_max_tags_unchanged(monkeypatch):
    """No manifest ⇒ max_tags is still bit-identical to the previously
    hardcoded prompt text (max_tags default is untouched by this design)."""
    monkeypatch.setattr(CONFIG, "vault_path", "")
    from silica.kernel.prep_delegation import render_prompt

    rendered = render_prompt(target="Concepts/AI")
    assert "at most **3 tags**" in rendered
    assert "{LANGUAGE}" not in rendered
    assert "{MAX_TAGS}" not in rendered


def test_render_prompt_no_manifest_no_source_text_degrades_to_english(monkeypatch):
    """No manifest and no source sample ⇒ language.detect("") degrades to
    "english" deterministically — {LANGUAGE} is still always a concrete name,
    never None/empty, even in the total-absence-of-signal case."""
    monkeypatch.setattr(CONFIG, "vault_path", "")
    from silica.kernel.prep_delegation import render_prompt

    rendered = render_prompt(target="Concepts/AI")
    assert "written in English" in rendered


def test_render_prompt_follows_source_language_italian(tmp_path, monkeypatch):
    """No `conventions.language` declared ⇒ follow the source: an Italian
    source sample resolves {LANGUAGE} to "Italian" via real detection."""
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    reset_manifest_cache()
    from silica.kernel.prep_delegation import render_prompt

    italian_text = (
        "Questo è un testo campione scritto interamente in lingua italiana "
        "per il rilevamento automatico della lingua del documento sorgente, "
        "che serve per verificare il comportamento di rilevamento."
    )
    rendered = render_prompt(target="Concepts/AI", source_text=italian_text)
    assert "written in Italian" in rendered


def test_render_prompt_follows_source_language_english(tmp_path, monkeypatch):
    """No `conventions.language` declared ⇒ follow the source: an English
    source sample resolves {LANGUAGE} to "English" via real detection."""
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    reset_manifest_cache()
    from silica.kernel.prep_delegation import render_prompt

    english_text = (
        "This is a sample text written entirely in the English language, "
        "used to verify that the source document's language is correctly "
        "detected and substituted into the prompt."
    )
    rendered = render_prompt(target="Concepts/AI", source_text=english_text)
    assert "written in English" in rendered


def test_render_prompt_declared_language_wins_over_source(tmp_path, monkeypatch):
    """A declared `conventions.language` is translation intent: it wins
    regardless of the source sample's detected language."""
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  language: Italian\n", encoding="utf-8"
    )
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    reset_manifest_cache()
    from silica.kernel.prep_delegation import render_prompt

    english_text = (
        "This English source text must be ignored for language selection "
        "because the manifest explicitly declares Italian as the target."
    )
    rendered = render_prompt(target="Concepts/AI", source_text=english_text)
    assert "written in Italian" in rendered


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


@patch("silica.agent.providers.get_provider")
def test_run_distiller_wires_payload_excerpts_into_language_detection(
    mock_get_provider, monkeypatch
):
    """Wiring guard: run_distiller must feed the payload's inbox_excerpt text
    into render_prompt's source_text (via _payload_sample_text), so with no
    declared conventions.language an Italian payload yields {LANGUAGE} =
    "Italian" in the prompt actually sent to the LLM. A rename of
    batches/concepts/inbox_excerpt in kernel/payload.py must fail here."""
    monkeypatch.setattr(CONFIG, "vault_path", "")  # no manifest ⇒ follow source
    from silica.kernel.prep_delegation import run_distiller

    mock_provider = MagicMock()
    mock_get_provider.return_value = mock_provider
    mock_response = MagicMock()
    mock_response.text = '{"updates": []}'
    mock_response.finish_reason = "stop"
    mock_provider.call_llm.return_value = mock_response

    payload = {
        "schema_version": 1,
        "batches": [{
            "inbox_file": "appunti.md",
            "concepts": [{
                "name": "Discesa del gradiente",
                "action_hint": "create",
                "inbox_excerpt": (
                    "Questo estratto è scritto interamente in lingua italiana "
                    "e descrive la discesa del gradiente, che serve per "
                    "verificare il rilevamento della lingua della sorgente."
                ),
                "vault_collision": None,
            }],
        }],
    }
    result = run_distiller(payload=payload, target="Concepts/AI")
    assert "error" not in result

    sent = mock_provider.call_llm.call_args.kwargs["messages"][0]["content"]
    assert "written in Italian" in sent
    assert "{LANGUAGE}" not in sent


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


# ---------------------------------------------------------------------------
# template conventions parsing
# ---------------------------------------------------------------------------


def test_template_conventions_parsed_from_vault_yaml(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  default_template: paper\n  templates_dir: tpl\n",
        encoding="utf-8",
    )
    c = load_manifest(tmp_path).conventions
    assert c.default_template == "paper"
    assert c.templates_dir == "tpl"


def test_template_conventions_defaults(tmp_path):
    c = load_manifest(tmp_path).conventions
    assert c.default_template is None
    assert c.templates_dir == "templates"


def test_templates_dir_traversal_guard(tmp_path):
    """Same trust boundary as wiki_dir: vault.yaml is user-authored and
    templates_dir reaches file reads — traversal/absolute paths fall back."""
    for bad in ("../outside", "/abs/path", "a/../../b", "C:\\evil"):
        (tmp_path / "vault.yaml").write_text(
            f"conventions:\n  templates_dir: {bad}\n", encoding="utf-8",
        )
        reset_manifest_cache()
        assert load_manifest(tmp_path).conventions.templates_dir == "templates"
