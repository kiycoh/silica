"""`conventions.reply_language` — chat language, distinct from note-content
`language`. Root cause it fixes: button/slash-command turns carry no natural
language, so the follow-the-user rule fell through to English."""
from __future__ import annotations

from silica.kernel.vault_manifest import load_manifest
from silica.prompts import SYSTEM_PROMPT, system_prompt


def test_reply_language_parsed_and_stripped(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  reply_language: ' Italian '\n", encoding="utf-8"
    )
    assert load_manifest(tmp_path).conventions.reply_language == "Italian"


def test_blank_reply_language_folds_to_none(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  reply_language: '   '\n", encoding="utf-8"
    )
    assert load_manifest(tmp_path).conventions.reply_language is None


def test_absent_reply_language_is_none(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  language: italian\n", encoding="utf-8"
    )
    assert load_manifest(tmp_path).conventions.reply_language is None


def test_reply_language_falls_back_to_content_language(tmp_path):
    """No reply_language ⇒ chat follows content language (the common case:
    set `language` only, chat matches it for free)."""
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  language: italian\n", encoding="utf-8"
    )
    conv = load_manifest(tmp_path).conventions
    assert (conv.reply_language or conv.language) == "italian"


def test_reply_language_overrides_content_language(tmp_path):
    """The divergence case: English papers, Italian chat."""
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  language: english\n  reply_language: italian\n",
        encoding="utf-8",
    )
    conv = load_manifest(tmp_path).conventions
    assert (conv.reply_language or conv.language) == "italian"


def test_prompt_with_preferred_language_names_it_and_covers_buttons():
    p = system_prompt("Italian")
    assert "Italian" in p
    assert "button" in p and "slash-command" in p  # the turn that had no signal


def test_prompt_without_preference_is_backcompat():
    assert system_prompt(None) == SYSTEM_PROMPT
    assert "language of the user's most recent message" in SYSTEM_PROMPT


def test_math_instruction_is_gui_only():
    """GUI renders dollar-math as MathML; TUI would show it raw, so only the
    GUI branch gets the instruction."""
    assert "$$" not in system_prompt("Italian")  # default (TUI): no math
    gui = system_prompt("Italian", math=True)
    assert "$$" in gui and "Italian" in gui  # math + language both present
