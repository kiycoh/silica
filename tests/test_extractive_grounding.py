# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Extractive invariant: every body content-line must be a verbatim span of the
source transcript. Enforced (reject/retry) under the `extractive` distill
profile so 'non-lossy' is a checked property, not a prompt hope."""

from silica.kernel.provenance import nonextractive_lines
from silica.kernel.validate import validate_operations

# Long enough that a body copied from it clears MIN_WRITE_SNIPPET_CHARS (100).
_EXCERPT = ("Elena: I finally signed up for the beginners pottery class at the "
            "community center downtown, it starts on the twentieth of May and "
            "runs every Tuesday evening with Mr. Alvarez, not my sister.")


def _payload(excerpt: str):
    return [{"batches": [{"inbox_file": "/inbox/session_1.md", "concepts": [
        {"name": "pottery class", "inbox_excerpt": excerpt, "vault_collision": None},
    ]}]}]


def _write_op(body: str):
    return {"op": "write", "path": "mem/Elena's pottery class.md",
            "heading": "pottery class", "source_basename": "session_1.md",
            "snippet": body}


def test_verbatim_lines_pass():
    src = ("Elena: I signed up for the pottery class at the community center.\n"
           "Sam: That's great!")
    body = "Elena: I signed up for the pottery class at the community center."
    assert nonextractive_lines(body, src) == []


def test_paraphrase_is_flagged():
    src = "Elena: I finally signed up for the pottery class at the community center!"
    body = "Elena enrolled in a ceramics course at the local rec hall."
    assert nonextractive_lines(body, src)  # reworded prose -> not a copied span


def test_markers_and_wikilinks_stripped():
    src = "Sam said he is switching to a new job in September at a fintech startup."
    body = ("- [[Sam]] said he is switching to a new job in September "
            "at a fintech startup.")
    # A bullet the model prepends and an autolink-shaped wikilink are structure,
    # not drift: the residual prose is still a verbatim span.
    assert nonextractive_lines(body, src) == []


def test_apostrophe_and_whitespace_normalized():
    src = "Elena: I don't teach the advanced   course, my sister does."
    body = "Elena: I don’t teach the advanced course, my sister does."  # curly + collapsed ws
    assert nonextractive_lines(body, src) == []


def test_headings_and_blank_lines_ignored():
    src = "Sam: The itinerary starts in Kyoto and ends in Osaka after five days."
    body = ("## Sam's trip\n\n"
            "Sam: The itinerary starts in Kyoto and ends in Osaka after five days.")
    assert nonextractive_lines(body, src) == []


# --- enforcement wired into validate_operations (gated) ---------------------

def test_verbatim_body_passes_under_enforce(tmp_vault, monkeypatch):
    monkeypatch.setenv("SILICA_EXTRACTIVE_ENFORCE", "1")
    verbatim = ("Elena: I finally signed up for the beginners pottery class at the "
                "community center downtown, it starts on the twentieth of May and "
                "runs every Tuesday evening with Mr. Alvarez, not my sister.")
    validated, rejected = validate_operations(
        [_write_op(verbatim)], _payload(_EXCERPT), "mem")
    assert rejected == []
    assert any(o.path == "mem/Elena's pottery class.md" for o in validated)


def test_paraphrase_rejected_under_enforce_but_passes_without(tmp_vault, monkeypatch):
    paraphrase = ("Elena enrolled in a beginners ceramics course at the local "
                  "recreation hall downtown, with the first session on May "
                  "twentieth held every Tuesday in the evening, taught by Alvarez.")
    # Gate ON -> the reworded body is rejected (routed to defer/steer).
    monkeypatch.setenv("SILICA_EXTRACTIVE_ENFORCE", "1")
    _, rejected_on = validate_operations(
        [_write_op(paraphrase)], _payload(_EXCERPT), "mem")
    assert any("extractive" in r.reason for r in rejected_on)
    # Gate OFF (default) -> the default distiller may paraphrase, so it must pass.
    monkeypatch.delenv("SILICA_EXTRACTIVE_ENFORCE", raising=False)
    validated_off, rejected_off = validate_operations(
        [_write_op(paraphrase)], _payload(_EXCERPT), "mem")
    assert not any("extractive" in r.reason for r in rejected_off)
    assert any(o.path == "mem/Elena's pottery class.md" for o in validated_off)


def test_short_verbatim_floor_is_env_lowerable(tmp_vault, monkeypatch):
    # A durable fact copied verbatim can be a short turn (<100 chars). The prose
    # placeholder floor would defer it; the extractive arm lowers it via env.
    short = "Elena: I signed up for the pottery class at the rec center."  # ~59 chars
    excerpt = short + "\nSam: Nice, when does it start?"
    monkeypatch.delenv("SILICA_MIN_WRITE_SNIPPET_CHARS", raising=False)
    _, rejected = validate_operations([_write_op(short)], _payload(excerpt), "mem")
    assert any("snippet too short" in r.reason for r in rejected)  # default 100 defers it
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "40")
    validated, rejected2 = validate_operations([_write_op(short)], _payload(excerpt), "mem")
    assert not any("snippet too short" in r.reason for r in rejected2)  # lowered floor admits it
    assert any(o.path == "mem/Elena's pottery class.md" for o in validated)
