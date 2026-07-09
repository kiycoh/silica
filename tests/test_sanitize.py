"""Regression tests for silica/kernel/sanitize.py normalizers."""
from __future__ import annotations

import pytest
from silica.kernel.sanitize import _strip_md_ext, normalize_ops
from silica.kernel.templates import slugify


# ---------------------------------------------------------------------------
# _strip_md_ext — wikilink normalizer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    # Basic cases
    ("[[Note.md]]",                           "[[Note]]"),
    ("[[path/to/Note.md]]",                   "[[path/to/Note]]"),
    ("See [[ROS 2.md]] for details.",         "See [[ROS 2]] for details."),
    # Anchor preserved
    ("[[Note.md#section]]",                   "[[Note#section]]"),
    # Alias preserved
    ("[[Note.md|alias]]",                     "[[Note|alias]]"),
    # Anchor + alias
    ("[[Note.md#s|alias]]",                   "[[Note#s|alias]]"),
    # Already correct — untouched
    ("[[Note]]",                              "[[Note]]"),
    ("[[path/to/Note]]",                      "[[path/to/Note]]"),
    # Path with spaces
    ("[[Agenti Autonomi/Virtualizzazione dell'hardware.md]]",
     "[[Agenti Autonomi/Virtualizzazione dell'hardware]]"),
    # Multiple wikilinks in text
    ("See [[A.md]] and [[B.md]] here.",       "See [[A]] and [[B]] here."),
    # No wikilinks — untouched
    ("plain text without links",              "plain text without links"),
])
def test_strip_md_ext(raw, expected):
    assert _strip_md_ext(raw) == expected


def test_strip_md_ext_in_frontmatter_related():
    """Simulates the reported bug: frontmatter related array contains .md links."""
    raw = 'related:\n  - "[[Agenti Autonomi/ROS 2.md]]"\n  - "[[HAL.md]]"'
    result = _strip_md_ext(raw)
    assert "ROS 2.md" not in result
    assert "HAL.md" not in result
    assert "[[Agenti Autonomi/ROS 2]]" in result
    assert "[[HAL]]" in result


# ---------------------------------------------------------------------------
# normalize_ops — full op normalizer
# ---------------------------------------------------------------------------

def test_normalize_ops_strips_md_from_snippet():
    ops = [{"op": "patch", "path": "notes/A.md", "heading": "A",
            "source_basename": "inbox.md",
            "snippet": "See [[ROS 2.md]] and [[HAL.md]] for details."}]
    result = normalize_ops(ops)
    assert "[[ROS 2]]" in result[0]["snippet"]
    assert ".md]]" not in result[0]["snippet"]


def test_normalize_ops_strips_md_from_content():
    ops = [{"op": "write", "path": "notes/B.md", "heading": "B",
            "source_basename": "inbox.md",
            "content": "# B\n\nSee [[B.md]] (self!) and [[C.md]]."}]
    result = normalize_ops(ops)
    assert "[[B]]" in result[0]["content"]
    assert "[[C]]" in result[0]["content"]
    assert ".md]]" not in result[0]["content"]


def test_normalize_ops_strips_md_from_related():
    ops = [{"op": "write", "path": "notes/D.md", "heading": "D",
            "source_basename": "inbox.md",
            "related": ["[[Inbox/lezione_15.md]]", "[[Concepts/E.md]]", "[[NoExt]]"]}]
    result = normalize_ops(ops)
    assert result[0]["related"] == ["[[Inbox/lezione_15]]", "[[Concepts/E]]", "[[NoExt]]"]


def test_normalize_ops_preserves_non_md_links():
    ops = [{"op": "patch", "path": "notes/F.md", "heading": "F",
            "source_basename": "inbox.md",
            "snippet": "See [[ROS 2]] and [[HAL]] (already correct)."}]
    result = normalize_ops(ops)
    assert result[0]["snippet"] == "See [[ROS 2]] and [[HAL]] (already correct)."


def test_normalize_ops_skip_ops_pass_through():
    ops = [{"op": "skip", "heading": "G", "source_basename": "inbox.md",
            "reason": "already covered"}]
    result = normalize_ops(ops)
    assert result[0]["op"] == "skip"


def test_normalize_ops_empty_list():
    assert normalize_ops([]) == []


def test_normalize_ops_non_list_passthrough():
    assert normalize_ops("not a list") == "not a list"  # type: ignore[arg-type]


def test_normalize_ops_sanitizes_literal_newlines():
    ops = [{
        "op": "write",
        "path": "notes/B.md",
        "heading": "B",
        "source_basename": "inbox.md",
        "snippet": "Prose with a literal \\n here. ```python\nprint('code with \\n preserved')\n``` More prose with \\n."
    }]
    result = normalize_ops(ops)
    expected_snippet = "Prose with a literal \n here. ```python\nprint('code with \\n preserved')\n``` More prose with \n."
    assert result[0]["snippet"] == expected_snippet


def test_normalize_ops_preserves_backslash_commands_in_math():
    # Prose \n must still become a newline, but math spans must be left alone so
    # `\nabla`/`\neq` are not shredded into newlines (root cause of defect 3).
    ops = [{
        "op": "write",
        "path": "notes/B.md",
        "heading": "B",
        "source_basename": "inbox.md",
        "snippet": "Prosa con \\n qui e math $\\nabla f$ e blocco $$x \\neq y$$ fine.",
    }]
    result = normalize_ops(ops)
    snip = result[0]["snippet"]
    assert "Prosa con \n qui" in snip          # prose \n -> real newline
    assert "$\\nabla f$" in snip               # inline math untouched
    assert "$$x \\neq y$$" in snip             # block math untouched


def test_normalize_ops_removes_trailing_literal_newlines():
    ops = [{
        "op": "write",
        "path": "notes/B.md",
        "heading": "B",
        "source_basename": "inbox.md",
        "snippet": "Some text.\\n"
    }]
    result = normalize_ops(ops)
    assert result[0]["snippet"] == "Some text."


def test_slugify_normalizes_whitespace():
    assert slugify("Performance\nElement") == "Performance Element"
    assert slugify("Performance\r\nElement  negli   agenti") == "Performance Element negli agenti"


# ---------------------------------------------------------------------------
# strip_degenerate_runs — collapses 5+ identical consecutive chars to 1
# ---------------------------------------------------------------------------

from silica.kernel.sanitize import strip_degenerate_runs


def test_strip_degenerate_slash_run():
    assert strip_degenerate_runs("/////") == "/"


def test_strip_degenerate_alpha_run():
    assert strip_degenerate_runs("aaaaa") == "a"


def test_strip_mixed_text_with_run():
    assert strip_degenerate_runs("some ///// text") == "some / text"


def test_strip_does_not_remove_line():
    result = strip_degenerate_runs("/////")
    assert result != ""


def test_strip_run_of_exactly_4_unchanged():
    assert strip_degenerate_runs("////") == "////"


def test_strip_multiple_different_runs():
    assert strip_degenerate_runs("aaaaa bbbbb") == "a b"


def test_strip_run_in_middle_of_word():
    assert strip_degenerate_runs("hellooooo world") == "hello world"


def test_strip_newline_not_collapsed():
    text = "line1\nline2"
    assert strip_degenerate_runs(text) == "line1\nline2"


def test_strip_preserves_markdown_structural_runs():
    # Markdown-structural chars legitimately repeat — never collapse them.
    assert strip_degenerate_runs("##### Prima versione") == "##### Prima versione"
    assert strip_degenerate_runs("###### H6") == "###### H6"
    assert strip_degenerate_runs("-----") == "-----"   # thematic break / setext
    assert strip_degenerate_runs("=====") == "====="   # setext H1 underline
    assert strip_degenerate_runs("*****") == "*****"
    assert strip_degenerate_runs("~~~~~") == "~~~~~"    # fence
    # but real garbage of non-structural chars still collapses
    assert strip_degenerate_runs("!!!!!") == "!"


def test_strip_degenerate_normalized_in_ops():
    ops = [{"op": "write", "path": "Dir/A.md", "heading": "A",
            "source_basename": "inbox.md",
            "content": "Noise: /////\nReal content here."}]
    result = normalize_ops(ops)
    assert "/" in result[0]["content"]
    assert "/////" not in result[0]["content"]


# ---------------------------------------------------------------------------
# collapse_nested_wikilinks — [[[[X]]]] → [[X]]
# ---------------------------------------------------------------------------

from silica.kernel.sanitize import collapse_nested_wikilinks


def test_collapse_quadruple_brackets():
    assert collapse_nested_wikilinks("see [[[[Reti (DL)]]]] here") == "see [[Reti (DL)]] here"


def test_collapse_triple_brackets():
    assert collapse_nested_wikilinks("[[[X]]]") == "[[X]]"


def test_collapse_leaves_valid_wikilink_untouched():
    assert collapse_nested_wikilinks("a [[X]] and x[[1]] code") == "a [[X]] and x[[1]] code"


def test_collapse_leaves_single_brackets_untouched():
    assert collapse_nested_wikilinks("[label](url) and [^1]") == "[label](url) and [^1]"


def test_collapse_nested_wikilinks_normalized_in_ops_body():
    ops = [{"op": "write", "path": "Dir/A.md", "heading": "A",
            "source_basename": "inbox.md",
            "content": "Vedi [[[[Reti Neurali Profonde (Deep Learning)]]]] nel testo.",
            "snippet": "Collega a [[[[IA Generativa]]]]."}]
    result = normalize_ops(ops)
    assert "[[[[" not in result[0]["content"] and "]]]]" not in result[0]["content"]
    assert "[[Reti Neurali Profonde (Deep Learning)]]" in result[0]["content"]
    assert "[[IA Generativa]]" in result[0]["snippet"]


# ---------------------------------------------------------------------------
# External body appendix — bodies carried OUTSIDE the JSON string so the model
# never hand-escapes backslashes (root fix for distiller LaTeX corruption).
# ---------------------------------------------------------------------------

from silica.kernel.sanitize import parse_json


def test_parse_json_resolves_external_body_into_snippet():
    raw = (
        '{"updates":[{"op":"write","path":"X.md","snippet_ref":1}]}\n'
        '\n'
        '===SILICA-BODY 1===\n'
        'corpo della nota'
    )
    parsed, _ = parse_json(raw)
    op = parsed["updates"][0]
    assert op["snippet"] == "corpo della nota"
    assert "snippet_ref" not in op


def test_parse_json_external_body_preserves_backslashes_verbatim():
    # THE regression: \top must NOT decode to TAB, \neq must NOT decode to
    # newline, matrix \\ must NOT double. Body lives outside JSON → no decoding.
    body = r"$A \in \mathbb{R}^{n}$, $A^\top$, $x \neq 0$, riga \\ matrice"
    raw = (
        '{"updates":[{"op":"write","path":"X.md","snippet_ref":1}]}\n'
        '\n'
        '===SILICA-BODY 1===\n'
        + body
    )
    parsed, _ = parse_json(raw)
    assert parsed["updates"][0]["snippet"] == body
    assert "\t" not in parsed["updates"][0]["snippet"]


def test_parse_json_multiple_external_bodies_map_by_ref():
    raw = (
        '{"updates":['
        '{"op":"write","path":"A.md","snippet_ref":1},'
        '{"op":"skip","reason":"noise"},'
        '{"op":"patch","path":"B.md","content_ref":2}'
        ']}\n'
        '\n'
        '===SILICA-BODY 1===\n'
        'primo $\\alpha$\n'
        '===SILICA-BODY 2===\n'
        'secondo $\\beta$'
    )
    parsed, _ = parse_json(raw)
    ups = parsed["updates"]
    assert ups[0]["snippet"] == r"primo $\alpha$"
    assert ups[2]["content"] == r"secondo $\beta$"
    assert "content_ref" not in ups[2]


def test_parse_json_no_appendix_is_backward_compatible():
    raw = '{"updates":[{"op":"write","path":"X.md","snippet":"vecchio stile"}]}'
    parsed, clean = parse_json(raw)
    assert parsed["updates"][0]["snippet"] == "vecchio stile"
    assert clean is True


def test_parse_json_top_level_list_of_ops_with_bodies():
    raw = (
        '[{"op":"write","path":"X.md","snippet_ref":1}]\n'
        '===SILICA-BODY 1===\n'
        'corpo'
    )
    parsed, _ = parse_json(raw)
    assert parsed[0]["snippet"] == "corpo"

