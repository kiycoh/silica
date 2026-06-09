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


