#!/usr/bin/env python3
"""Regression tests: golden notes calibrate the linter's threshold.

These three fixtures are the *corpus di calibrazione*: they embody the real
note style and contain two latent bugs the pipeline previously silenced.
Any future rule tightening that would break a legitimate pattern (H1 not on
line 1, callout types in UPPERCASE, date+time suffix, CSV-scalar tags,
cross-domain linking via any of parent/related/body) will fail here.

The golden notes NEVER enter runtime context — they shape the CODE at
design-time and guard it at test-time.
"""

# --- hermes_common bootstrap ---
import os, sys
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "hermes_common")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
# --- end bootstrap ---

import unittest
from pathlib import Path

from hermes_common.ofm import ofm_lint
from hermes_common.frontmatter import split, normalize_tags, clean_tag, _ensure_tag_list

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


# ===================================================================
# Core invariant: ZERO violations on all three golden notes
# ===================================================================

class TestGoldenNotesZeroViolations(unittest.TestCase):
    """The cardinal rule: golden notes must NEVER trigger violations.

    If a new check causes a violation on a well-formed real note, the check
    is miscalibrated, not the note.
    """

    def test_connessionismo_zero_violations(self):
        r = ofm_lint(_read("connessionismo_ia.md"), stem="Connessionismo (IA)")
        self.assertEqual(r["violations"], [],
                         f"Connessionismo violations should be empty: {r['violations']}")

    def test_sistema_esperto_zero_violations(self):
        r = ofm_lint(_read("sistema_esperto.md"), stem="Sistema Esperto")
        self.assertEqual(r["violations"], [],
                         f"Sistema Esperto violations should be empty: {r['violations']}")

    def test_krr_zero_violations(self):
        r = ofm_lint(_read("krr.md"), stem="Knowledge Representation and Reasoning")
        self.assertEqual(r["violations"], [],
                         f"KRR violations should be empty: {r['violations']}")


# ===================================================================
# Flag calibration: exact expected flags per golden note
# ===================================================================

class TestGoldenNoteFlags(unittest.TestCase):
    """Assert the exact flags each golden note should produce.

    This is the calibration contract: future rule changes that suppress
    a valid warning or introduce a false positive will fail.
    """

    def test_connessionismo_flags_inline_csv(self):
        """Bug #1: tags is a YAML scalar with commas, not a list."""
        r = ofm_lint(_read("connessionismo_ia.md"), stem="Connessionismo (IA)")
        csv_flags = [f for f in r["flags"] if "inline-CSV" in f]
        self.assertEqual(len(csv_flags), 1,
                         f"Expected exactly 1 inline-CSV flag, got: {r['flags']}")

    def test_sistema_esperto_flags_tags_empty(self):
        """Bug #2: tags key is present but empty → not indexable."""
        r = ofm_lint(_read("sistema_esperto.md"), stem="Sistema Esperto")
        empty_flags = [f for f in r["flags"] if "tags empty" in f]
        self.assertEqual(len(empty_flags), 1,
                         f"Expected 'tags empty' flag, got: {r['flags']}")

    def test_krr_clean(self):
        """KRR is well-formed: zero violations and zero flags."""
        r = ofm_lint(_read("krr.md"), stem="Knowledge Representation and Reasoning")
        self.assertEqual(r["violations"], [])
        self.assertEqual(r["flags"], [],
                         f"KRR should be completely clean, got flags: {r['flags']}")


# ===================================================================
# Calibration guardrails: patterns that must NOT trigger violations
# ===================================================================

class TestCalibrationGuardrails(unittest.TestCase):
    """Protect against over-strict rules that would reject valid patterns."""

    def test_h1_not_on_first_line_is_ok(self):
        """Connessionismo opens with a callout, then table, then H1. Must pass."""
        content = _read("connessionismo_ia.md")
        lines = content.split("\n")
        # Find the H1 line
        h1_lines = [i for i, l in enumerate(lines) if l.startswith("# ")]
        self.assertTrue(len(h1_lines) > 0, "Fixture must have an H1")
        # H1 is NOT on line 0 (after frontmatter)
        r = ofm_lint(content, stem="Connessionismo (IA)")
        self.assertNotIn("no H1 heading", r["flags"])

    def test_uppercase_callout_types_pass(self):
        """[!WARNING] and [!IMPORTANT] in uppercase must not trigger violation."""
        content = _read("connessionismo_ia.md")
        r = ofm_lint(content, stem="Connessionismo (IA)")
        callout_violations = [v for v in r["violations"] if "callout" in v.lower()]
        self.assertEqual(callout_violations, [],
                         f"Uppercase callout types should pass: {callout_violations}")

    def test_date_with_time_suffix_no_violation(self):
        """Sistema Esperto has '2026, 03, 17 18:3:43' — time suffix must be tolerated."""
        content = _read("sistema_esperto.md")
        r = ofm_lint(content, stem="Sistema Esperto")
        date_violations = [v for v in r["violations"] if "last modified" in v]
        self.assertEqual(date_violations, [],
                         f"Date with time suffix should not violate: {date_violations}")

    def test_no_parent_note_with_related_not_orphan(self):
        """KRR has no parent note but has related + body links → not orphan."""
        r = ofm_lint(_read("krr.md"), stem="Knowledge Representation and Reasoning")
        orphan_flags = [f for f in r["flags"] if "orphan" in f]
        self.assertEqual(orphan_flags, [],
                         f"KRR has related links, should not be flagged orphan: {orphan_flags}")

    def test_h1_text_does_not_need_to_match_filename(self):
        """'# Definizione di Connessionismo (IA)' ≠ stem 'Connessionismo (IA)'. Must pass."""
        r = ofm_lint(_read("connessionismo_ia.md"), stem="Connessionismo (IA)")
        title_violations = [v for v in r["violations"] if "title" in v.lower() or "H1" in v]
        self.assertEqual(title_violations, [],
                         "H1 text should not need to match filename")


# ===================================================================
# Upstream fix: normalize_tags must be comma-aware
# ===================================================================

class TestNormalizeTagsCommaAware(unittest.TestCase):
    """The inline-CSV scalar must be split into individual tags, not fused."""

    def test_csv_scalar_splits(self):
        data = {"tags": "connessionismo-ia, intelligenza-artificiale, reti-neurali"}
        result = normalize_tags(data)
        self.assertEqual(result["tags"], ["connessionismo-ia", "intelligenza-artificiale", "reti-neurali"])

    def test_csv_scalar_no_fusion(self):
        """The old bug: commas removed, spaces→hyphens → one monster tag."""
        data = {"tags": "a, b, c"}
        result = normalize_tags(data)
        self.assertNotIn("a-b-c", result["tags"],
                         "CSV scalar must NOT be fused into a single tag")
        self.assertEqual(result["tags"], ["a", "b", "c"])

    def test_normal_list_unchanged(self):
        data = {"tags": ["alpha", "beta"]}
        result = normalize_tags(data)
        self.assertEqual(result["tags"], ["alpha", "beta"])

    def test_empty_tags_yields_empty_list(self):
        data = {"tags": None}
        result = normalize_tags(data)
        self.assertEqual(result["tags"], [])

    def test_ensure_tag_list_csv(self):
        self.assertEqual(_ensure_tag_list("a, b, c"), ["a", "b", "c"])

    def test_ensure_tag_list_single_string(self):
        self.assertEqual(_ensure_tag_list("solo-tag"), ["solo-tag"])

    def test_ensure_tag_list_proper_list(self):
        self.assertEqual(_ensure_tag_list(["x", "y"]), ["x", "y"])

    def test_ensure_tag_list_empty(self):
        self.assertEqual(_ensure_tag_list(None), [])
        self.assertEqual(_ensure_tag_list(""), [])
        self.assertEqual(_ensure_tag_list([]), [])


# ===================================================================
# OFM structural integrity: synthetic edge cases
# ===================================================================

class TestOFMLintStructural(unittest.TestCase):
    """Synthetic notes to verify balanced-delimiter and heading checks."""

    def _note(self, body, **fm_fields):
        import yaml
        fm = {"tags": ["test"], "AI": True, "last modified": "2026, 01, 01",
              "related": ['"[[Test]]"']}
        fm.update(fm_fields)
        raw = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()
        return f"---\n{raw}\n---\n\n{body}"

    def test_unbalanced_math_block(self):
        r = ofm_lint(self._note("# T\n\n$$\nx\n"), stem="T")
        self.assertIn("unbalanced $$ block", r["violations"])

    def test_balanced_math_block(self):
        r = ofm_lint(self._note("# T\n\n$$\nx\n$$\n"), stem="T")
        self.assertNotIn("unbalanced $$ block", r["violations"])

    def test_unclosed_fence(self):
        r = ofm_lint(self._note("# T\n\n```python\nprint('x')\n"), stem="T")
        self.assertIn("unclosed code fence", r["violations"])

    def test_unknown_callout_type(self):
        r = ofm_lint(self._note("# T\n\n> [!banana] Weird\n> Content"), stem="T")
        self.assertIn("unknown callout type [!banana]", r["violations"])

    def test_known_callout_case_insensitive(self):
        r = ofm_lint(self._note("# T\n\n> [!WARNING] Attenzione\n> Testo"), stem="T")
        callout_v = [v for v in r["violations"] if "callout" in v]
        self.assertEqual(callout_v, [])

    def test_heading_level_jump(self):
        r = ofm_lint(self._note("# T\n\n### Skipped H2\n"), stem="T")
        jump_flags = [f for f in r["flags"] if "heading level jump" in f]
        self.assertTrue(len(jump_flags) > 0,
                        f"Expected heading level jump flag, got: {r['flags']}")

    def test_no_h1_flag(self):
        r = ofm_lint(self._note("## Only H2\n\nContent"), stem="T")
        self.assertIn("no H1 heading", r["flags"])

    def test_missing_ai_field(self):
        r = ofm_lint(self._note("# T\n\nBody", AI="maybe"), stem="T")
        self.assertIn("frontmatter 'AI' missing or not boolean", r["violations"])

    def test_unbalanced_wikilink(self):
        r = ofm_lint(self._note("# T\n\n[[Broken link"), stem="T")
        self.assertIn("unbalanced [[wikilink]]", r["violations"])

    def test_unbalanced_highlight(self):
        r = ofm_lint(self._note("# T\n\nSome ==highlighted text"), stem="T")
        self.assertIn("unbalanced == highlight", r["violations"])

    def test_inline_code_with_equals_no_false_positive(self):
        """Inline `if x == y` must NOT trigger 'unbalanced == highlight'."""
        r = ofm_lint(self._note("# T\n\nUse `if x == y` to compare."), stem="T")
        eq_violations = [v for v in r["violations"] if "==" in v]
        self.assertEqual(eq_violations, [],
                         f"Inline code == should not trigger: {eq_violations}")

    def test_inline_code_with_dollar_no_false_positive(self):
        """Inline `$$` in code span must not trigger unbalanced math block."""
        r = ofm_lint(self._note("# T\n\nThe delimiter `$$` starts a block."), stem="T")
        math_violations = [v for v in r["violations"] if "$$" in v]
        self.assertEqual(math_violations, [],
                         f"Inline code $$ should not trigger: {math_violations}")

    def test_inline_code_with_wikilink_no_false_positive(self):
        """Inline `[[` in code span must not trigger unbalanced wikilink."""
        r = ofm_lint(self._note("# T\n\nUse `[[note]]` syntax for links."), stem="T")
        wl_violations = [v for v in r["violations"] if "wikilink" in v]
        self.assertEqual(wl_violations, [],
                         f"Inline code [[ should not trigger: {wl_violations}")

    def test_literal_newline_sequence_in_body(self):
        """Literal \\n in body outside code blocks must trigger violation."""
        r = ofm_lint(self._note("# T\n\nSome text\\nwith literal newline"), stem="T")
        self.assertIn("literal '\\n' character sequence detected in body", r["violations"])

    def test_literal_newline_sequence_in_code_block_ok(self):
        """Literal \\n inside code blocks must NOT trigger violation."""
        r = ofm_lint(self._note("# T\n\n```python\nprint('hello\\nworld')\n```"), stem="T")
        self.assertNotIn("literal '\\n' character sequence detected in body", r["violations"])


if __name__ == "__main__":
    unittest.main()
