from unittest.mock import patch


from silica.kernel.validate import validate_operations


def test_write_rejects_sibling_directory_with_same_prefix(tmp_path):
    """A target_dir prefix match must not allow writes into sibling folders."""
    target_dir = tmp_path / "Dir"
    sibling_dir = tmp_path / "Directory"
    target_dir.mkdir()
    sibling_dir.mkdir()

    ops = [
        {
            "op": "write",
            "path": str(sibling_dir / "Bad.md"),
            "heading": "Bad",
            "source_basename": "inbox.md",
        }
    ]

    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=RuntimeError("not found")):
        validated, rejected = validate_operations(ops, [], str(target_dir))

    assert not validated
    assert len(rejected) == 1
    assert "not in target folder" in rejected[0].reason


def test_any_inbox_subfolder_is_forbidden_target(tmp_path):
    """The inbox guard must be vault-wide, not per-run-folder: a patch aimed at
    a *different* Inbox subfolder than the run's own previously slipped through
    validate and reached WRITE (2026-07-17 nucleate run, the SVM book note)."""
    target_dir = tmp_path / "Informatica"
    target_dir.mkdir()

    ops = [
        {
            "op": "patch",
            "path": "Inbox/svm-book/01-intro.md",
            "heading": "SVM",
            "source_basename": "Lezione 1.md",
            "snippet": "x" * 120,
        }
    ]

    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=RuntimeError("not found")):
        validated, rejected = validate_operations(ops, [], str(target_dir))

    assert not validated
    assert len(rejected) == 1
    assert "forbidden inbox segment" in rejected[0].reason


def test_skip_ops_are_never_rejected(tmp_path):
    """A skip op is a no-op — it must not be counted as a rejection even when
    its path is forbidden. Dedup/axis demotion turn ops into skips with their
    original (possibly Inbox) path intact; rejecting those inflated the
    rejection rate past 100% (2026-07-17 nucleate run: 150%/200% readouts)."""
    target_dir = tmp_path / "Informatica"
    target_dir.mkdir()

    ops = [
        {"op": "skip", "path": "Inbox/svm-book/01-intro.md",
         "heading": "SVM", "source_basename": "Lezione 1.md",
         "reason": "Duplicate operation to the same path"},
    ]

    validated, rejected = validate_operations(ops, [], str(target_dir))

    assert rejected == []
    assert validated == []


def test_validate_note_downgrades_size_limits_to_warnings():
    """validate_note must place max_lines and max_chars limit violations in warnings instead of errors."""
    from silica.kernel.linter import validate_note
    from unittest.mock import MagicMock
    
    # Create content that exceeds max_lines (400) and max_chars (20000)
    # 401 lines of text, with enough characters to exceed 20k
    lines = ["This is line {} with extra text to ensure we exceed twenty thousand characters overall".format(i) for i in range(450)]
    long_content = "---\ntitle: Long Note\ntype: concept\n---\n\n" + "\n".join(lines)
    
    class FakeNoteContent:
        content = long_content
        
    read_mock = MagicMock(return_value=FakeNoteContent())
    
    with patch("silica.kernel.linter.DRIVER.read_note", read_mock):
        errors, warnings = validate_note("some_path.md", hub=None)
        
    # Verify size violations are warnings, not errors
    size_warnings = [w for w in warnings if "too long" in w or "too large" in w]
    assert len(size_warnings) == 2
    size_errors = [e for e in errors if "too long" in e or "too large" in e]
    assert len(size_errors) == 0


def test_parse_ops_salvages_invalid_op_type():
    """One invalid op enum from the non-structured fallback must not raise —
    it killed a whole multi-file run (2026-07-17, FSM error at Lezione 4).
    The bad item degrades to a skip; valid siblings survive untouched."""
    from silica.kernel.ops_io import parse_ops
    from silica.kernel.ops import OpType

    ops = parse_ops([
        {"op": "update", "heading": "H", "source_basename": "s.md", "path": "A/B.md"},
        {"op": "write", "heading": "K", "source_basename": "s.md",
         "path": "A/C.md", "snippet": "x" * 120},
        "not even a dict",
    ])

    assert [o.op for o in ops] == [OpType.skip, OpType.write]
    assert ops[0].path == "A/B.md"
    assert "salvaged" in (ops[0].reason or "")


def test_heading_matches_payload_concept_modulo_case_and_apostrophe(tmp_path):
    """The distiller re-cases concept names and swaps typographic for straight
    apostrophes ('Storia dell’AI' vs "storia dell'AI") — a byte-exact
    heading check rejected 3 ops in the 2026-07-17 nucleate run. Normalized
    unique matches must remap to the canonical payload name instead."""
    target_dir = tmp_path / "Informatica"
    target_dir.mkdir()

    payloads = [{"batches": [{
        "inbox_file": "Inbox/machine_learning/Lezione 2.md",
        "concepts": [
            {"name": "storia dell'AI", "inbox_excerpt": "x" * 200},
            {"name": "Metodi Kernel", "inbox_excerpt": "y" * 200},
        ],
    }]}]

    ops = [{
        "op": "write",
        "path": f"{target_dir}/Storia dell'AI.md",
        "heading": "Storia dell’AI",
        "source_basename": "Lezione 2.md",
        "snippet": "z" * 150,
    }]

    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=RuntimeError("not found")), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, payloads, str(target_dir))

    assert rejected == []
    # (validated may also hold an auto-created hub op — match on heading)
    assert "storia dell'AI" in [o.heading for o in validated]   # canonical payload name


_HUB_NOTE = """---
related:
  - "[[Machine Learning]]"
last modified: 2026-07-17
AI: true
---

# Titolo

Corpo della nota con [[Altra nota]].
"""


def _lint_note(content, path, hub, op_type="patch"):
    from silica.kernel.linter import validate_note
    from unittest.mock import MagicMock

    class FakeNoteContent:
        pass

    FakeNoteContent.content = content
    with patch("silica.kernel.linter.DRIVER.read_note", MagicMock(return_value=FakeNoteContent())):
        return validate_note(path, hub=hub, op_type=op_type)


def test_hub_wikilink_check_is_case_insensitive():
    """Obsidian resolves [[Machine Learning]] and [[Machine learning]] to the
    same note — lint must not fail a patch because the existing link differs in
    case (real incident: 2026-07-17, L'apprendimento.md deferred over it)."""
    errors, _ = _lint_note(_HUB_NOTE, "Psicologia/L'apprendimento.md", hub="Machine learning")
    assert not any("Missing wikilink" in e for e in errors)


def test_hub_note_itself_needs_no_self_link():
    """Patching the hub note must not demand a [[hub]] self-link."""
    content = _HUB_NOTE.replace('  - "[[Machine Learning]]"', '  - "[[Informatica]]"')
    errors, _ = _lint_note(
        content, "Informatica/Intelligenza artificiale/Machine learning/Machine learning.md",
        hub="Machine learning",
    )
    assert not any("Missing wikilink" in e for e in errors)


def test_hub_wikilink_still_required_when_absent():
    content = _HUB_NOTE.replace('  - "[[Machine Learning]]"', '  - "[[Informatica]]"')
    errors, _ = _lint_note(content, "Psicologia/L'apprendimento.md", hub="Machine learning")
    assert any("Missing wikilink to [[Machine learning]]" in e for e in errors)
