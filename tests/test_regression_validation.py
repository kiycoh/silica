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
