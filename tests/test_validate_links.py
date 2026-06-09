"""Prospective link check in validate_operations (section 4).

Rules under test:
- write ops are exempt: their wikilinks may be forward references.
- patch/overwrite ops: every wikilink in snippet/content must resolve in the
  current vault OR point to a note being created by a write op in the same batch.
- Links already in the vault (search_names returns a match) are allowed.
"""
from unittest.mock import patch, MagicMock
import pytest

from silica.driver.base import NoteContent, NoteRef
from silica.kernel.validate import validate_operations


def _note(name: str, path: str) -> NoteRef:
    return NoteRef(name=name, path=path)


def _existing_read(path: str) -> NoteContent:
    return NoteContent(ref=NoteRef(name=path, path=path), content="# Note", size=6)


# ---------------------------------------------------------------------------
# Helpers for composing mocks
# ---------------------------------------------------------------------------

class _ReadSideEffect:
    """DRIVER.read_note mock: succeeds for known paths, raises for others."""

    def __init__(self, known: set[str]):
        self._known = known

    def __call__(self, ref):
        key = ref if isinstance(ref, str) else ref.path
        if any(k in str(key) for k in self._known):
            return _existing_read(str(key))
        raise RuntimeError(f"not found: {key}")


# ---------------------------------------------------------------------------
# Test: write ops are exempt from the link check
# ---------------------------------------------------------------------------

def test_write_op_forward_reference_non_exempt_rejected():
    """write ops with unresolvable links must be rejected now."""
    ops = [
        {
            "op": "write",
            "path": "Concepts/Neural Network.md",
            "heading": "Neural Network",
            "source_basename": "inbox.md",
            "snippet": "See [[Vanishing Gradient]] and [[Backpropagation]].",
        }
    ]
    read_mock = _ReadSideEffect(known=set())  # nothing exists
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert len(rejected) == 1
    assert "Vanishing Gradient" in rejected[0].reason
    assert not any(op.get("heading") == "Neural Network" for op in validated)


# ---------------------------------------------------------------------------
# Test: patch op — link resolved in vault
# ---------------------------------------------------------------------------

def test_patch_op_link_resolved_in_vault():
    """patch op whose snippet links to an existing vault note must be accepted."""
    existing_ref = _note("Quantum Computing", "Concepts/Quantum Computing.md")
    read_mock = _ReadSideEffect(known={"Classical Computing", "Quantum Computing"})

    ops = [
        {
            "op": "patch",
            "path": "Concepts/Classical Computing.md",
            "heading": "Classical Computing",
            "source_basename": "inbox.md",
            "snippet": "Compare with [[Quantum Computing]].",
        }
    ]
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[existing_ref]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert not rejected
    assert any(op["heading"] == "Classical Computing" for op in validated)


# ---------------------------------------------------------------------------
# Test: patch op — broken link rejected
# ---------------------------------------------------------------------------

def test_patch_op_broken_link_rejected():
    """patch op that introduces a wikilink not in the vault or the batch must be rejected."""
    read_mock = _ReadSideEffect(known={"Existing Note"})

    ops = [
        {
            "op": "patch",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "Links to [[Ghost Note That Does Not Exist]].",
        }
    ]
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert len(rejected) == 1
    assert "Ghost Note That Does Not Exist" in rejected[0].reason
    assert not any(op.get("heading") == "Existing Note" for op in validated)


# ---------------------------------------------------------------------------
# Test: patch op — link resolved by write op in same batch
# ---------------------------------------------------------------------------

def test_patch_op_link_resolved_by_same_batch_write():
    """patch op linking to a note being created in the same batch must be accepted."""
    read_mock = _ReadSideEffect(known={"Existing Note"})

    ops = [
        {
            "op": "write",
            "path": "Concepts/New Concept.md",
            "heading": "New Concept",
            "source_basename": "inbox.md",
            "snippet": "A new concept.",
        },
        {
            "op": "patch",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "See also [[New Concept]].",
        },
    ]
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert not rejected
    assert any(op["heading"] == "Existing Note" for op in validated)
    assert any(op["heading"] == "New Concept" for op in validated)


# ---------------------------------------------------------------------------
# Test: overwrite op — broken link rejected
# ---------------------------------------------------------------------------

def test_overwrite_op_broken_link_rejected():
    """overwrite ops obey the same rule as patch ops."""
    read_mock = _ReadSideEffect(known={"Existing Note"})

    ops = [
        {
            "op": "overwrite",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "content": "Full rewrite with [[Ghost Link]].",
        }
    ]
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert len(rejected) == 1
    assert "Ghost Link" in rejected[0].reason


# ---------------------------------------------------------------------------
# Test: patch op with no snippet — skips the check
# ---------------------------------------------------------------------------

def test_patch_op_no_snippet_skips_check():
    """patch op with empty snippet must not be rejected by the link check."""
    read_mock = _ReadSideEffect(known={"Existing Note"})

    ops = [
        {
            "op": "patch",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "",
        }
    ]
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert not rejected
    assert any(op["heading"] == "Existing Note" for op in validated)


# ---------------------------------------------------------------------------
# Test: auto-created hub note is in batch_created_names → link to it resolves
# ---------------------------------------------------------------------------

def test_patch_op_link_to_auto_hub_resolves():
    """patch op linking to the auto-created hub note for the target_dir must be accepted."""
    read_mock = _ReadSideEffect(known={"Existing Note"})  # hub does not exist yet

    ops = [
        {
            "op": "patch",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            # hub name == basename of target_dir == "Concepts"
            "snippet": "Parent: [[Concepts]].",
        }
    ]
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    # hub auto-op is a write whose path includes "Concepts.md",
    # so "concepts" enters batch_created_names and resolves the link.
    assert not rejected


# ---------------------------------------------------------------------------
# Test: write ops with links in future_ref_whitelist are accepted
# ---------------------------------------------------------------------------

def test_write_op_whitelist_link_accepted():
    """write ops with links in future_ref_whitelist must be accepted."""
    ops = [
        {
            "op": "write",
            "path": "Concepts/Neural Network.md",
            "heading": "Neural Network",
            "source_basename": "inbox.md",
            "snippet": "See [[Vanishing Gradient]] and [[Backpropagation]].",
        }
    ]
    read_mock = _ReadSideEffect(known=set())
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(
            ops, [], "Concepts", future_ref_whitelist=["Vanishing Gradient", "Backpropagation"]
        )

    assert not rejected
    assert any(op.get("heading") == "Neural Network" for op in validated)


# ---------------------------------------------------------------------------
# Test: deduplication happens before coercion, avoiding incorrect reject
# ---------------------------------------------------------------------------

def test_deduplication_before_coercion_prevents_unnecessary_rejection():
    """Duplicate ops for an existing file should not fail validation after coercion.
    
    If two write/patch ops to the same existing path are processed:
    1. Deduplication skips the redundant op, keeping the richest one as 'write' (if write/overwrite was in group).
    2. Coercion converts it to 'patch' because the path exists.
    3. It is validated successfully rather than rejected as a duplicate write error.
    """
    ops = [
        {
            "op": "write",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "A short snippet.",
        },
        {
            "op": "write",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "A much longer and richer snippet with more content.",
        }
    ]
    # Path exists in vault
    read_mock = _ReadSideEffect(known={"Existing Note"})
    
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert not rejected
    # The list contains the auto-created hub note operation first, then the deduplicated note operation.
    assert len(validated) == 2
    assert validated[0].heading == "Concepts"
    assert validated[1].op == "patch"
    assert "much longer" in validated[1].snippet
