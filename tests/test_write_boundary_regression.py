"""Regression tests for the 2026-07-08 merge-run vault corruption.

Two independent defects compounded:
1. Model-supplied Op paths missing the `.md` extension reached the driver
   verbatim; the cli backend's verbatim-write fallback then fabricated
   extensionless phantom files (202 in one night) and every post-write
   verify failed, aborting all 52 merges.
2. Model-emitted `\\u0000` JSON escapes decoded to real NUL bytes in note
   content; NUL in a subprocess argv raises "embedded null byte", and the
   same fallback persisted the NULs into the vault.
"""
from silica.kernel.ops import Op, OpType


def test_op_path_gets_md_extension():
    op = Op(op=OpType.overwrite, path="Folder/Nota", content="x", heading="h", source_basename="s.pdf")
    assert op.path == "Folder/Nota.md"


def test_op_move_paths_get_md_extension():
    op = Op(op=OpType.move, from_path="A/Uno", to_path="B/Due", heading="h", source_basename="s.pdf")
    assert op.from_path == "A/Uno.md"
    assert op.to_path == "B/Due.md"


def test_op_path_with_md_untouched():
    op = Op(op=OpType.delete, path="Folder/Nota.md", heading="h", source_basename="s.pdf")
    assert op.path == "Folder/Nota.md"


def test_op_content_nul_bytes_stripped():
    op = Op(op=OpType.overwrite, path="N.md", content="($\x00lambda \x00left)", heading="h", source_basename="s.pdf")
    assert "\x00" not in op.content
    assert op.content == "($lambda left)"


def test_op_snippet_nul_bytes_stripped():
    op = Op(op=OpType.patch, path="N.md", snippet="a\x00b", heading="H", source_basename="s.pdf")
    assert op.snippet == "ab"
