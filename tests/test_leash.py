"""Tests for the Leash capability envelope (silica/agent/leash.py)."""
from silica.agent.leash import (
    Leash,
    dedup_leash,
    refiner_leash,
    orphan_leash,
    make_no_info_loss_guard,
    _wikilinks,
)
from silica.kernel.ops import Op, OpType


def _op(op_type, path, *, heading="H", content=None, snippet=""):
    return Op(
        op=op_type,
        heading=heading,
        source_basename="inbox.md",
        path=path,
        content=content,
        snippet=snippet,
    )


# --- dedup leash -----------------------------------------------------------

def test_dedup_leash_allows_patch_on_larger():
    leash = dedup_leash("Concepts/Big Note.md")
    ops = [_op(OpType.patch, "Concepts/Big Note.md")]
    kept, rejected = leash.enforce(ops)
    assert len(kept) == 1
    assert not rejected


def test_dedup_leash_rejects_overwrite_and_delete_and_write():
    leash = dedup_leash("Concepts/Big Note.md")
    ops = [
        _op(OpType.overwrite, "Concepts/Big Note.md", content="x"),
        _op(OpType.delete, "Concepts/Big Note.md"),
        _op(OpType.write, "Concepts/New Note.md"),
    ]
    kept, rejected = leash.enforce(ops)
    assert kept == []
    assert len(rejected) == 3
    assert all("not permitted" in r["reason"] for r in rejected)


def test_dedup_leash_rejects_patch_on_other_note():
    leash = dedup_leash("Concepts/Big Note.md")
    ops = [_op(OpType.patch, "Concepts/Small Note.md")]
    kept, rejected = leash.enforce(ops)
    assert kept == []
    assert "outside leash" in rejected[0]["reason"]


def test_dedup_leash_never_touches_hub():
    leash = dedup_leash("Concepts/Big Note.md", hub="Concepts/Big Note.md")
    # Even though it is the "larger" path, being the hub makes it forbidden.
    ops = [_op(OpType.patch, "Concepts/Big Note.md")]
    kept, rejected = leash.enforce(ops)
    assert kept == []
    assert "outside leash" in rejected[0]["reason"]


# --- refiner leash ---------------------------------------------------------

def test_refiner_leash_allows_lossless_overwrite():
    original = "# Note\n\nSee [[Alpha]] and [[Beta]].\n" + ("body " * 100)
    leash = refiner_leash("Notes/Target.md")
    new = "# Note\n\n> [!note]\nSee [[Alpha]] and [[Beta]].\n" + ("body " * 100)
    ops = [_op(OpType.overwrite, "Notes/Target.md", content=new)]
    kept, rejected = leash.enforce(ops, read_note=lambda p: original)
    assert len(kept) == 1
    assert not rejected


def test_refiner_leash_rejects_dropped_wikilink():
    original = "See [[Alpha]] and [[Beta]]." + ("x" * 200)
    leash = refiner_leash("Notes/Target.md")
    new = "See [[Alpha]] only." + ("x" * 200)  # dropped [[Beta]]
    ops = [_op(OpType.overwrite, "Notes/Target.md", content=new)]
    kept, rejected = leash.enforce(ops, read_note=lambda p: original)
    assert kept == []
    assert "dropped wikilink" in rejected[0]["reason"]


def test_refiner_leash_rejects_shrink():
    original = "[[Alpha]]\n" + ("content " * 100)
    leash = refiner_leash("Notes/Target.md")
    new = "[[Alpha]]\nshort"
    ops = [_op(OpType.overwrite, "Notes/Target.md", content=new)]
    kept, rejected = leash.enforce(ops, read_note=lambda p: original)
    assert kept == []
    assert "shrank" in rejected[0]["reason"]


# --- orphan leash ----------------------------------------------------------

def test_orphan_leash_allows_patch_that_adds_link():
    leash = orphan_leash("Notes/Orphan.md")
    op = _op(OpType.patch, "Notes/Orphan.md", snippet="## Related\n\n- [[Neighbor]]\n")
    kept, rejected = leash.enforce(op_list := [op])
    assert len(kept) == 1 and not rejected


def test_orphan_leash_rejects_patch_without_link():
    leash = orphan_leash("Notes/Orphan.md")
    op = _op(OpType.patch, "Notes/Orphan.md", snippet="## Related\n\n(no links here)\n")
    kept, rejected = leash.enforce([op])
    assert kept == []
    assert "no wikilink" in rejected[0]["reason"]


def test_orphan_leash_rejects_overwrite_and_other_targets():
    leash = orphan_leash("Notes/Orphan.md")
    ops = [
        _op(OpType.overwrite, "Notes/Orphan.md", content="[[X]]"),
        _op(OpType.patch, "Notes/Other.md", snippet="[[X]]"),
    ]
    kept, rejected = leash.enforce(ops)
    assert kept == []
    assert len(rejected) == 2


# --- skip + helpers --------------------------------------------------------

def test_skip_ops_always_pass():
    leash = dedup_leash("Concepts/Big Note.md")
    skip = Op(op=OpType.skip, heading="H", source_basename="inbox.md", reason="noop")
    kept, rejected = leash.enforce([skip])
    assert kept == [skip]
    assert not rejected


def test_wikilinks_extraction_handles_alias_and_anchor():
    links = _wikilinks("[[Alpha|alias]] [[Beta#section]] [[Gamma]]")
    assert links == {"alpha", "beta", "gamma"}


def test_no_info_loss_guard_direct():
    guard = make_no_info_loss_guard(floor_ratio=0.85)
    op = Op(op=OpType.overwrite, heading="H", source_basename="i.md",
            path="N.md", content="[[A]] kept content here")
    assert guard(op, "[[A]] kept content here") is None
    assert "dropped wikilink" in guard(op, "[[A]] [[B]] original longer text here")
