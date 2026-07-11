# tests/test_undo_journal.py
import hashlib
from silica.kernel.ops import InverseOp, InverseOpKind
from silica.kernel.undo_journal import UndoJournalStore, revert_run


def _inv(path: str, content: str = "old") -> InverseOp:
    return InverseOp(kind=InverseOpKind.restore_version, path=path, prior_content=content)


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_start_record_and_lifo_read(tmp_path):
    store = UndoJournalStore(tmp_path / "j.db")
    run_id = store.start_run(source="inbox/meeting.md")
    assert isinstance(run_id, str) and run_id

    store.record(run_id, _inv("a.md"), post_hash="ha")
    store.record(run_id, _inv("b.md"), post_hash="hb")
    store.record(run_id, _inv("c.md"), post_hash="hc")

    entries = store.inverses_for(run_id)                 # LIFO: c, b, a
    assert [inv.path for inv, _ in entries] == ["c.md", "b.md", "a.md"]
    assert [h for _, h in entries] == ["hc", "hb", "ha"]


def test_last_active_run_ignores_reverted(tmp_path):
    store = UndoJournalStore(tmp_path / "j.db")
    r1 = store.start_run("one")
    store.record(r1, _inv("a.md"), post_hash="ha")
    r2 = store.start_run("two")
    store.record(r2, _inv("b.md"), post_hash="hb")
    assert store.last_active_run() == r2
    store.mark_reverted(r2)
    assert store.last_active_run() == r1
    store.mark_reverted(r1)
    assert store.last_active_run() is None


def test_write_over_existing_note_yields_restore_inverse(tmp_vault):
    """A write op whose path already holds a note must undo by RESTORING it,
    not deleting it — else /revert turns an accidental clobber into data loss."""
    from silica.tools.wrapped import build_txn
    from silica.kernel.ops import Op, OpType

    path = tmp_vault.note("Ideas/Note.md", "PRE-EXISTING body")
    op = Op(op=OpType.write, heading="Note", source_basename="s.md",
            path=path, hub="Hub", snippet="new body")
    invs = [i for i in build_txn([op]).inverses if i.path == path]
    assert len(invs) == 1
    assert invs[0].kind == InverseOpKind.restore_version
    assert invs[0].prior_content == "PRE-EXISTING body"


def test_write_new_note_yields_delete_inverse(tmp_vault):
    """A write to a genuinely new path still undoes by deletion (unchanged)."""
    import os
    from silica.tools.wrapped import build_txn
    from silica.kernel.ops import Op, OpType

    seed = tmp_vault.note("seed.md", "seed")            # materialise the vault
    new_path = os.path.join(os.path.dirname(seed), "Fresh.md")  # not created
    op = Op(op=OpType.write, heading="Fresh", source_basename="s.md",
            path=new_path, hub="Hub", snippet="body")
    invs = [i for i in build_txn([op]).inverses if i.path == new_path]
    assert len(invs) == 1
    assert invs[0].kind == InverseOpKind.delete_created


def test_corrupt_journal_is_quarantined_and_usable(tmp_path):
    """A corrupt db must not brick startup: quarantine it and start fresh."""
    dbpath = tmp_path / "j.db"
    dbpath.write_bytes(b"not a sqlite database at all -- garbage bytes")
    store = UndoJournalStore(dbpath)          # must not raise
    run_id = store.start_run("inbox/x.md")    # must be usable
    assert run_id
    assert dbpath.with_suffix(".corrupt").exists()


def test_revert_restores_unmodified_notes_and_skips_modified(tmp_vault, tmp_path):
    ada = tmp_vault.note("People/Ada.md", "PATCHED ada")
    grace = tmp_vault.note("People/Grace.md", "PATCHED grace")

    store = UndoJournalStore(tmp_path / "j.db")
    run_id = store.start_run("inbox/meeting.md")
    store.record(run_id, InverseOp(kind=InverseOpKind.restore_version, path=ada,
                                   prior_content="ORIGINAL ada"), post_hash=_h("PATCHED ada"))
    store.record(run_id, InverseOp(kind=InverseOpKind.restore_version, path=grace,
                                   prior_content="ORIGINAL grace"), post_hash=_h("PATCHED grace"))

    # Simulate a later refine on Grace -> its current hash no longer matches
    tmp_vault.write(grace, "REFINED grace")

    result = revert_run(run_id, store=store)

    assert tmp_vault.read(ada) == "ORIGINAL ada"
    assert tmp_vault.read(grace) == "REFINED grace"
    assert ada in result["reverted"]
    assert any(s["path"] == grace for s in result["skipped"])
    assert store.last_active_run() is None
