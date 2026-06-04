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
    r2 = store.start_run("two")
    assert store.last_active_run() == r2
    store.mark_reverted(r2)
    assert store.last_active_run() == r1
    store.mark_reverted(r1)
    assert store.last_active_run() is None


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
