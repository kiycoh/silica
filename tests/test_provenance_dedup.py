# tests/test_provenance_dedup.py
from silica.config import CONFIG
from silica.kernel.templates import provenance_header, block_present
from silica.kernel.ops import Op, OpType
from silica.kernel.bulk import execute_one
from silica.kernel.provenance import append_record, note_authored_by


def test_provenance_helpers():
    hdr = provenance_header("Async IO", "meeting.md")
    assert hdr == "## Note aggiuntive — Async IO (da meeting.md)"
    body = f"seed\n\n{hdr}\n\nfacts\n"
    assert block_present(body, "Async IO", "meeting.md") is True
    assert block_present(body, "Async IO", "other.md") is False
    assert block_present("seed only", "Async IO", "meeting.md") is False


def test_double_patch_is_idempotent(tmp_vault):
    target = tmp_vault.note("Topics/AsyncIO.md", "---\n---\nseed\n")
    op = Op(op=OpType.patch, heading="Async IO", source_basename="meeting.md",
            path=target, snippet="first fact", hub="Hub")

    execute_one(op)
    after_first = tmp_vault.read(target)
    res = execute_one(op)                          # same op again

    assert res.get("skipped") == "duplicate"
    assert tmp_vault.read(target) == after_first   # no second block appended
    assert after_first.count("## Note aggiuntive — Async IO (da meeting.md)") == 1


def test_patch_skipped_when_source_already_authored_note(tmp_vault):
    """Re-ingest idempotency: a source must not re-append its own concepts into
    a note it already authored. The note was WRITTEN on the first ingest (no
    provenance block), so block_present can't catch it — the provenance ledger
    (this source -> this note) does. Real incident: re-ingesting an edited
    lecture re-patched every unchanged concept into its own prior note."""
    target = tmp_vault.note("Concepts/Machine Learning.md", "---\n---\nseed body\n")
    append_record("lezione_1.md", "sha-v1", "run-1",
                  ["Concepts/Machine Learning"], vault_path=CONFIG.vault_path)

    assert note_authored_by(target, "lezione_1.md", vault_path=CONFIG.vault_path)

    op = Op(op=OpType.patch, heading="Machine Learning",
            source_basename="lezione_1.md", path=target,
            snippet="re-distilled excerpt", hub="Hub")
    res = execute_one(op)

    assert res.get("skipped") == "duplicate"
    assert "## Note aggiuntive" not in tmp_vault.read(target)


def test_patch_proceeds_for_a_different_source(tmp_vault):
    """A DIFFERENT source enriching the same note is a legit cross-source
    patch, not a re-ingest — it must still land."""
    target = tmp_vault.note("Concepts/Machine Learning.md", "---\nAI: true\n---\nseed\n")
    append_record("lezione_1.md", "sha-v1", "run-1",
                  ["Concepts/Machine Learning"], vault_path=CONFIG.vault_path)

    op = Op(op=OpType.patch, heading="Machine Learning",
            source_basename="lezione_9.md", path=target,   # other source
            snippet="new fact from a different lecture", hub="Hub")
    res = execute_one(op)

    assert res.get("skipped") is None
    assert "## Note aggiuntive — Machine Learning (da lezione_9.md)" in tmp_vault.read(target)


def test_duplicate_block_still_repairs_hub_link(tmp_vault):
    """A note holding the provenance block but NOT the hub link (state left by
    an interrupted run, or a pre-injection silica version) must gain the link
    on re-patch — otherwise the post-write lint fails the op on every retry
    (real incident: 2026-07-17 nucleate run, Claude Shannon.md)."""
    target = tmp_vault.note(
        "Topics/AsyncIO.md",
        "---\nAI: true\n---\nseed\n\n## Note aggiuntive — Async IO (da meeting.md)\n\nfacts\n",
    )
    op = Op(op=OpType.patch, heading="Async IO", source_basename="meeting.md",
            path=target, snippet="first fact", hub="Hub")

    res = execute_one(op)

    assert res.get("skipped") == "duplicate"
    after = tmp_vault.read(target)
    assert '[[Hub]]' in after                       # link repaired
    assert after.count("## Note aggiuntive") == 1   # snippet still skipped
