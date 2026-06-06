# tests/test_provenance_dedup.py
from silica.kernel.templates import provenance_header, block_present
from silica.kernel.ops import Op, OpType
from silica.kernel.bulk import execute_one


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
