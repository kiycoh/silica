"""Wiring of the 3-way merge into the write path (kernel/bulk._execute_overwrite).

An overwrite Op may carry `base_content` — the note content at the time the op
was built. If the note on disk has changed since (concurrent user edit), the
write must not stomp silently: the incoming content is written with a conflict
callout prepended (kernel/merge.py), and the result reports the conflict.
"""
import pytest

from silica.kernel.bulk import execute_one
from silica.kernel.merge import CONFLICT_CALLOUT_HEADER
from silica.kernel.ops import Op, OpType


@pytest.fixture(autouse=True)
def clean_ledger(tmp_path):
    """Reset the global ledger singleton to a fresh temp DB before each test."""
    import silica.kernel.ledger as _ledger_mod
    fresh = _ledger_mod.Ledger(tmp_path / "test_ledger.db")
    old = _ledger_mod._ledger
    _ledger_mod._ledger = fresh
    yield
    _ledger_mod._ledger = old


def _overwrite_op(content: str, base: str | None = None) -> Op:
    return Op(
        op=OpType.overwrite,
        heading="Nota",
        source_basename="src.md",
        path="Nota.md",
        content=content,
        base_content=base,
    )


class TestOverwriteConflictWiring:
    def test_stale_base_injects_conflict_callout(self, tmp_vault):
        path = tmp_vault.note("Nota.md", "v1 originale\n")
        # concurrent user edit after the op's snapshot
        tmp_vault.write(path, "v2 modifica utente\n")

        res = execute_one(_overwrite_op("v3 agente\n", base="v1 originale\n"))

        assert res["success"] is True
        assert res["conflict"] is True
        on_disk = tmp_vault.read(path)
        assert CONFLICT_CALLOUT_HEADER in on_disk
        assert "v3 agente" in on_disk

    def test_matching_base_writes_clean(self, tmp_vault):
        path = tmp_vault.note("Nota.md", "v1 originale\n")

        res = execute_one(_overwrite_op("v2 agente\n", base="v1 originale\n"))

        assert res["success"] is True
        assert not res.get("conflict")
        assert tmp_vault.read(path) == "v2 agente\n"

    def test_without_base_keeps_legacy_behavior(self, tmp_vault):
        path = tmp_vault.note("Nota.md", "v1 originale\n")
        tmp_vault.write(path, "v2 modifica utente\n")

        res = execute_one(_overwrite_op("v3 agente\n"))

        assert res["success"] is True
        assert not res.get("conflict")
        assert tmp_vault.read(path) == "v3 agente\n"


class TestValidateSnapshotsBaseContent:
    """Overwrite ops flowing through validate_operations (collision/distill
    path, deferred retries) must snapshot the current note into base_content
    so the write path can detect a concurrent edit — the refiner does this at
    triage time, every other producer relies on validation as the choke point."""

    def _overwrite_dict(self, path: str, content: str = "v-nuova\n") -> dict:
        import os
        return {
            "op": "overwrite",
            "path": path,
            "heading": os.path.splitext(os.path.basename(path))[0],
            "source_basename": "src.md",
            "content": content,
            "hub": os.path.splitext(os.path.basename(path))[0],
        }

    def test_overwrite_on_existing_note_snapshots_base_content(self, tmp_vault):
        import os
        from silica.kernel.validate import validate_operations

        path = tmp_vault.note("Nota.md", "v1 originale\n")

        validated, rejected = validate_operations(
            [self._overwrite_dict(path)],
            payloads=[],
            target_dir=os.path.dirname(path),
        )

        assert not rejected
        overwrites = [o for o in validated if o.op == OpType.overwrite]
        assert overwrites, "expected the overwrite op to survive validation"
        assert overwrites[0].base_content == "v1 originale\n"

    def test_existing_base_content_is_not_clobbered(self, tmp_vault):
        """A producer that already snapshotted (the refiner, at triage read
        time) carries the more faithful base: validation must keep it."""
        import os
        from silica.kernel.validate import validate_operations

        path = tmp_vault.note("Nota.md", "v2 corrente\n")
        op_dict = self._overwrite_dict(path)
        op_dict["base_content"] = "v1 snapshot del refiner\n"

        validated, _ = validate_operations(
            [op_dict], payloads=[], target_dir=os.path.dirname(path)
        )

        op = next(o for o in validated if o.op == OpType.overwrite)
        assert op.base_content == "v1 snapshot del refiner\n"

    def test_validated_overwrite_detects_concurrent_edit_at_write(self, tmp_vault):
        """End-to-end: validate snapshots the base, a concurrent edit lands,
        the write injects the conflict callout instead of stomping."""
        import os
        from silica.kernel.validate import validate_operations

        path = tmp_vault.note("Nota.md", "v1 originale\n")

        validated, _ = validate_operations(
            [self._overwrite_dict(path, content="v3 agente\n")],
            payloads=[],
            target_dir=os.path.dirname(path),
        )
        op = next(o for o in validated if o.op == OpType.overwrite)

        # concurrent user edit between validation and write
        tmp_vault.write(path, "v2 modifica utente\n")

        res = execute_one(op)

        assert res["success"] is True
        assert res["conflict"] is True
        assert CONFLICT_CALLOUT_HEADER in tmp_vault.read(path)
