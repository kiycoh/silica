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


class TestRefinerProducesBaseContent:
    def test_reformat_overwrite_op_carries_base_content(self, tmp_path):
        """The refiner triage must snapshot the read content into base_content
        so the write path can detect a concurrent edit (charter UC6)."""
        import silica.config
        import silica.driver
        from silica.router.refiner_fsm import RefinerFSM

        folder = tmp_path / "notes"
        folder.mkdir()
        silica.config.CONFIG.backend = "fs"
        silica.config.CONFIG.vault_path = str(folder)
        silica.driver._driver = None

        # Dirty tags in frontmatter + non-lean body → "reformat" → overwrite op
        original = (
            "---\ntags: [Tag Uno, TagDue]\n---\n# Titolo\n\n"
            + ("Contenuto sostanzioso della nota. " * 30)
            + "\n"
        )
        (folder / "Nota.md").write_text(original, encoding="utf-8")

        fsm = RefinerFSM(str(folder))
        fsm._handle_triage()

        overwrites = [
            o for o in fsm.context["mechanical_ops"] if o["op"] == "overwrite"
        ]
        assert overwrites, "expected a reformat overwrite op"
        for o in overwrites:
            assert o.get("base_content") == original
