import os
import pytest
from unittest.mock import patch, MagicMock
from silica.router.refiner_fsm import RefinerFSM, RefinerState
from silica.driver import DRIVER
from silica.kernel.ledger import get_ledger
import silica.config
import silica.driver

@pytest.fixture(autouse=True)
def clean_ledger(tmp_path):
    """Reset the global ledger singleton to a fresh temp DB before each test."""
    import silica.kernel.ledger as _ledger_mod
    fresh = _ledger_mod.Ledger(tmp_path / "test_ledger.db")
    old = _ledger_mod._ledger
    _ledger_mod._ledger = fresh
    yield
    _ledger_mod._ledger = old

@patch("silica.agent.llm.call_llm")
def test_refiner_full_flow(mock_call_llm, tmp_path):
    # Set up mock folder
    folder = tmp_path / "notes"
    folder.mkdir()

    # Configure driver to use FS backend pointed at our temp folder
    silica.config.CONFIG.backend = "fs"
    silica.config.CONFIG.vault_path = str(folder)
    silica.driver._driver = None  # Reset lazy singleton
    
    # Note 1: Monolith (over limits and >= 2 H2 headings)
    # limit is max_chars=20000 or max_lines=400. We can write 405 lines to trigger max_lines.
    monolith_content = "---\nAI: false\n---\n# Monolith\n\n" + "\n".join(f"## Section {i}\nContent {i}" for i in range(3))
    monolith_lines = monolith_content + "\n" * 400
    monolith_path = folder / "monolith.md"
    monolith_path.write_text(monolith_lines, encoding="utf-8")
    
    # Note 2: Lean/empty note (<600 chars)
    lean_path = folder / "lean.md"
    lean_path.write_text("---\nAI: false\ntags: [TagOne, TagTwo]\n---\n# Lean Note\nSome lean text.", encoding="utf-8")
    
    # Note 3: Reformat (has frontmatter issues but normal size)
    reformat_path = folder / "reformat.md"
    reformat_body = "# Normal Note\n" + "x" * 700
    reformat_path.write_text(f"---\nAI: true\ntags: [TagOne, TagTwo]\n---\n{reformat_body}", encoding="utf-8")
    
    # Note 4: OK (correct frontmatter and body >600 chars, no frontmatter issues)
    ok_path = folder / "ok.md"
    ok_body = "# OK Note\n" + "x" * 700
    ok_path.write_text(f"---\nAI: true\ntags: [tag-one, tag-two]\nlast modified: 2026, 05, 25\nparent note: \"[[Hub]]\"\n---\n{ok_body}", encoding="utf-8")

    # Mock call_llm response for enrichment — use [[lean]] (resolved link) instead of [[Lean Note]] (unresolved link)
    mock_response = MagicMock()
    mock_response.text = '{"content": "---\\ntags:\\n  - tag-one\\n  - tag-two\\nAI: true\\n---\\n# Lean Note\\nEnriched content here.\\n\\n[[lean]]"}'
    mock_call_llm.return_value = mock_response

    # Mock validate, lint, restore, snapshot to succeed
    with patch("silica.router.refiner_fsm.silica_validate_ops") as mock_validate, \
         patch("silica.router.refiner_fsm.silica_lint") as mock_lint, \
         patch("silica.tools.wrapped.silica_snapshot") as mock_snapshot:
         
        mock_validate.return_value = {"success": True, "rejection_rate": 0.0}
        mock_lint.return_value = {"success": True}
        mock_snapshot.return_value = {"success": True, "txn_id": "test_txn_123", "inverses": []}

        fsm = RefinerFSM(str(folder))
        res = fsm.run()

        
        assert res.get("final_status") == "Success"
        summary = res["triage_summary"]
        assert summary["decouple"] == 1
        assert summary["enrich"] == 1
        assert summary["reformat"] == 1
        assert len(res["ops"]) > 0

        # Verify ledger has committed records (keyed by path-canonical, not basename)
        ledger = get_ledger()
        committed_canonicals = {
            row[0]
            for row in ledger._conn.execute(
                "SELECT source_canonical FROM ops WHERE status='committed'"
            ).fetchall()
        }
        # Canonical keys are folder-relative, lowercase, no .md
        assert any(c.endswith("monolith") for c in committed_canonicals), (
            f"No committed row for 'monolith' found. Committed: {committed_canonicals}"
        )
        assert any(c.endswith("lean") for c in committed_canonicals), (
            f"No committed row for 'lean' found. Committed: {committed_canonicals}"
        )
        assert any(c.endswith("reformat") for c in committed_canonicals), (
            f"No committed row for 'reformat' found. Committed: {committed_canonicals}"
        )

@patch("silica.agent.llm.call_llm")
def test_refiner_rollback_on_lint_failure(mock_call_llm, tmp_path):
    # Set up mock folder
    folder = tmp_path / "notes"
    folder.mkdir()

    # Configure driver to use FS backend pointed at our temp folder
    silica.config.CONFIG.backend = "fs"
    silica.config.CONFIG.vault_path = str(folder)
    silica.driver._driver = None  # Reset lazy singleton
    
    reformat_path = folder / "reformat.md"
    reformat_body = "# Normal Note\n" + "x" * 700
    reformat_path.write_text(f"---\nAI: true\ntags: [TagOne, TagTwo]\n---\n{reformat_body}", encoding="utf-8")

    # Mock call_llm response for enrichment in case it triggers
    mock_response = MagicMock()
    mock_response.text = '{"content": "---\\ntags:\\n  - tag-one\\n  - tag-two\\nAI: true\\n---\\n# Normal Note\\nEnriched content here."}'
    mock_call_llm.return_value = mock_response

    # Mock validate, lint, restore, snapshot
    with patch("silica.router.refiner_fsm.silica_validate_ops") as mock_validate, \
         patch("silica.router.refiner_fsm.silica_lint") as mock_lint, \
         patch("silica.tools.wrapped.silica_snapshot") as mock_snapshot, \
         patch("silica.tools.wrapped.silica_restore") as mock_restore:
         
        mock_validate.return_value = {"success": True, "rejection_rate": 0.0}
        # Lint fails!
        mock_lint.return_value = {"success": False, "errors": ["Malformed headings"]}
        mock_snapshot.return_value = {"success": True, "txn_id": "test_txn_456", "inverses": [{"kind": "restore_version", "path": str(reformat_path), "version": 1}]}
        mock_restore.return_value = {"success": True}

        fsm = RefinerFSM(str(folder))
        res = fsm.run()

        
        assert "Rolled Back" in res.get("final_status", "")
        mock_restore.assert_called_once_with(txn_id="test_txn_456", inverses=[{"kind": "restore_version", "path": str(reformat_path), "version": 1}])
        
        # Verify ledger does NOT mark reformat.md as committed
        ledger = get_ledger()
        committed_canonicals = {
            row[0]
            for row in ledger._conn.execute(
                "SELECT source_canonical FROM ops WHERE status='committed'"
            ).fetchall()
        }
        assert not any(c.endswith("reformat") for c in committed_canonicals), (
            f"'reformat' should not be committed after rollback. Committed: {committed_canonicals}"
        )
