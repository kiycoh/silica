from unittest.mock import patch, MagicMock
from silica.router.orchestrator import InjectorFSM, InjectorState
from silica.tools.registry import TOOLS

def test_injector_fsm_initialization():
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm.state.name == "INIT"

def test_silica_run_injector_is_registered():
    assert "silica_run_injector" in TOOLS
    tool = TOOLS["silica_run_injector"]
    assert tool.cls == "composed"

@patch("silica.agent.delegate.delegate")
@patch("silica.kernel.prep_delegation.run_distiller")
def test_fsm_delegate_merge_dedup(mock_run_distiller, mock_delegate):
    # Setup mock to return multiple chunks
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context["payload"] = {
        "chunks": [
            {"chunk_id": 0},
            {"chunk_id": 1}
        ]
    }
    fsm.state = InjectorState.DELEGATE

    # mock delegate to return results from two workers
    # chunk 0 writes to note1.md
    # chunk 1 patches note1.md (shorter snippet) and writes to note2.md
    mock_delegate.return_value = [
        {"updates": [{"op": "write", "path": "notes/note1.md", "heading": "Note 1", "snippet": "Long snippet"}]},
        {"updates": [
            {"op": "patch", "path": "notes/note1.md", "heading": "Note 1", "snippet": "Short"},
            {"op": "write", "path": "notes/note2.md", "heading": "Note 2", "snippet": "Snippet 2"}
        ]}
    ]

    with patch.object(fsm, "_make_tmp", return_value="temp_merged_path.json") as mock_make_tmp:
        fsm.step()
        
        # Verify delegate was called with the 2 chunks and run_one function
        mock_delegate.assert_called_once()
        args, kwargs = mock_delegate.call_args
        assert len(args[0]) == 2
        assert kwargs["max_workers"] == 7

        # Verify that merged results are passed to make_tmp
        mock_make_tmp.assert_called_once()
        merged_data = mock_make_tmp.call_args[0][0]
        
        # Check that note1.md patch was marked as "skip" because note1.md write has the richer snippet
        updates = merged_data["updates"]
        assert len(updates) == 3
        
        note1_write = next(u for u in updates if u["path"] == "notes/note1.md" and u["op"] == "write")
        note1_patch = next(u for u in updates if u["path"] == "notes/note1.md" and u["op"] == "skip")
        note2_write = next(u for u in updates if u["path"] == "notes/note2.md" and u["op"] == "write")

        assert note1_write["snippet"] == "Long snippet"
        assert note1_patch["op"] == "skip"
        assert "Duplicate" in note1_patch["reason"]
        assert note2_write["snippet"] == "Snippet 2"

        # Verify state transition to SANITIZE
        assert fsm.state == InjectorState.SANITIZE
        assert fsm.context["distiller_output_path"] == "temp_merged_path.json"


def test_fsm_recipe_configuration():
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm._recipe is not None
    assert fsm._recipe["name"] == "injector"
    
    # Check gates configuration
    assert fsm._get_recipe_gate("rejection_rate_max", 0.05) == 0.10
    assert fsm._get_recipe_gate("graph_regression", "allow") == "forbid_new_orphans"

    # Check phases configuration
    payload_conf = fsm._get_recipe_phase("payload")
    assert payload_conf.get("partition_if_over") == 200
    
    distill_conf = fsm._get_recipe_phase("distill")
    assert distill_conf.get("max_workers") == 7


@patch("silica.router.orchestrator.silica_validate_ops")
def test_fsm_gate_rejection(mock_validate):
    # Setup mock validation with high rejection rate to trigger gate abort
    mock_validate.return_value = {
        "success": False,
        "rejection_rate": 0.15,
        "total": 10,
        "rejected_count": 2,
    }

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context["payload"] = {"payload": {"chunk_id": 0}}
    fsm.context["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE

    fsm.step()

    # Verify transition to ERROR because rejection rate 15% >= 10%
    assert fsm.state == InjectorState.ERROR
    assert "Rejection rate 15.0% >= 10.0%" in fsm.context["abort_reason"]


@patch("silica.router.orchestrator.silica_lint")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_restore")
@patch("builtins.open")
def test_fsm_graph_regression_gate_rollback(mock_open, mock_restore, mock_driver, mock_lint):
    # Setup mock file reading for ops_path
    mock_open.return_value.__enter__.return_value.read.return_value = '[]'
    
    # Lint passes
    mock_lint.return_value = {"success": True}
    
    # Pre-graph exists
    pre_graph = MagicMock()
    post_graph = MagicMock()
    mock_driver.graph_snapshot.return_value = post_graph
    
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context["ops_path"] = "dummy_ops.json"
    fsm._pre_graph = pre_graph
    fsm._txn = MagicMock()
    fsm._txn.created_paths = ["notes/NoteA.md"]
    
    # Setup snapshot data for rollback inverse application
    inverses = [{"op": "delete", "path": "notes/NoteA.md"}]
    fsm.context["snapshot"] = {
        "txn_id": "txn_123",
        "inverses": inverses
    }
    
    fsm.state = InjectorState.LINT
    
    # Mock the regression check to fail
    with patch("silica.kernel.graph_diff.check_graph_regression", return_value=(False, ["New orphans detected"])) as mock_check:
        fsm.step()
        
        # Verify check_graph_regression was called with pre_graph, post_graph, and created_paths
        mock_check.assert_called_once_with(pre_graph, post_graph, ["notes/NoteA.md"])
        
        # Verify transition to ROLLBACK on gate rejection
        assert fsm.state == InjectorState.ROLLBACK
        assert "Graph regression gate failed: New orphans detected" in fsm.context["abort_reason"]
        
    # Now run the ROLLBACK step
    mock_restore.return_value = {"success": True}
    fsm.step()
    
    # Verify restore was called with the correct parameters
    mock_restore.assert_called_once_with(txn_id="txn_123", inverses=inverses)
    
    # Verify final status transitions to ERROR
    assert fsm.state == InjectorState.ERROR
    assert "Rolled Back" in fsm.context["final_status"]


def test_fsm_recipe_transition_sequence():
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    
    # Verify sequential progression
    fsm.state = InjectorState.RECON
    fsm._transition_success()
    assert fsm.state == InjectorState.PAYLOAD
    
    fsm._transition_success()
    assert fsm.state == InjectorState.DELEGATE
    
    fsm._transition_success()
    assert fsm.state == InjectorState.SANITIZE
    
    fsm._transition_success()
    assert fsm.state == InjectorState.VALIDATE
    
    fsm._transition_success()
    assert fsm.state == InjectorState.SNAPSHOT
    
    fsm._transition_success()
    assert fsm.state == InjectorState.WRITE
    
    fsm._transition_success()
    assert fsm.state == InjectorState.LINT
    
    fsm._transition_success()
    assert fsm.state == InjectorState.CLEANUP
    
    fsm._transition_success()
    assert fsm.state == InjectorState.DONE


