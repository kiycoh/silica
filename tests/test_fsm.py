from unittest.mock import patch, MagicMock
from silica.router.orchestrator import InjectorFSM, InjectorState
from silica.tools.registry import TOOLS

def test_injector_fsm_initialization():
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm.state.name == "INIT"


def test_injector_fsm_hub_inheritance():
    # If hub is None and target_dir is provided, it should inherit from target_dir's basename
    fsm = InjectorFSM("Inbox/test.md", "Deep Learning/Concepts")
    assert fsm.hub == "Concepts"

    # If hub is provided explicitly, it should preserve it
    fsm_explicit = InjectorFSM("Inbox/test.md", "Deep Learning/Concepts", hub="MyExplicitHub")
    assert fsm_explicit.hub == "MyExplicitHub"


def test_validate_operations_hub_inheritance():
    from silica.kernel.validate import validate_operations
    ops = [
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "source_basename": "inbox.md"}
    ]
    # We patch path_exists to return False so that the write operation is validated as a creation
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=RuntimeError("File not found")):
        validated, rejected = validate_operations(ops, [], "Deep Learning/Concepts")
    assert len(rejected) == 0
    # Returns 2: 1 for the auto-generated Hub note and 1 for the Neural Network spoke note
    assert len(validated) == 2
    assert validated[0]["heading"] == "Concepts"
    assert validated[0]["op"] == "write"
    assert validated[1]["heading"] == "Neural Network"
    assert validated[1]["hub"] == "Concepts"


def test_validate_operations_auto_creates_missing_hub():
    from silica.kernel.validate import validate_operations

    # Case A: Hub note doesn't exist anywhere in the vault
    ops_missing = [
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "hub": "Concepts", "source_basename": "inbox.md"}
    ]
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=RuntimeError("File not found")):
        validated, _ = validate_operations(ops_missing, [], "Deep Learning/Concepts")
    assert len(validated) == 2
    assert validated[0]["heading"] == "Concepts"
    assert validated[0]["path"] == "Deep Learning/Concepts/Concepts.md"

    # Case B: Hub note already exists in the vault
    ops_exists = [
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "hub": "Concepts", "source_basename": "inbox.md"}
    ]
    # Here read_note succeeds for the hub "Concepts" but fails for the spoke note
    def mock_read_note(ref):
        if ref == "Concepts":
            return MagicMock()
        raise RuntimeError("File not found")

    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=mock_read_note):
        validated, _ = validate_operations(ops_exists, [], "Deep Learning/Concepts")
    # Only 1 because Hub note already exists
    assert len(validated) == 1
    assert validated[0]["heading"] == "Neural Network"

    # Case C: Hub note is already being created by another write operation in the list
    ops_already_creating = [
        {"op": "write", "path": "Deep Learning/Concepts/Concepts.md", "heading": "Concepts", "hub": "Concepts", "source_basename": "inbox.md"},
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "hub": "Concepts", "source_basename": "inbox.md"}
    ]
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=RuntimeError("File not found")):
        validated, _ = validate_operations(ops_already_creating, [], "Deep Learning/Concepts")
    # 2 because the explicit creation is preserved, and no duplicate is injected
    assert len(validated) == 2




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


@patch("silica.router.orchestrator.silica_recon")
@patch("silica.router.orchestrator.silica_payload")
@patch("silica.agent.delegate.delegate")
@patch("silica.router.orchestrator.silica_sanitize")
@patch("silica.router.orchestrator.silica_validate_ops")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_snapshot")
@patch("silica.router.orchestrator.silica_bulk_write")
@patch("silica.router.orchestrator.silica_lint")
@patch("silica.tools.wrapped.silica_cleanup")
def test_fsm_recipe_end_to_end_flow(
    mock_cleanup, mock_lint, mock_write, mock_snapshot, mock_driver,
    mock_validate, mock_sanitize, mock_delegate, mock_payload, mock_recon
):
    # Setup mocks
    mock_recon.return_value = {"success": True}
    mock_payload.return_value = {"payload": {"chunk_id": 0}}
    mock_delegate.return_value = [{"updates": []}]
    mock_sanitize.return_value = {"parsed": []}
    mock_validate.return_value = {"success": True, "rejection_rate": 0.0, "validated_count": 1, "rejected_count": 0}
    mock_snapshot.return_value = {"txn_id": "txn_123", "inverses": []}
    mock_write.return_value = {"success": True}
    mock_lint.return_value = {"success": True}
    mock_cleanup.return_value = {"success": True}

    # Setup graph mocks
    pre_graph = MagicMock()
    post_graph = MagicMock()
    mock_driver.graph_snapshot.return_value = post_graph

    # Initialize FSM (loads actual injector.yaml recipe)
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    
    # Track states visited
    states_visited = []
    original_step = fsm.step
    def step_wrapper():
        states_visited.append(fsm.state)
        original_step()
    fsm.step = step_wrapper

    with patch("silica.kernel.graph_diff.check_graph_regression", return_value=(True, [])):
        res = fsm.run()

    # Check that it reached DONE state
    assert fsm.state == InjectorState.DONE
    assert res.get("final_status") == "Success"

    # Verify the sequence of states visited exactly matches the injector.yaml phases
    expected_sequence = [
        InjectorState.RECON,
        InjectorState.PAYLOAD,
        InjectorState.DELEGATE,
        InjectorState.SANITIZE,
        InjectorState.VALIDATE,
        InjectorState.SNAPSHOT,
        InjectorState.WRITE,
        InjectorState.LINT,
        InjectorState.CLEANUP,
    ]
    assert states_visited == expected_sequence


def test_fsm_already_ingested():
    # We patch get_ledger to return a mock ledger where is_committed is True
    with patch("silica.kernel.ledger.get_ledger") as mock_get_ledger:
        mock_ledger = MagicMock()
        mock_ledger.is_committed.return_value = True
        mock_get_ledger.return_value = mock_ledger

        fsm = InjectorFSM("Inbox/already_processed.md", "TargetDir")
        res = fsm.run()

        # Should short-circuit pre-RECON
        assert fsm.state == InjectorState.INIT
        assert res.get("final_status") == "already_ingested"
        mock_ledger.is_committed.assert_called_once_with("already_processed.md")


@patch("silica.agent.llm.call_llm")
@patch("silica.agent.providers.get_provider")
def test_worker_read_only(mock_get_provider, mock_call_llm):
    # Test 1: Verify that run_distiller calls call_llm with tools=None (single-shot variant)
    from silica.kernel.prep_delegation import run_distiller
    
    mock_get_provider.side_effect = Exception("Mocked provider failure")
    
    mock_response = MagicMock()
    mock_response.text = '{"updates": []}'
    mock_call_llm.return_value = mock_response

    payload = {"schema_version": 1, "batches": []}
    run_distiller(payload, target="TargetDir", hub="Hub")

    mock_call_llm.assert_called_once()
    _, kwargs = mock_call_llm.call_args
    assert kwargs.get("tools") is None

    # Test 2: Verify that build_worker_toolset excludes all mutation / wrapped / composed tools
    from silica.workers import build_worker_toolset, WORKER_BLOCKED_CLASSES, BLOCKED_TOOL_NAMES
    
    worker_tools = build_worker_toolset()
    
    for name, tool in worker_tools.items():
        assert tool.cls not in WORKER_BLOCKED_CLASSES
        assert name not in BLOCKED_TOOL_NAMES
        # Ensure only atomic read-only operations are returned
        assert tool.cls == "atomic"


@patch("silica.router.orchestrator.silica_validate_ops")
def test_fsm_short_circuit_no_ops(mock_validate):
    # Setup mock validation with 0 validated and 0 rejected (all skip/no-op)
    mock_validate.return_value = {
        "success": True,
        "validated_count": 0,
        "rejected_count": 0,
        "rejection_rate": 0.0,
        "total": 5,
    }

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context["payload"] = {"payload": {"chunk_id": 0}}
    fsm.context["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE

    with patch.object(fsm, "_make_tmp", return_value="dummy_ops.json"):
        fsm.step()

    # Verify transition directly to CLEANUP and final_status set to "no_ops"
    assert fsm.state == InjectorState.CLEANUP
    assert fsm.context["final_status"] == "no_ops"


@patch("silica.router.orchestrator.silica_recon")
@patch("silica.router.orchestrator.silica_payload")
@patch("silica.agent.delegate.delegate")
@patch("silica.router.orchestrator.silica_sanitize")
@patch("silica.router.orchestrator.silica_validate_ops")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_snapshot")
@patch("silica.tools.wrapped.silica_restore")
@patch("silica.tools.wrapped.silica_cleanup")
def test_fsm_create_settle_timeout_rollback(
    mock_cleanup, mock_restore, mock_snapshot, mock_driver,
    mock_validate, mock_sanitize, mock_delegate, mock_payload, mock_recon
):
    from silica.driver.base import SettleTimeout
    
    mock_recon.return_value = {"success": True}
    mock_payload.return_value = {"payload": {"chunk_id": 0}}
    mock_delegate.return_value = [{"updates": []}]
    mock_sanitize.return_value = {"parsed": [{"op": "write", "path": "test.md", "heading": "Test", "hub": "Hub", "source_basename": "test_fsm_settle.md"}]}
    
    import json
    def side_effect_validate(ops_json_path, **kwargs):
        with open(ops_json_path, 'w', encoding='utf-8') as f:
            json.dump([{"op": "write", "path": "test.md", "heading": "Test", "hub": "Hub", "source_basename": "test_fsm_settle.md"}], f)
        return {
            "success": True, "rejection_rate": 0.0, "validated_count": 1, "rejected_count": 0,
        }
    mock_validate.side_effect = side_effect_validate
    
    mock_snapshot.return_value = {
        "txn_id": "txn_123",
        "inverses": [{"kind": "delete_created", "path": "test.md"}]
    }
    mock_restore.return_value = {"success": True}
    
    pre_graph = MagicMock()
    mock_driver.graph_snapshot.return_value = pre_graph
    
    fsm = InjectorFSM("Inbox/test_fsm_settle.md", "TargetDir")
    
    # We patch silica.kernel.bulk.DRIVER.create to raise SettleTimeout
    with patch("silica.kernel.bulk.DRIVER.create", side_effect=SettleTimeout("Settle timeout mock error")):
        with patch("silica.kernel.graph_diff.check_graph_regression", return_value=(True, [])):
            res = fsm.run()
        
    assert fsm.state == InjectorState.ERROR
    assert "Rolled Back" in res.get("final_status")
    assert "Settle timeout mock error" in res.get("final_status")
    
    # Check that cleanup is not called (inbox not moved)
    mock_cleanup.assert_not_called()
    # Check that restore (rollback) was called
    mock_restore.assert_called_once()





