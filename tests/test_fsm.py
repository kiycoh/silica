from unittest.mock import patch, MagicMock, call
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


def test_validate_operations_hub_coercion():
    from silica.kernel.validate import validate_operations
    ops = [
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "hub": "import_from_inbox", "source_basename": "inbox.md"}
    ]
    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=RuntimeError("File not found")):
        validated, rejected = validate_operations(ops, [], "Deep Learning/Concepts", hub="CustomConceptsHub")
    assert len(rejected) == 0
    assert len(validated) == 2
    assert validated[0]["heading"] == "CustomConceptsHub"
    assert validated[0]["hub"] == "CustomConceptsHub"
    assert validated[1]["heading"] == "Neural Network"
    assert validated[1]["hub"] == "CustomConceptsHub"


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

@patch("silica.kernel.prep_delegation.run_distiller")
def test_fsm_delegate_single_chunk(mock_run_distiller):
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [
        {"chunk_id": 0, "concepts": ["a"]},
        {"chunk_id": 1, "concepts": ["b"]}
    ]
    fsm._current_chunk_idx = 0
    fsm.state = InjectorState.DELEGATE

    mock_run_distiller.return_value = {"updates": [{"op": "write", "path": "notes/note1.md", "heading": "Note 1"}]}

    with patch.object(fsm, "_make_tmp", return_value="temp_chunk_path.json") as mock_make_tmp:
        fsm.step()
        
        call_kwargs = mock_run_distiller.call_args.kwargs
        assert call_kwargs["payload"] == {"chunk_id": 0, "concepts": ["a"]}
        assert call_kwargs["target"] == "TargetDir"
        assert call_kwargs["hub"] == "TargetDir"
        # ledger_digest is injected by Phase 2 — just assert it's a string or None
        assert isinstance(call_kwargs.get("ledger_digest"), (str, type(None)))
        mock_make_tmp.assert_called_once_with({"updates": [{"op": "write", "path": "notes/note1.md", "heading": "Note 1"}]})
        assert fsm.state == InjectorState.SANITIZE
        assert fsm.context["distiller_output_path"] == "temp_chunk_path.json"


@patch("silica.router.orchestrator.silica_recon")
@patch("silica.router.orchestrator.silica_payload")
@patch("silica.kernel.prep_delegation.run_distiller")
@patch("silica.router.orchestrator.silica_sanitize")
@patch("silica.router.orchestrator.silica_validate_ops")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_snapshot")
@patch("silica.router.orchestrator.silica_bulk_write")
@patch("silica.router.orchestrator.silica_lint")
@patch("silica.tools.wrapped.silica_cleanup")
def test_fsm_multi_chunk_loop(
    mock_cleanup, mock_lint, mock_write, mock_snapshot, mock_driver,
    mock_validate, mock_sanitize, mock_run_distiller, mock_payload, mock_recon
):
    # Setup mock payload with 2 chunks
    mock_recon.return_value = {"success": True}
    mock_payload.return_value = {
        "chunks": [
            {"chunk_id": 0, "concepts": ["a"]},
            {"chunk_id": 1, "concepts": ["b"]}
        ]
    }
    mock_run_distiller.return_value = {"updates": [{"op": "write", "path": "notes/NoteA.md"}]}
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

    # Initialize FSM
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

    # Verify that the sequence of states visited loops back to DELEGATE
    expected_sequence = [
        # First chunk cycle
        InjectorState.RECON,
        InjectorState.PAYLOAD,
        InjectorState.SALIENCE,   # Phase 2.05 (best-effort, embedder unavailable → skip)
        InjectorState.COLLISION,  # Phase 5 (best-effort, no index → skip)
        InjectorState.DELEGATE,
        InjectorState.SANITIZE,
        InjectorState.VALIDATE,
        InjectorState.SNAPSHOT,
        InjectorState.WRITE,
        InjectorState.HUB_UPDATE,
        InjectorState.AUTOLINK,   # Phase 4
        InjectorState.LINT,
        InjectorState.CLEANUP,
        # Second chunk cycle (SALIENCE does not re-run)
        InjectorState.COLLISION,  # Phase 5 (best-effort, no index → skip)
        InjectorState.DELEGATE,
        InjectorState.SANITIZE,
        InjectorState.VALIDATE,
        InjectorState.SNAPSHOT,
        InjectorState.WRITE,
        InjectorState.HUB_UPDATE,
        InjectorState.AUTOLINK,   # Phase 4
        InjectorState.LINT,
        InjectorState.CLEANUP,
    ]
    assert states_visited == expected_sequence

    # Verify run_distiller was called twice, once for each chunk
    assert mock_run_distiller.call_count == 2
    # Phase 2: run_distiller now receives ledger_digest kwarg — check payloads only
    payloads_seen = [c.kwargs["payload"] for c in mock_run_distiller.call_args_list]
    assert {"chunk_id": 0, "concepts": ["a"]} in payloads_seen
    assert {"chunk_id": 1, "concepts": ["b"]} in payloads_seen

    # Verify cleanup (file move) was only called once (on the final chunk completion)
    mock_cleanup.assert_called_once_with("Inbox/test.md", "done")


def test_fsm_recipe_configuration():
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm._recipe is not None
    assert fsm._recipe["name"] == "injector"
    
    # Check gates configuration
    assert fsm._get_recipe_gate("rejection_rate_max", 0.05) == 0.10
    assert fsm._get_recipe_gate("graph_regression", "allow") == "forbid_new_orphans"

    # Check phases configuration
    payload_conf = fsm._get_recipe_phase("payload")
    assert payload_conf.get("partition_if_over") == 7
    
    distill_conf = fsm._get_recipe_phase("distill")
    assert distill_conf.get("max_workers") == 7


@patch("silica.router.orchestrator.silica_validate_ops")
def test_fsm_gate_all_rejected_steers_then_defers(mock_validate):
    # Phase 6: when ALL ops are rejected, VALIDATE steers back to DELEGATE (max 2
    # attempts). After the budget is exhausted, it goes to CLEANUP with "no_ops".
    all_rejected = {
        "success": True,
        "rejection_rate": 1.0,
        "total": 2,
        "validated_count": 0,
        "rejected_count": 2,
        "rejected_ops": [{"reason": "bad path"}],
        "validated_ops": [],
    }
    mock_validate.return_value = all_rejected

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [{"schema_version": 1, "batches": []}]
    fsm._current_chunk_idx = 0
    fsm.context["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE

    # First call: steer attempt 1 → go to DELEGATE
    fsm.step()
    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context.get("chunk_0_steer_attempts") == 1
    assert fsm.context.get("chunk_0_steer_context") is not None

    # Put FSM back at VALIDATE and call again (simulating steer attempt 2)
    fsm.context["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE
    fsm.step()
    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context.get("chunk_0_steer_attempts") == 2

    # Third call: budget exhausted → CLEANUP with "no_ops"
    fsm.context["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE
    fsm.step()
    assert fsm.state == InjectorState.CLEANUP
    assert fsm.context.get("final_status") == "no_ops"


@patch("silica.router.orchestrator.silica_validate_ops")
def test_fsm_gate_partial_rejection_continues(mock_validate):
    # When some ops pass and some are rejected, VALIDATE must continue the
    # pipeline (SNAPSHOT) and log a warning — it does NOT abort.
    mock_validate.return_value = {
        "success": True,
        "rejection_rate": 0.44,   # > 10% but partial
        "total": 9,
        "validated_count": 5,
        "rejected_count": 4,
        "rejected_ops": [],       # empty so deferred store is not called
        "validated_ops": [],
    }

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context["payload"] = {"payload": {"chunk_id": 0}}
    fsm.context["sanitized"] = {"parsed": []}
    fsm.context["source_content_hash"] = ""  # no deferred store call
    fsm.state = InjectorState.VALIDATE

    fsm.step()

    # Pipeline advances past VALIDATE (to SNAPSHOT)
    assert fsm.state == InjectorState.SNAPSHOT
    assert "abort_reason" not in fsm.context


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
    assert fsm.state == InjectorState.SALIENCE   # Phase 2.05

    fsm._transition_success()
    assert fsm.state == InjectorState.COLLISION  # Phase 5

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
    assert fsm.state == InjectorState.HUB_UPDATE

    fsm._transition_success()
    assert fsm.state == InjectorState.AUTOLINK  # Phase 4

    fsm._transition_success()
    assert fsm.state == InjectorState.LINT

    fsm._transition_success()
    assert fsm.state == InjectorState.CLEANUP
    
    fsm._transition_success()
    assert fsm.state == InjectorState.DONE


@patch("silica.router.orchestrator.silica_recon")
@patch("silica.router.orchestrator.silica_payload")
@patch("silica.kernel.prep_delegation.run_distiller")
@patch("silica.router.orchestrator.silica_sanitize")
@patch("silica.router.orchestrator.silica_validate_ops")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_snapshot")
@patch("silica.router.orchestrator.silica_bulk_write")
@patch("silica.router.orchestrator.silica_lint")
@patch("silica.tools.wrapped.silica_cleanup")
def test_fsm_recipe_end_to_end_flow(
    mock_cleanup, mock_lint, mock_write, mock_snapshot, mock_driver,
    mock_validate, mock_sanitize, mock_run_distiller, mock_payload, mock_recon
):
    # Setup mocks
    mock_recon.return_value = {"success": True}
    mock_payload.return_value = {"payload": {"chunk_id": 0}}
    mock_run_distiller.return_value = {"updates": []}
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
        InjectorState.SALIENCE,   # Phase 2.05 (best-effort, embedder unavailable → skip)
        InjectorState.COLLISION,  # Phase 5 (best-effort, empty index → skip)
        InjectorState.DELEGATE,
        InjectorState.SANITIZE,
        InjectorState.VALIDATE,
        InjectorState.SNAPSHOT,
        InjectorState.WRITE,
        InjectorState.HUB_UPDATE,
        InjectorState.AUTOLINK,   # Phase 4
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
        # The mock is called with (source_canonical, content_hash=...) — canonical drops .md, lowercased
        call_args = mock_ledger.is_committed.call_args
        assert call_args[0][0] == "already_processed", (
            f"Expected canonical key 'already_processed', got {call_args[0][0]!r}"
        )
        # content_hash kwarg must be present (may be empty str if file not found)
        assert "content_hash" in call_args[1]


# ---------------------------------------------------------------------------
# Phase 5 — Collision routing tests
# ---------------------------------------------------------------------------

def _make_fsm_at_collision(chunk_concepts: list, tmp_path=None) -> InjectorFSM:
    """Helper: build an FSM positioned at COLLISION with given chunk."""
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [{"schema_version": 1, "batches": [
        {"inbox_file": "Inbox/test.md", "concepts": chunk_concepts}
    ]}]
    fsm._current_chunk_idx = 0
    fsm.state = InjectorState.COLLISION
    fsm.context["source_content_hash"] = "abc123"
    return fsm


def test_collision_high_similarity_routes_to_patch():
    """score ≥ τ_high → pre-routed patch op stored in context, concept removed from chunk."""
    fsm = _make_fsm_at_collision([{"name": "Neural Networks", "excerpt": "A neural net intro."}])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5  # non-empty index
    mock_store.cosine_top_k.return_value = [{"path": "DL/Neural Networks.md", "name": "Neural Networks", "score": 0.92}]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.router.orchestrator.DRIVER") as mock_driver, \
         patch("silica.kernel.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65
        mock_driver.read_note.return_value = MagicMock()  # graph confirms node exists

        fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    collision_ops = fsm.context.get("chunk_0_collision_ops", [])
    assert len(collision_ops) == 1
    assert collision_ops[0]["op"] == "patch"
    assert collision_ops[0]["path"] == "DL/Neural Networks.md"
    # Concept removed from modified chunk
    remaining = fsm._chunks[0].get("batches", [])
    assert remaining == [] or all(len(b.get("concepts", [])) == 0 for b in remaining)


def test_collision_low_similarity_keeps_for_distillation():
    """score ≤ τ_low → concept kept in chunk for normal distillation, no patch ops."""
    fsm = _make_fsm_at_collision([{"name": "Quantum Entanglement", "excerpt": "Something unrelated."}])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5
    mock_store.cosine_top_k.return_value = [{"path": "Physics/Photons.md", "name": "Photons", "score": 0.40}]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.kernel.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65

        fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    collision_ops = fsm.context.get("chunk_0_collision_ops", [])
    assert collision_ops == []
    # Concept still in chunk
    concepts_left = [
        c
        for b in fsm._chunks[0].get("batches", [])
        for c in b.get("concepts", [])
    ]
    assert len(concepts_left) == 1
    assert concepts_left[0]["name"] == "Quantum Entanglement"


def test_collision_borderline_deferred():
    """τ_low < score < τ_high → concept deferred, removed from chunk, deferred store called."""
    fsm = _make_fsm_at_collision([{"name": "Backprop", "excerpt": "Backpropagation intro."}])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5
    mock_store.cosine_top_k.return_value = [{"path": "DL/Gradient Descent.md", "name": "Gradient Descent", "score": 0.75}]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_deferred = MagicMock()

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.kernel.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder), \
         patch("silica.kernel.deferred.get_deferred_store", return_value=mock_deferred):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65

        fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    collision_ops = fsm.context.get("chunk_0_collision_ops", [])
    assert collision_ops == []
    # Concept removed from chunk (deferred)
    concepts_left = [
        c
        for b in fsm._chunks[0].get("batches", [])
        for c in b.get("concepts", [])
    ]
    assert concepts_left == []
    # Deferred store was called
    mock_deferred.put.assert_called_once()
    put_kwargs = mock_deferred.put.call_args[1]
    assert put_kwargs["content_hash"] == "abc123"
    assert len(put_kwargs["rejected_ops"]) == 1


def test_collision_empty_index_skips_transparently():
    """Empty embedding index → COLLISION is a no-op, chunk flows unchanged."""
    concepts = [{"name": "Test Concept"}, {"name": "Another Concept"}]
    fsm = _make_fsm_at_collision(concepts)

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 0  # empty index

    with patch("silica.kernel.embed.EmbedStore", return_value=mock_store):
        fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context.get("chunk_0_collision_ops", []) == []
    # Chunk is UNCHANGED (not modified by collision)
    concepts_left = [
        c
        for b in fsm._chunks[0].get("batches", [])
        for c in b.get("concepts", [])
    ]
    assert len(concepts_left) == 2


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
@patch("silica.kernel.prep_delegation.run_distiller")
@patch("silica.router.orchestrator.silica_sanitize")
@patch("silica.router.orchestrator.silica_validate_ops")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_snapshot")
@patch("silica.tools.wrapped.silica_restore")
@patch("silica.tools.wrapped.silica_cleanup")
def test_fsm_create_settle_timeout_rollback(
    mock_cleanup, mock_restore, mock_snapshot, mock_driver,
    mock_validate, mock_sanitize, mock_run_distiller, mock_payload, mock_recon
):
    from silica.driver.base import SettleTimeout

    mock_recon.return_value = {"success": True}
    mock_payload.return_value = {"payload": {"chunk_id": 0}}
    mock_run_distiller.return_value = {"updates": []}
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





