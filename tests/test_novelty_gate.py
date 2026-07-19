"""SAGE-style novelty gate (Tier 2 cost): pre-chunk diversion to the dedup lane.

The gate's order parameter is TITLE-vs-title cosine (like-vs-like): a concept
name is scored against stored note title vectors, never against full bodies.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from silica.router.states import setup as s


def _payload(*names):
    return {"schema_version": 1, "batches": [{
        "inbox_file": "in.md",
        "concepts": [{"name": n, "excerpt": f"about {n}"} for n in names],
    }]}


def _gate_fsm(queue=True):
    fsm = SimpleNamespace()
    fsm.context = {}
    fsm.inbox_file = "in.md"
    fsm.target_dir = "Notes"
    fsm.hub = "[[Hub]]"
    fsm._current_content_hash = "cafebabe"
    fsm._defer_ops = MagicMock(return_value=True)
    fsm.work_queue = MagicMock() if queue else None
    return fsm


def _hit(path="Notes/Existing.md", name="Existing", score=0.96):
    return {"path": path, "name": name, "score": score}


def _names(payload):
    return [c["name"] for b in payload.get("batches", []) for c in b.get("concepts", [])]


def _run_gate(fsm, payload, tau, hits_by_name):
    """Run the gate with the store's title search mocked per concept name.

    The embedder returns a one-element vector encoding each concept's position,
    so the title-search mock can map a vector back to its concept name without
    depending on how _note_title_text renders the name.
    """
    order = list(dict.fromkeys(_names(payload)))
    store = MagicMock()
    store.__len__ = MagicMock(return_value=5)
    store.title_cosine_top_k.side_effect = (
        lambda vec, k=5, exclude=None: hits_by_name.get(order[int(vec[0])], [])
    )
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [[float(i)] for i in range(len(texts))]
    with patch.object(s.orch.CONFIG, "novelty_tau", tau), \
         patch("silica.kernel.embed.get_store", return_value=store), \
         patch("silica.agent.providers.get_embedder", return_value=embedder), \
         patch("silica.kernel.paths.is_inbox_path", side_effect=lambda p: p.startswith("Inbox")):
        return s.novelty_gate(fsm, payload)


def test_tau_zero_returns_payload_untouched():
    fsm = _gate_fsm()
    payload = _payload("A")
    with patch.object(s.orch.CONFIG, "novelty_tau", 0.0), \
         patch("silica.kernel.embed.get_store",
               side_effect=AssertionError("gate must not touch the store at tau=0")):
        out, n = s.novelty_gate(fsm, payload)
    assert out is payload and n == 0


def test_diverts_at_tau_and_keeps_below():
    fsm = _gate_fsm()
    # The diverting concept's name must agree with its match (COLLISION's
    # lexical guard); "Existing" ~ note "Existing".
    hits = {"Existing": [_hit(name="Existing", score=0.96)],
            "Fresh": [_hit(name="Fresh", score=0.50)]}
    out, n = _run_gate(fsm, _payload("Existing", "Fresh"), 0.93, hits)
    assert _names(out) == ["Fresh"] and n == 1
    assert fsm._defer_ops.call_args.kwargs["phase"] == "NOVELTY"
    item = fsm.work_queue.enqueue.call_args.args[0]
    assert item.kind == "dedup" and item.target_path == "Notes/Existing.md"
    assert item.context["concept"] == "Existing"


def test_high_cosine_but_names_disagree_kept():
    # Negation pair: title cosine is high but the concepts are opposites, so
    # the lexical guard must keep the concept (no false divert).
    fsm = _gate_fsm()
    out, n = _run_gate(fsm, _payload("context-free"), 0.93,
                       {"context-free": [_hit(name="non context-free", score=0.97)]})
    assert _names(out) == ["context-free"] and n == 0
    fsm._defer_ops.assert_not_called()


def test_note_without_title_match_never_diverts():
    # A concept with no title neighbour (e.g. notes predate title_vec) → 0.0
    # score → empty hits → kept in the payload, nothing deferred.
    fsm = _gate_fsm()
    out, n = _run_gate(fsm, _payload("A"), 0.93, {"A": []})
    assert _names(out) == ["A"] and n == 0
    fsm._defer_ops.assert_not_called()


def test_inbox_candidate_filtered_before_decision():
    fsm = _gate_fsm()
    out, n = _run_gate(fsm, _payload("A"), 0.93,
                       {"A": [_hit(path="Inbox/staging.md", score=0.99),
                              _hit(score=0.50)]})
    assert _names(out) == ["A"] and n == 0


def test_queue_absent_defers_only():
    fsm = _gate_fsm(queue=False)
    out, n = _run_gate(fsm, _payload("Existing"), 0.93,
                       {"Existing": [_hit(name="Existing", score=0.97)]})
    assert n == 1 and _names(out) == []
    fsm._defer_ops.assert_called_once()


def test_handle_payload_registers_single_empty_chunk_when_fully_diverted():
    fsm = SimpleNamespace()
    fsm._current_file_idx = 0
    fsm.inbox_files = ["in.md"]
    fsm.inbox_file = "in.md"
    fsm.context = {"recon": [{"concepts": []}], "vault_graph_ctx": {}}
    fsm._progress_note = MagicMock()
    fsm._make_tmp = MagicMock(return_value="/tmp/recon.json")
    fsm._get_recipe_phase = MagicMock(return_value={})
    fsm.progress = MagicMock()
    fsm._chunks = []
    fsm._file_chunks = {}
    fsm._chunk_flat_to_fi_ci = {}
    fsm._transition_success = MagicMock()
    res = {"chunks": [{"schema_version": 1, "batches": [
        {"inbox_file": "in.md", "concepts": [{"name": "dup", "excerpt": "x"}]}]}]}
    with patch.object(s.orch, "silica_payload", return_value=res), \
         patch.object(s, "novelty_gate",
                      return_value=({"schema_version": 1, "batches": []}, 1)):
        s.handle_payload(fsm)
    assert len(fsm._chunks) == 1
    assert sum(len(b.get("concepts", [])) for b in fsm._chunks[0].get("batches", [])) == 0
    fsm._transition_success.assert_called_once()
