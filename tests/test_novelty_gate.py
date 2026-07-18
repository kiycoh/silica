"""SAGE-style novelty gate (Tier 2 cost): pre-chunk diversion to the dedup lane."""
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


def _cand(path="Notes/Existing.md", name="Existing", score=0.96):
    return SimpleNamespace(path=path, name=name, embed_score=score)


def _run_gate(fsm, payload, tau, related_side_effect):
    store = MagicMock()
    store.__len__ = MagicMock(return_value=5)
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [[0.1, 0.2]] * len(texts)
    with patch.object(s.orch.CONFIG, "novelty_tau", tau), \
         patch("silica.kernel.embed.get_store", return_value=store), \
         patch("silica.agent.providers.get_embedder", return_value=embedder), \
         patch("silica.kernel.cooccurrence.get_cooccur_store", side_effect=Exception("absent")), \
         patch("silica.kernel.paths.is_inbox_path", side_effect=lambda p: p.startswith("Inbox")), \
         patch("silica.kernel.relatedness.related_notes_for_query",
               side_effect=related_side_effect):
        return s.novelty_gate(fsm, payload)


def _names(payload):
    return [c["name"] for b in payload.get("batches", []) for c in b.get("concepts", [])]


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
    sides = {"Dup": [_cand(score=0.96)], "Fresh": [_cand(score=0.50)]}
    out, n = _run_gate(fsm, _payload("Dup", "Fresh"), 0.93,
                       lambda **kw: sides[kw["query_text"].split("\n")[0]])
    assert _names(out) == ["Fresh"] and n == 1
    assert fsm._defer_ops.call_args.kwargs["phase"] == "NOVELTY"
    item = fsm.work_queue.enqueue.call_args.args[0]
    assert item.kind == "dedup" and item.target_path == "Notes/Existing.md"
    assert item.context["concept"] == "Dup"


def test_cooccur_only_candidate_never_diverts():
    fsm = _gate_fsm()
    out, n = _run_gate(fsm, _payload("A"), 0.93,
                       lambda **kw: [_cand(score=None)])
    assert _names(out) == ["A"] and n == 0
    fsm._defer_ops.assert_not_called()


def test_inbox_candidate_filtered_before_decision():
    fsm = _gate_fsm()
    out, n = _run_gate(fsm, _payload("A"), 0.93,
                       lambda **kw: [_cand(path="Inbox/staging.md", score=0.99),
                                     _cand(score=0.50)])
    assert _names(out) == ["A"] and n == 0


def test_queue_absent_defers_only():
    fsm = _gate_fsm(queue=False)
    out, n = _run_gate(fsm, _payload("Dup"), 0.93, lambda **kw: [_cand(score=0.97)])
    assert n == 1 and _names(out) == []
    fsm._defer_ops.assert_called_once()


def test_low_tau_warns_but_proceeds(caplog):
    import logging
    fsm = _gate_fsm()
    with caplog.at_level(logging.WARNING):
        _run_gate(fsm, _payload("A"), 0.5, lambda **kw: [_cand(score=0.4)])
    assert any("novelty_tau" in r.message for r in caplog.records)


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
