# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Offline harness tests for the LoCoMo e2e leg (fsm ingest + agent answer).

Zero LLM: every product seam is monkeypatched, per the adapter-test pattern.
"""
import json

from tests.eval.locomo import runner


# The one-shot system prompt as shipped today (baseline cell). The e2e leg's
# comparability rule: agent and one-shot prompts differ ONLY in the memory
# delivery sentence, so the judge sees the same contract.
_ONESHOT_SNAPSHOT = (
    "You are a helpful assistant answering questions from your memory of "
    "past conversations between Ann and Bob. Today's "
    "date is 2023-06-01. Use ONLY the memory provided. A 'Personal memory' "
    "section, when present, lists dated facts distilled from those "
    "conversations — treat them as reliable memory on par with the session "
    "transcripts. Answer concisely with only the information asked for. If "
    "the memory does not contain the answer, reply that you do not have "
    "that information — never guess."
)


def test_answer_contract_shared_and_oneshot_unchanged():
    open_ = runner._CONTRACT_OPEN.format(a="Ann", b="Bob", now="2023-06-01")
    oneshot = open_ + runner._ONESHOT_DELIVERY + runner._CONTRACT_CLOSE
    agent = open_ + runner._AGENT_DELIVERY + runner._CONTRACT_CLOSE
    assert oneshot == _ONESHOT_SNAPSHOT
    assert agent != oneshot
    assert agent.startswith(open_) and agent.endswith(runner._CONTRACT_CLOSE)


from types import SimpleNamespace


def _fsm_inst():
    """Two-session conversation, minimal locomo shape."""
    return {
        "sample_id": "conv-t",
        "conversation": {
            "speaker_a": "Ann", "speaker_b": "Bob",
            "session_1": [{"speaker": "Ann", "dia_id": "D1:1", "text": "I got a puppy."}],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_2": [{"speaker": "Bob", "dia_id": "D2:1", "text": "Nice puppy!"}],
            "session_2_date_time": "2:00 pm on 9 May, 2023",
        },
        "qa": [{"question": "Who got a puppy?", "answer": "Ann",
                "evidence": ["D1:1"], "category": 4}],
    }


class _StubCoordinator:
    """Records constructor kwargs; scripted run() results via class attrs."""
    calls: list = []
    results: list = []          # each entry: dict to return, or Exception to raise

    def __init__(self, **kw):
        type(self).calls.append(kw)
        self.fsm = SimpleNamespace(progress=SimpleNamespace(
            run_id=f"run{len(type(self).calls):02d}"))

    def run(self):
        r = type(self).results[len(type(self).calls) - 1] \
            if len(type(self).calls) <= len(type(self).results) else {}
        if isinstance(r, Exception):
            raise r
        return r


def _patch_fsm_seams(monkeypatch):
    _StubCoordinator.calls = []
    _StubCoordinator.results = []
    import silica.router.coordinator as coord_mod
    import silica.tools.pipeline as pipeline_mod
    monkeypatch.setattr(coord_mod, "Coordinator", _StubCoordinator)
    monkeypatch.setattr(pipeline_mod, "silica_anneal",
                        lambda steer=False, limit=0: {"bundles": 1, "written": 1,
                                                      "still_deferred": 0})
    monkeypatch.setattr(runner, "_clear_fsm_state", lambda: None)
    monkeypatch.setattr(runner, "_wipe_index_namespace", lambda: None)


def test_fsm_ingest_fresh_runs_sequentially_and_writes_marker(tmp_path, monkeypatch):
    _patch_fsm_seams(monkeypatch)
    vault = tmp_path / "conv-t"
    vault.mkdir()
    marker = runner.fsm_ingest_conversation(vault, _fsm_inst(), reuse=False,
                                            key_schema=False)
    assert marker["complete"] is True
    assert marker["sessions"] == ["session_1", "session_2"]
    assert marker["anneal"]["still_deferred"] == 0
    assert [c["inbox_files"] for c in _StubCoordinator.calls] == [
        ["inbox/session_1.md"], ["inbox/session_2.md"]]
    assert [c["seen_override"] for c in _StubCoordinator.calls] == [
        "2023-05-08", "2023-05-09"]
    assert "Ann: I got a puppy." in (vault / "inbox" / "session_1.md").read_text(encoding="utf-8")
    runs = json.loads((vault / "fsm_runs.json").read_text(encoding="utf-8"))
    assert runs == {"run01": "session_1", "run02": "session_2"}
    assert json.loads((vault / "fsm_ingest.json").read_text(encoding="utf-8"))["complete"]


def test_fsm_ingest_reuse_accepts_complete_marker_only(tmp_path, monkeypatch):
    _patch_fsm_seams(monkeypatch)
    vault = tmp_path / "conv-t"
    vault.mkdir()
    # Complete + consistent marker: zero Coordinator calls.
    (vault / "fsm_ingest.json").write_text(json.dumps(
        {"complete": True, "sessions": ["session_1", "session_2"],
         "anneal": {"still_deferred": 0}, "reused": False}), encoding="utf-8")
    marker = runner.fsm_ingest_conversation(vault, _fsm_inst(), reuse=True,
                                            key_schema=False)
    assert marker["reused"] is True
    assert _StubCoordinator.calls == []
    # Stale marker (session list mismatch): re-ingest from scratch.
    (vault / "fsm_ingest.json").write_text(json.dumps(
        {"complete": True, "sessions": ["session_1"]}), encoding="utf-8")
    marker = runner.fsm_ingest_conversation(vault, _fsm_inst(), reuse=True,
                                            key_schema=False)
    assert marker["reused"] is False
    assert len(_StubCoordinator.calls) == 2


def test_fsm_ingest_retry_once_then_fail_conversation(tmp_path, monkeypatch):
    _patch_fsm_seams(monkeypatch)
    vault = tmp_path / "conv-t"
    vault.mkdir()
    # Session 1: first attempt errors, retry succeeds. Session 2: both fail.
    _StubCoordinator.results = [{"error": "boom"}, {},
                                {"final_status": "partial"}, RuntimeError("crash")]
    marker = runner.fsm_ingest_conversation(vault, _fsm_inst(), reuse=False,
                                            key_schema=False)
    assert marker is None
    assert len(_StubCoordinator.calls) == 4
    assert not (vault / "fsm_ingest.json").exists()


def test_provenance_session_map_and_recall(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps([
        {"source": "session_1.md", "sha256": "a", "run_id": "r1",
         "date": "2023-05-08", "notes": ["memory/Puppy"]},
        {"source": "session_2.md", "sha256": "b", "run_id": "r2",
         "date": "2023-05-09", "notes": ["memory/Puppy", "memory/Bob"]},
        {"source": "session_3.md", "sha256": "c", "run_id": "r3",
         "date": "2023-05-10", "notes": ["memory/Puppy"]},
        {"source": "not-a-session.md", "sha256": "d", "run_id": "r4",
         "date": "2023-05-11", "notes": ["memory/Noise"]},
    ]), encoding="utf-8")
    m = runner._provenance_session_map(tmp_path)
    # A note merged from 3 sessions counts for all 3 (honest fusion semantics).
    assert m["memory/Puppy"] == {"session_1", "session_2", "session_3"}
    assert m["memory/Bob"] == {"session_2"}
    # Non-session sources are ignored; unknown notes count for no session.
    assert "memory/Noise" not in m
    assert runner._sessions_for(m, "memory/Ghost") == set()
    # Wikilink-name refs (silica_read_note takes names) fall back to basename.
    assert runner._sessions_for(m, "Puppy") == {"session_1", "session_2", "session_3"}
    assert runner._sessions_for(m, "memory/Bob") == {"session_2"}


def test_run_question_session_recall_via_session_map(monkeypatch):
    from silica.kernel import perception

    blocks = [SimpleNamespace(path="memory/Puppy"), SimpleNamespace(path="memory/Bob")]
    monkeypatch.setattr(perception, "perceive",
                        lambda *a, **kw: SimpleNamespace(
                            blocks=blocks, fact_chains=[], fact_hits=[],
                            render=lambda **k: "ctx"))
    session_map = {"memory/Puppy": {"session_1", "session_3"},
                   "memory/Bob": {"session_2"}}
    row = runner.run_question(
        {"question": "q?", "answer": "Ann", "evidence": ["D1:1", "D3:1"],
         "category": 4},
        "conv-t_q0", {}, model="stub", judge_model="stub", k=2, stuff=False,
        use_embedder=False, use_rerank=False, retrieval_only=True,
        distill=True, episodic_ttl=0, flat_context=False, facts_last=False,
        windows=None, window_chars=None, now="2023-05-09",
        speakers=("Ann", "Bob"), session_map=session_map, n_sessions=3)
    # gold = sessions 1 and 3; retrieved = 1, 2, 3 via the map -> recall 1.0
    assert row["session_recall"] == 1.0
    assert row["sessions"] == 3
