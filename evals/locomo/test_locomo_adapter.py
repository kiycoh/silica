# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""End-to-end check for the LoCoMo adapter, fully offline.

Same idiom as the LongMemEval adapter test: stub the LLM (answer + judge) so
load -> index -> perceive -> answer -> judge -> aggregate runs with no
network. The stub answer echoes the retrieved context and the stub judge says
'yes' iff the gold string appears in it, so a correct retrieval of the
evidence session yields a correct verdict.
"""
from __future__ import annotations

import silica.agent.llm as llm_mod
from silica.agent.llm import LLMResponse
from evals.locomo import runner


def _conv_inst():
    # session_1 holds the gold token (rare: zorblax); s2/s3 are distractors.
    return {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1",
                 "text": "I adopted a zorblax puppy today!",
                 "blip_caption": "a small dog on a rug"},
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "Congrats!"},
            ],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_2": [
                {"speaker": "Melanie", "dia_id": "D2:1",
                 "text": "The weather has been rainy all week."},
            ],
            "session_2_date_time": "10:00 am on 21 May, 2023",
            "session_3": [
                {"speaker": "Caroline", "dia_id": "D3:1",
                 "text": "I started a pottery class downtown."},
            ],
            "session_3_date_time": "3:12 pm on 2 June, 2023",
        },
        "qa": [
            {"question": "What kind of pet did Caroline adopt, the zorblax one?",
             "answer": "a zorblax puppy", "evidence": ["D1:1"], "category": 4},
            {"question": "What is the name of Melanie's boat?",
             "adversarial_answer": "Sea Breeze", "category": 5},
        ],
    }


def _install_stub(monkeypatch, gold="zorblax puppy"):
    def fake_call_llm(model, messages, **kw):
        prompt = messages[-1]["content"]
        if "Answer yes or no" in prompt:  # judge turn
            return LLMResponse(text="yes" if gold.lower() in prompt.lower() else "no")
        return LLMResponse(text=prompt)  # answer turn: echo the memory context
    monkeypatch.setattr(llm_mod, "call_llm", fake_call_llm)


# --- Pure dataset-shape helpers ----------------------------------------------

def test_parse_date_time():
    assert runner.parse_date_time("1:56 pm on 8 May, 2023") == "2023-05-08"
    assert runner.parse_date_time("8 May, 2023") == "2023-05-08"
    assert runner.parse_date_time("nonsense") == ""
    assert runner.parse_date_time("") == ""


def test_render_turn_keeps_speaker_and_photo():
    line = runner.render_turn({"speaker": "Caroline", "text": "Look!",
                               "blip_caption": "a dog on a beach"})
    assert line == "Caroline: Look! [shares a photo: a dog on a beach]"
    # Photo-only turn still renders (evidence can point at photos).
    assert runner.render_turn({"speaker": "Mel", "blip_caption": "sunset"}) == \
        "Mel: [shares a photo: sunset]"


def test_evidence_sessions():
    assert runner.evidence_sessions(["D1:3", "D12:9", "junk"]) == \
        {"session_1", "session_12"}
    assert runner.evidence_sessions([]) == set()


def test_conversation_sessions_numeric_order():
    conv = {"speaker_a": "A", "speaker_b": "B",
            "session_10": [{"speaker": "A", "text": "late"}],
            "session_10_date_time": "1 May, 2024",
            "session_2": [{"speaker": "B", "text": "early"}],
            "session_2_date_time": "1 May, 2023"}
    nums = [n for n, _dt, _turns in runner.conversation_sessions(conv)]
    assert nums == [2, 10]  # numeric, not lexicographic


# --- Vault lifecycle ---------------------------------------------------------

def test_load_conversation_vault_and_reuse(tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    runner.bind_vault(vault)
    index = runner.load_conversation_vault(vault, _conv_inst())
    assert set(index) == {"sessions/s0000", "sessions/s0001", "sessions/s0002"}
    assert index["sessions/s0000"] == {"session_id": "session_1", "date": "2023-05-08"}
    note = (vault / "sessions" / "s0000.md").read_text()
    assert "Caroline: I adopted a zorblax puppy today!" in note
    assert "[shares a photo: a small dog on a rug]" in note
    assert "source: locomo" in note
    assert runner._conv_now(index) == "2023-06-02"
    # reuse adopts the notes untouched and rebuilds the index from frontmatter.
    assert runner.load_conversation_vault(vault, _conv_inst(), reuse=True) == index


def test_distill_receives_named_speaker_excerpt(tmp_path, monkeypatch):
    seen = {}

    def fake_distill(sid, date, excerpt):
        seen[sid] = (date, excerpt)
        return "distilled body"

    monkeypatch.setattr(runner, "distill_session", fake_distill)
    vault = tmp_path / "v"
    vault.mkdir()
    runner.bind_vault(vault)
    runner.load_conversation_vault(vault, _conv_inst(), distill=True)
    date, excerpt = seen["session_1"]
    assert date == "2023-05-08"
    # The shared distiller seam gets the NAMED-speaker rendering, not User/Assistant.
    assert "Caroline: I adopted a zorblax puppy today!" in excerpt
    assert "distilled body" in (vault / "sessions" / "s0000.md").read_text()


# --- Pipeline ----------------------------------------------------------------

def test_stuff_pipeline_scores_correct(tmp_path, monkeypatch):
    _install_stub(monkeypatch)
    doc = runner.run([_conv_inst()], tmp_path / "run", model="stub",
                     judge_model="stub", k=10, stuff=True, use_embedder=False)
    m = doc["metrics"]
    assert m["overall_accuracy"] == 1.0  # gold session fed -> judge says yes
    assert m["answerable_n"] == 1
    assert m["by_type"]["single-hop"]["accuracy"] == 1.0
    assert m["abstention_n"] == 1  # category 5 lands in the abstention bucket
    # --stuff feeds all sessions, so per-question session recall is not computed.
    assert m["session_recall_mean"] is None


def test_facade_pipeline_session_recall(tmp_path, monkeypatch):
    import silica.kernel.embed as embed_mod

    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    _install_stub(monkeypatch)
    doc = runner.run([_conv_inst()], tmp_path / "run", model="stub",
                     judge_model="stub", k=1, stuff=False, use_embedder=False,
                     use_rerank=False)
    by_type = {r["question_type"]: r for r in doc["questions"]}
    row = by_type["single-hop"]
    # k=1 on 3 sessions: only the shared rare token can rank the gold first.
    assert row["session_recall"] == 1.0
    assert row["correct"] is True
    assert row["gold_in_context"] is True
    assert doc["metrics"]["overall_accuracy"] == 1.0
    # Adversarial row: no evidence, recall undefined, not a 0.0 miss.
    assert by_type["adversarial"]["session_recall"] is None


def test_categories_filter_and_limit(tmp_path, monkeypatch):
    _install_stub(monkeypatch)
    doc = runner.run([_conv_inst()], tmp_path / "run", model="stub",
                     judge_model="stub", k=10, stuff=True, use_embedder=False,
                     categories={4})
    assert [r["question_type"] for r in doc["questions"]] == ["single-hop"]
    assert doc["config"]["categories"] == [4]
    doc2 = runner.run([_conv_inst()], tmp_path / "run2", model="stub",
                      judge_model="stub", k=10, stuff=True, use_embedder=False,
                      limit=1)
    assert len(doc2["questions"]) == 1
