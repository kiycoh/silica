"""Frozen probe: does the fixed abstention rubric credit correct false-premise
rejections that the old rubric scored as failures?

Re-judges STORED responses (no model re-run) with the current _JUDGE_ABSTENTION.
Default: the 3 known LoCoMo cat-5 false-negatives (model caught the Caroline/
Melanie swap and abstained, old judge said "no") + 1 synthetic confabulation.
Asserts the corrections flip to True and the confabulation stays False.

    uv run python -m evals.probe_abstention_rubric              # 4-case self-check
    uv run python -m evals.probe_abstention_rubric --full       # re-score all 47 adversarial in R26b

Needs a judge model (CONFIG.model) reachable, i.e. API key set. ponytail: the
4-case assert is the check; --full is the operator re-score, not the gate.
"""
import argparse
import json
from pathlib import Path

from silica.config import CONFIG
from evals.longmemeval.runner import judge

BENCH = Path("bench")
R26B = BENCH / "locomo_e2e.R26b.metrics.json"
LOCO = BENCH / "locomo10.json"

# The three responses from the diagnosis: each abstains AND explains the false
# premise; the old rubric marked all three wrong.
KNOWN_FALSE_NEGATIVES = ["conv-26_q152", "conv-26_q153", "conv-26_q155"]


def _load():
    rows = {q["question_id"]: q for q in json.load(open(R26B))["questions"]}
    qa = {c["sample_id"]: c["qa"] for c in json.load(open(LOCO))}
    def adversarial(qid):
        conv, i = qid.rsplit("_q", 1)
        return qa[conv][int(i)].get("adversarial_answer")
    def question(qid):
        conv, i = qid.rsplit("_q", 1)
        return qa[conv][int(i)]["question"]
    return rows, question, adversarial


def selfcheck(model):
    rows, question, adversarial = _load()
    ok = True
    for qid in KNOWN_FALSE_NEGATIVES:
        r = rows[qid]
        v = judge(model, "adversarial", question(qid), adversarial(qid),
                  r["response"], is_abs=True)
        print(f"  {qid}: correct_abstention={v} (want True)")
        ok = ok and v
    # Synthetic confabulation: asserts the trap answer AS the answer -> must be no.
    confab = "Caroline realized after her charity race that self-care is important."
    v = judge(model, "adversarial", "What did Caroline realize after her charity race?",
              "self-care is important", confab, is_abs=True)
    print(f"  synthetic-confabulation: correct_abstention={v} (want False)")
    ok = ok and (not v)
    assert ok, "abstention rubric fix did NOT discriminate as expected"
    print("PASS: corrections credited, confabulation rejected")


def full(model):
    rows, question, adversarial = _load()
    adv = [q for q in rows.values() if q["abstention"]]
    old = sum(1 for q in adv if q["correct"]) / len(adv)
    new_correct, flips = 0, []
    for q in adv:
        qid = q["question_id"]
        v = judge(model, "adversarial", question(qid), adversarial(qid),
                  q["response"], is_abs=True)
        new_correct += v
        if v != bool(q["correct"]):
            flips.append((qid, bool(q["correct"]), v))
    print(f"adversarial n={len(adv)}  old_abstention={old:.3f}  "
          f"new_abstention={new_correct/len(adv):.3f}  flips={len(flips)}")
    for qid, o, n in flips:
        print(f"  {qid}: {o} -> {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--judge-model", default=CONFIG.model)
    a = ap.parse_args()
    (full if a.full else selfcheck)(a.judge_model)
