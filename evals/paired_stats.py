# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Paired significance test over two benchmark metrics.json files.

The suite eyeballs deltas against ad hoc question-count bands (audit lane 2): at
n=18-199 a +0.02 delta is inside one SE. Everything a paired test needs is
already persisted — each metrics.json stores per-question rows with
``question_id`` and ``correct`` — so this is the missing ~tooling, not missing
data.

Two arms, paired on the questions both graded (correct in {true, false}):

  * McNemar exact — a two-sided binomial on the discordant pairs (A right/B
    wrong vs A wrong/B right). The exact test, not the chi-square approximation,
    because discordant counts here are small.
  * Bootstrap 95% CI on the accuracy delta (accA - accB), seeded so the interval
    is reproducible.

  uv run python -m evals.paired_stats bench/armA.metrics.json bench/armB.metrics.json
  uv run python -m evals.paired_stats A.json B.json --include-abstention
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scipy.stats import binomtest


def _correct_by_qid(doc: dict, *, include_abstention: bool) -> dict[str, bool]:
    """{question_id: correct} over graded rows (correct is not None). Abstention
    rows are excluded by default, matching overall_accuracy's denominator."""
    out: dict[str, bool] = {}
    for r in doc.get("questions", []):
        if r.get("correct") is None:
            continue
        if r.get("abstention") and not include_abstention:
            continue
        out[r["question_id"]] = bool(r["correct"])
    return out


def paired(a: dict, b: dict, *, include_abstention: bool = False,
           iters: int = 10000, seed: int = 42) -> dict:
    ca = _correct_by_qid(a, include_abstention=include_abstention)
    cb = _correct_by_qid(b, include_abstention=include_abstention)
    qids = sorted(set(ca) & set(cb))
    pairs = [(ca[q], cb[q]) for q in qids]
    n = len(pairs)
    b_wins = sum(1 for x, y in pairs if x and not y)   # A right, B wrong
    c_wins = sum(1 for x, y in pairs if y and not x)   # A wrong, B right
    disc = b_wins + c_wins
    # Exact two-sided McNemar = binomial(disc, 0.5) on the winner split.
    p_value = binomtest(b_wins, disc, 0.5).pvalue if disc else 1.0

    delta = (sum(x for x, _ in pairs) - sum(y for _, y in pairs)) / n if n else 0.0
    rng = random.Random(seed)
    boot = []
    for _ in range(iters):
        s = [pairs[rng.randrange(n)] for _ in range(n)] if n else []
        boot.append((sum(x for x, _ in s) - sum(y for _, y in s)) / n if n else 0.0)
    boot.sort()
    lo = boot[int(0.025 * iters)] if boot else 0.0
    hi = boot[int(0.975 * iters)] if boot else 0.0
    return {
        "n_paired": n,
        "acc_a": round(sum(x for x, _ in pairs) / n, 4) if n else None,
        "acc_b": round(sum(y for _, y in pairs) / n, 4) if n else None,
        "delta": round(delta, 4),
        "delta_ci95": [round(lo, 4), round(hi, 4)],
        "discordant": {"a_only": b_wins, "b_only": c_wins},
        "mcnemar_p": round(p_value, 4),
        "significant_05": bool(p_value < 0.05),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals.paired_stats")
    ap.add_argument("a", help="metrics.json for arm A")
    ap.add_argument("b", help="metrics.json for arm B")
    ap.add_argument("--include-abstention", action="store_true",
                    help="pair abstention rows too (default: overall-accuracy set)")
    ap.add_argument("--iters", type=int, default=10000, help="bootstrap resamples")
    args = ap.parse_args(argv)

    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    res = paired(a, b, include_abstention=args.include_abstention, iters=args.iters)
    if res["n_paired"] == 0:
        print("no shared graded questions — nothing to compare "
              "(check question_id namespaces / dataset identity)")
        return 1
    print(f"paired on {res['n_paired']} questions")
    print(f"  acc A {res['acc_a']}   acc B {res['acc_b']}   "
          f"delta {res['delta']:+}  95% CI {res['delta_ci95']}")
    print(f"  discordant: A-only {res['discordant']['a_only']}  "
          f"B-only {res['discordant']['b_only']}")
    print(f"  McNemar exact p = {res['mcnemar_p']}  "
          f"({'significant' if res['significant_05'] else 'not significant'} at 0.05)")
    return 0


def demo() -> None:
    """ponytail: asserts the two failure paths the eyeball misses — a real gap
    is flagged significant, pure noise is not."""
    def doc(correct_by_q):
        return {"questions": [{"question_id": q, "correct": c, "abstention": False}
                              for q, c in correct_by_q.items()]}

    # 12 questions A gets right and B wrong, 0 the other way -> clearly significant.
    a = doc({f"q{i}": True for i in range(12)})
    b = doc({f"q{i}": False for i in range(12)})
    r = paired(a, b)
    assert r["n_paired"] == 12 and r["discordant"] == {"a_only": 12, "b_only": 0}
    assert r["delta"] == 1.0 and r["significant_05"], r

    # Symmetric disagreement (4 vs 4) -> delta 0, p = 1.0, not significant.
    a2 = doc({**{f"q{i}": True for i in range(4)}, **{f"q{i}": False for i in range(4, 8)}})
    b2 = doc({**{f"q{i}": False for i in range(4)}, **{f"q{i}": True for i in range(4, 8)}})
    r2 = paired(a2, b2)
    assert r2["delta"] == 0.0 and r2["mcnemar_p"] == 1.0 and not r2["significant_05"], r2

    # No shared qids -> empty, no crash.
    assert paired(doc({"x": True}), doc({"y": False}))["n_paired"] == 0
    print("ok")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "--demo":
        demo()
    else:
        raise SystemExit(main())
