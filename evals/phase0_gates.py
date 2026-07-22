# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Phase-0 calibration run for the retrieval gates (spec 2026-07-14).

Collects per-query gate signals through the pre-registered hooks
(COOCCUR_GATE_PROBE, RERANK_GATE_PROBE) across the corpora available on disk:

  lme_s  — 46-q stratified LongMemEval_s sample on the frozen bench/lme_s_raw
           vaults (chat sessions: the corpus where gate 2b must fire)
  vault  — fusion-probe masked pairs + production-mirror rerank pass on the
           real vault (the cooccur leg's home turf: both gates must stay silent)

MuSiQue — the corpus where gate 1 (cooccur confidence) is supposed to fire —
is no longer on disk (its bench vault lived in a cleaned scratchpad), so
_COOCCUR_MIN_CONFIDENCE stays dormant at 0.0; this run records the no-fire
side's coverage/flatness distributions as the reference for a future re-run.

The reranker SERVICE need not be running: both hooks fire before the HTTP
call, and a dead endpoint only degrades the (ignored) arm-B ordering.

  uv run python -m evals.phase0_gates --vault ~/Documents/Obsidian/test
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

_FACTORS = (2, 3, 4, 5, 8, 10)          # candidate _RERANK_WINDOW_FACTOR values
_COVERAGE_REFS = (0.02, 0.05, 0.1, 0.2)  # reference thresholds for gate 1

_REPO = Path(__file__).resolve().parents[1]
OUT_PATH = _REPO / "bench" / "phase0_gates.json"


def _pcts(vals: list[float]) -> dict[str, float]:
    """Nearest-rank percentiles — deterministic, no interpolation."""
    s = sorted(vals)

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]

    return {"min": s[0], "p10": q(.10), "p25": q(.25), "p50": q(.50),
            "p75": q(.75), "p90": q(.90), "max": s[-1]}


def summarize(rows: list[dict]) -> dict:
    """Per (corpus, gate) signal distributions + fire-rates at candidate thresholds."""
    out: dict[str, dict] = {}
    for corpus in sorted({r["corpus"] for r in rows}):
        c: dict[str, Any] = {}
        rr = [r for r in rows if r["corpus"] == corpus and r["gate"] == "rerank"]
        if rr:
            ratios = [r["median_len"] / r["window"] for r in rr]
            c["rerank"] = {
                "n": len(rr),
                "ratio": _pcts(ratios),
                "fire_rate": {str(f): round(sum(1 for x in ratios if x > f) / len(ratios), 4)
                              for f in _FACTORS},
            }
        cc = [r for r in rows if r["corpus"] == corpus and r["gate"] == "cooccur"]
        if cc:
            c["cooccur"] = {
                "n": len(cc),
                "coverage": _pcts([r["coverage"] for r in cc]),
                "flatness": _pcts([r["flatness"] for r in cc]),
                "coverage_fire_rate": {str(t): round(sum(1 for r in cc if r["coverage"] < t) / len(cc), 4)
                                       for t in _COVERAGE_REFS},
            }
        out[corpus] = c
    return out


def separation(summary: dict, *, fire: str = "lme_s", silent: str = "vault") -> dict:
    """Gate 2b freeze criterion: worst-case fire-side ratio (p10) against
    worst-case silent-side ratio (p90). Order-of-magnitude gap = freezable."""
    lo = summary[fire]["rerank"]["ratio"]["p10"]
    hi = summary[silent]["rerank"]["ratio"]["p90"]
    return {"fire_p10": lo, "silent_p90": hi,
            "gap": round(lo / hi, 2) if hi > 0 else float("inf")}


def main(argv=None) -> int:
    # Before any silica import: CONFIG reads env at import time, and the gate
    # probe only fires when a reranker CLIENT exists (liveness irrelevant).
    os.environ.setdefault("SILICA_RERANK_BASE_URL", "http://localhost:1235/v1")
    os.environ.setdefault("SILICA_RERANK_MODEL", "bge-reranker-v2-m3")

    ap = argparse.ArgumentParser(prog="python -m evals.phase0_gates")
    ap.add_argument("--vault", required=True, help="real vault (fusion-probe side)")
    ap.add_argument("--lme-data", default=str(_REPO / "docs/eval-handoff/lme_s_sample.json"))
    ap.add_argument("--lme-root", default=str(_REPO / "bench/lme_s_raw"))
    ap.add_argument("--k", type=int, default=15, help="facade top-k on the LME arm")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    from silica.kernel import relatedness, rerank

    rows: list[dict] = []
    ctx = {"corpus": "", "qid": ""}
    relatedness.COOCCUR_GATE_PROBE = lambda sig: rows.append({"gate": "cooccur", **ctx, **sig})
    rerank.RERANK_GATE_PROBE = lambda sig: rows.append({"gate": "rerank", **ctx, **sig})

    # --- corpus 1: LME_s sample, retrieval-only on frozen vaults -------------
    from evals.longmemeval.runner import run_instance

    sample = json.loads(Path(args.lme_data).read_text(encoding="utf-8"))
    if args.limit:
        sample = sample[: args.limit]
    ctx["corpus"] = "lme_s"
    srs: list[float] = []
    for i, inst in enumerate(sample):
        ctx["qid"] = inst["question_id"]
        row = run_instance(inst, Path(args.lme_root), model="", judge_model="",
                           k=args.k, stuff=False, use_embedder=True, use_rerank=True,
                           retrieval_only=True, reuse=True)
        if row["session_recall"] is not None:
            srs.append(row["session_recall"])
        if args.verbose:
            print(f"  [{i + 1}/{len(sample)}] {inst['question_id']} sr={row['session_recall']}")
    mean_sr = round(sum(srs) / len(srs), 4) if srs else None
    print(f"lme_s: {len(sample)} questions, mean session_recall {mean_sr}")

    # --- corpus 2: real vault — fusion probe + production-mirror rerank ------
    ctx.update(corpus="vault", qid="")
    from silica.agent.providers import get_reranker
    from silica.config import CONFIG
    from silica.kernel.health import fusion_probe
    from evals.golden import probe_fusion
    from evals.golden.runner import _open_stores, resolve_vault

    vault = resolve_vault(args.vault)
    store, embed_store = _open_stores(vault)
    fz = fusion_probe(vault, store, embed_store=embed_store, verbose=args.verbose)
    print(f"vault: fusion recall@10 {fz['recall_at_10']} (frozen baseline 0.73 must hold)")
    probe_fusion.run_rerank_ab(vault, store, embed_store=embed_store,
                               reranker=get_reranker(CONFIG), verbose=args.verbose)

    # --- report ---------------------------------------------------------------
    summary = summarize(rows)
    sep = separation(summary)
    report = {
        "spec": "2026-07-14-retrieval-gates-design.md phase 0",
        "config": {"k": args.k, "lme_questions": len(sample), "lme_mean_sr": mean_sr,
                   "fusion_recall_at_10": fz["recall_at_10"],
                   "musique": "absent — gate 1 stays dormant (0.0)"},
        "summary": summary,
        "separation_2b": sep,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n{len(rows)} probe rows -> {args.out}")
    for corpus, gates in summary.items():
        for gate, s in gates.items():
            sig = "ratio" if gate == "rerank" else "coverage"
            p = s[sig]
            print(f"  {corpus:<7} {gate:<8} n={s['n']:<5} {sig}: "
                  f"p10 {p['p10']:.3g}  p50 {p['p50']:.3g}  p90 {p['p90']:.3g}")
    print(f"\ngate 2b separation: fire p10 {sep['fire_p10']:.2f} vs silent p90 "
          f"{sep['silent_p90']:.2f} -> gap {sep['gap']}x")
    factor = rerank._RERANK_WINDOW_FACTOR
    inside = sep["silent_p90"] < factor < sep["fire_p10"]
    print(f"current _RERANK_WINDOW_FACTOR={factor}: "
          + ("inside the gap — freezable" if inside else "OUTSIDE the gap — do not freeze"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
