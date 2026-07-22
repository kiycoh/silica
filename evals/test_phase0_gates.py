# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""summarize()/separation() arithmetic on hand-built probe rows (phase-0,
retrieval-gates spec 2026-07-14)."""
from evals.phase0_gates import separation, summarize


def _rr(corpus: str, median_len: float, window: int = 800) -> dict:
    return {"gate": "rerank", "corpus": corpus, "qid": "q",
            "median_len": median_len, "window": window,
            "fired": median_len > 3 * window}


def _cc(corpus: str, coverage: float, flatness: float) -> dict:
    return {"gate": "cooccur", "corpus": corpus, "qid": "q",
            "coverage": coverage, "flatness": flatness, "fired": False}


def test_summarize_and_separation() -> None:
    rows = ([_rr("lme_s", 8000), _rr("lme_s", 16000), _rr("lme_s", 24000)]
            + [_rr("vault", 400), _rr("vault", 800), _rr("vault", 1600)]
            + [_cc("vault", 0.01, 1.0), _cc("vault", 0.5, 3.0)])
    s = summarize(rows)

    assert s["lme_s"]["rerank"]["n"] == 3
    assert s["lme_s"]["rerank"]["ratio"]["p50"] == 20.0          # ratios 10/20/30
    assert s["lme_s"]["rerank"]["fire_rate"]["3"] == 1.0
    assert s["vault"]["rerank"]["fire_rate"]["3"] == 0.0         # ratios .5/1/2
    assert "rerank" not in s.get("lme_s", {}).get("cooccur", {})  # gates kept apart
    assert s["vault"]["cooccur"]["n"] == 2
    assert s["vault"]["cooccur"]["coverage_fire_rate"]["0.05"] == 0.5
    assert s["vault"]["cooccur"]["flatness"]["max"] == 3.0

    sep = separation(s)
    assert sep["fire_p10"] == 10.0
    assert sep["silent_p90"] == 2.0
    assert sep["gap"] == 5.0
