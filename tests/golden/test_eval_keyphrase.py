"""Eval harness — keyphrase extraction recall/precision on a real corpus.

Mirrors test_eval_autolink: golden cases in keyphrase_cases.json, one hard
per-case gate (must_appear) + one aggregate measurement that prints per-genre
recall/precision and soft-gates. Per-genre recall IS the Fase 1 -> Fase 2 gate:
high paper recall = the thesis holds; collapsing lecture recall = evidence the
structural leg (Fase 2) is worth building.

Concept matching is fuzzy (stem-overlap), since "rete neurale" / "reti neurali"
/ "della rete neurale" are the same concept.

Run with: uv run pytest tests/golden/test_eval_keyphrase.py -v -s
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

from silica.kernel.cooccurrence import tokenize
from silica.kernel.keyphrase import extract_keyphrases

GOLDEN_PATH = Path(__file__).parent / "keyphrase_cases.json"
_EXAMPLE_OVERLAYS = Path(__file__).resolve().parent.parent.parent / "examples" / "overlays"

# Case schema (populate keyphrase_cases.json with real vault excerpts):
#   {"id": str, "genre": "paper"|"lecture", "lang": "italian",
#    "body": "<excerpt, no need to strip markup — keyphrase strips frontmatter>",
#    "gold": ["concept", ...],          # hand-labelled concepts that SHOULD surface
#    "must_appear": ["concept", ...]}   # 2-3 unmistakable ones (hard per-case gate)


def _stems(phrase: str, lang: str) -> frozenset[str]:
    """Content stems of *phrase* (stopwords / short tokens dropped by tokenize)."""
    return frozenset(stem for sent in tokenize(phrase, lang=lang) for stem, _surface in sent)


def concept_recalled(gold: str, extracted: list[str], lang: str) -> bool:
    """True if some extracted phrase *contains* the gold concept at stem level."""
    g = _stems(gold, lang)
    if not g:
        return False
    return any(g <= _stems(e, lang) for e in extracted)


# ---------------------------------------------------------------------------
# Matcher unit tests (no corpus needed)
# ---------------------------------------------------------------------------

class TestConceptRecalled:
    def test_morphological_and_inflected_variant_matches(self):
        extracted = ["della rete neurale profonda", "tasso di apprendimento"]
        assert concept_recalled("rete neurale", extracted, "italian")
        assert concept_recalled("reti neurali", extracted, "italian")  # plural gold

    def test_stopwords_in_gold_ignored(self):
        extracted = ["meccanismo attenzione"]
        assert concept_recalled("meccanismo di attenzione", extracted, "italian")

    def test_absent_concept_not_recalled(self):
        extracted = ["rete neurale", "funzione di perdita"]
        assert not concept_recalled("backpropagation", extracted, "italian")

    def test_partial_overlap_is_not_a_match(self):
        # "rete" alone must not satisfy the 2-word concept "rete neurale"
        assert not concept_recalled("rete neurale", ["rete stradale"], "italian")


# ---------------------------------------------------------------------------
# Corpus harness (skips until keyphrase_cases.json is populated)
# ---------------------------------------------------------------------------

def _load_cases() -> list[dict]:
    if not GOLDEN_PATH.exists():
        return []
    return json.loads(GOLDEN_PATH.read_text())


def _overlay():
    from silica.kernel.overlay import DEFAULT_OVERLAY, load_overlay
    path = _EXAMPLE_OVERLAYS / "it-academic.yaml"
    return load_overlay(path) if path.exists() else DEFAULT_OVERLAY


def _maybe_embedder():
    """Real embedder if the endpoint answers, else None (eval degrades to YAKE rank)."""
    try:
        from silica.agent.providers import get_embedder
        from silica.config import CONFIG
        emb = get_embedder(CONFIG)
        emb.embed(["probe"])  # force a real call so a dead endpoint => None
        return emb
    except Exception:
        return None


_EMBEDDER = _maybe_embedder()


def _extract(case) -> list[str]:
    cands = extract_keyphrases(case["body"], overlay=_overlay(),
                               lang=case["lang"], embedder=_EMBEDDER)
    return [c.phrase for c in cands]


_CASES = _load_cases()


@pytest.mark.skipif(not _CASES, reason="no keyphrase_cases.json corpus yet")
@pytest.mark.skipif(not os.getenv("SILICA_EVAL"), reason="slow (real embedder); set SILICA_EVAL=1 to run")
def test_recall_precision_by_genre():
    """Measurement tool (not a CI gate): PRINT per-genre recall/precision + must_appear
    misses. This is the Fase 1->2 calibration loop — read the printout with `-s`.

    Asserts only against catastrophic failure (zero extraction = broken pipeline).
    The real signal is the numbers: high paper recall = YAKE recovers prose concepts;
    a collapsing lecture recall = the structural leg (Fase 2) earns its keep.
    """
    by_genre: dict[str, list[float]] = defaultdict(list)
    prec_by_genre: dict[str, list[float]] = defaultdict(list)

    for case in _CASES:
        phrases = _extract(case)
        gold = case.get("gold", [])
        lang = case["lang"]
        hits = sum(concept_recalled(g, phrases, lang) for g in gold)
        recall = hits / len(gold) if gold else 1.0
        precise = sum(any(concept_recalled(g, [p], lang) for g in gold) for p in phrases)
        precision = precise / len(phrases) if phrases else 1.0
        misses = [c for c in case.get("must_appear", []) if not concept_recalled(c, phrases, lang)]
        by_genre[case["genre"]].append(recall)
        prec_by_genre[case["genre"]].append(precision)
        print(f"\n[{case['id']:<24} {case['genre']:<8}] recall={recall:.0%} precision={precision:.0%} "
              f"({hits}/{len(gold)} gold, {len(phrases)} extracted)")
        if misses:
            print(f"    must_appear MISSED: {misses}")

    for genre in sorted(by_genre):
        r = sum(by_genre[genre]) / len(by_genre[genre])
        p = sum(prec_by_genre[genre]) / len(prec_by_genre[genre])
        print(f"[GENRE {genre:<8}] mean recall={r:.0%}  mean precision={p:.0%}")

    for case in _CASES:  # catastrophic guard only: every doc must yield SOMETHING
        assert _extract(case), f"[{case['id']}] zero concepts — pipeline broken"
