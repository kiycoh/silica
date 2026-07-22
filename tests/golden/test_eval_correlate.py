# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Unit tests for probe_correlate — masked-pair recovery lift (ADR-0013).

The lift's real magnitude is a corpus property (the spec measured +47 on the
golden vault); these pin the probe's mechanics on synthetic vaults: recovery of
an eligible linked pair, the >2-hop eligibility filter, proxy-precision counting,
and the union >= expanded invariant.
"""
from __future__ import annotations

from silica.kernel.cooccurrence import CooccurStore, build_contribution
from evals.golden import probe_correlate


def _store(tmp_path, notes: dict[str, str]) -> CooccurStore:
    st = CooccurStore(path=tmp_path / "idx" / "c.json", lang="english")
    for key, body in notes.items():
        st.upsert_note(key, build_contribution(key, body))
    return st


def test_probe_empty_store_is_zeros(tmp_path):
    st = CooccurStore(path=tmp_path / "idx" / "c.json", lang="english")
    res = probe_correlate.run(tmp_path, st)
    assert res["pairs_evaluated"] == 0 and res["lift"] == 0.0 and res["edges"] == 0


def test_probe_recovers_eligible_linked_pair(tmp_path):
    (tmp_path / "A.md").write_text("quick sort compares array elements\n[[B]]")
    (tmp_path / "B.md").write_text("quick sort swaps array elements")
    st = _store(tmp_path, {
        "A": "quick sort compares array elements",
        "B": "quick sort swaps array elements",
    })
    res = probe_correlate.run(tmp_path, st)
    assert res["pairs_evaluated"] == 1          # A-B, eligible after masking the link
    assert res["edges"] >= 1                     # A-B is a direct edge (jaccard >= tau)
    assert res["recall_union"] >= res["recall_expanded"]  # union never below expanded
    assert res["recall_union"] == 1.0            # the pair is recovered


def test_probe_skips_pairs_reachable_via_shared_hub(tmp_path):
    # Triangle A-B-H: every pair has the third as a common neighbour, so masking
    # any edge still leaves a 2-hop path -> none is a fair recovery test.
    (tmp_path / "A.md").write_text("quick sort array\n[[B]]\n[[H]]")
    (tmp_path / "B.md").write_text("quick sort heap\n[[A]]\n[[H]]")
    (tmp_path / "H.md").write_text("quick sort tree\n[[A]]\n[[B]]")
    st = _store(tmp_path, {
        "A": "quick sort array", "B": "quick sort heap", "H": "quick sort tree",
    })
    res = probe_correlate.run(tmp_path, st)
    assert res["pairs_evaluated"] == 0           # all pairs suppressed by the hub


def test_probe_proxy_precision_counts_unlinked_edges(tmp_path):
    # A-B, A-C, B-C are all text-similar (direct edges) but only A-B is wikilinked,
    # so the fraction of edges already wikilinked is strictly between 0 and 1.
    (tmp_path / "A.md").write_text("quick sort array\n[[B]]")
    (tmp_path / "B.md").write_text("quick sort heap")
    (tmp_path / "C.md").write_text("quick sort tree")
    st = _store(tmp_path, {
        "A": "quick sort array", "B": "quick sort heap", "C": "quick sort tree",
    })
    res = probe_correlate.run(tmp_path, st)
    assert res["edges"] >= 2
    assert 0.0 < res["edges_wikilinked_frac"] < 1.0
