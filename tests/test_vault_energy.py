# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""E(vault) is a pure composition of an existing VaultReport (docs IV.1)."""
from __future__ import annotations

from silica.kernel.graph_report import compute_report
from silica.kernel.graph_report.models import (
    ClusterStat,
    ContestedNote,
    IntegrationDeficit,
    StructuralGap,
    VaultReport,
)
from tests.eval.vault_energy import Weights, vault_energy


def _report(**over) -> VaultReport:
    base = dict(
        generated_at="", scope="", totals={}, god_nodes=[], bridges=[],
        orphans=[], dangling=[], clusters=[],
    )
    base.update(over)
    return VaultReport(**base)


def test_terms_sum_to_total():
    r = _report(
        clusters=[ClusterStat(0, 3, "h", [], cohesion=0.5)],
        orphans=["a", "b"],
        dangling=[{"target": "x", "refs": 1}],
        structural_gaps=[StructuralGap(0, 1, "a", "b", 0, gap_score=4.0, gap_density=0.75)],
        integration_deficits=[IntegrationDeficit("n", 6, 1, score=3.0)],
        contested=[ContestedNote("c", [])],
    )
    e = vault_energy(r)
    assert e.total == e.cohesion + e.orphans + e.dangling + e.gaps + e.deficits + e.contested
    assert e.cohesion == -0.5  # only negative term
    assert e.orphans == 2 and e.dangling == 1 and e.contested == 1
    assert e.gaps == 0.75 and e.deficits == 3.0  # gaps sums bounded density, not gap_score


def test_cohesion_lowers_energy_others_raise_it():
    empty = vault_energy(_report()).total
    assert empty == 0.0
    # bonds pull E down
    assert vault_energy(_report(clusters=[ClusterStat(0, 2, "h", [], 0.9)])).total < empty
    # every entropic term pushes E up
    assert vault_energy(_report(orphans=["a"])).total > empty
    assert vault_energy(_report(contested=[ContestedNote("c", [])])).total > empty


def test_weights_scale_their_term():
    r = _report(orphans=["a", "b", "c"])
    assert vault_energy(r, Weights(orphans=2.0)).total == 6.0
    assert vault_energy(r, Weights(orphans=0.0)).total == 0.0


def test_compute_report_feeds_vault_energy():
    """The harness handoff: a real compute_report output has every field E reads
    (right names, right depth). Override path = no live driver needed."""
    nodes = [{"id": n, "label": n} for n in ("a", "b", "c")]
    edges = [{"from": "a", "to": "b", "type": "EXTRACTED"}]
    e = vault_energy(compute_report(analytics=True, _nodes_edges_override=(nodes, edges)))
    assert isinstance(e.total, float)
    assert e.orphans >= 2.0  # a and c have in-degree 0
