# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Frozen-corpus characterization of E(vault) (docs IV.1).

E's unit tests prove only trivial monotonicity (one orphan raises E). This
bench answers the two questions the doc leaves open, on a frozen synthetic
graph, at zero LLM/API cost — and formalizes what it found:

  1. SIGN-CORRECTNESS per term. On single-term edits E moves the right way:
     resolving an orphan or strengthening a cluster lowers E; orphaning a note,
     breaking a link, weakening a cluster, contesting or under-linking a note
     raises it. (test_single_term_edits_have_correct_sign)

  2. THE GAP TERM IS SIZE-INVARIANT (fixed 2026-07-19). E sums gap_density =
     1 - inter/(size_a*size_b) ∈ [0,1), not the size-scaling gap_score. So an
     improving edit that enlarges a cluster (resolving an orphan INTO it,
     materializing a missing note) now LOWERS E, and the gap term no longer
     dwarfs cohesion. (test_gap_term_no_longer_size_pathological,
     test_unit_weights_gap_commensurate_with_cohesion.) The prior form made E
     reward fragmentation over growth; these tests are the regression that pins
     the fix — revert the density and they trip.

Driver-free (nodes/edges override): the override path does NOT populate the
dangling term, so dangling/contested/deficit signs are checked by report-level
injection instead.
"""
from __future__ import annotations

from silica.kernel.graph_report import compute_report
from silica.kernel.graph_report.models import (
    ContestedNote,
    IntegrationDeficit,
    VaultReport,
)
from silica.kernel.vault_energy import Weights, vault_energy


def _clique(members: list[str]) -> list[dict]:
    return [
        {"from": i, "to": j, "type": "EXTRACTED"}
        for i in members for j in members if i != j
    ]


def _E(nodes, edges, weights=Weights()):
    return vault_energy(
        compute_report(analytics=True, _nodes_edges_override=(nodes, edges)), weights
    )


A = ["a1", "a2", "a3", "a4"]
B = ["b1", "b2", "b3", "b4"]

# --- Base 1: ONE clique + one isolated orphan. No gap → isolates the orphan
# and cohesion terms so their sign is read without gap-term interference.
ONE_NODES = [{"id": n, "label": n} for n in A + ["orph"]]
ONE_EDGES = _clique(A)

# --- Base 2: TWO cliques bridged by one edge → a structural gap, plus orphan.
TWO_NODES = [{"id": n, "label": n} for n in A + B + ["orph"]]
TWO_EDGES = _clique(A) + _clique(B) + [{"from": "a1", "to": "b1", "type": "EXTRACTED"}]


def test_single_term_edits_have_correct_sign():
    """On a gap-free graph every controlled edit moves E the right way."""
    base = _E(ONE_NODES, ONE_EDGES).total

    # improving: link the orphan into the cluster → E down
    resolved = _E(ONE_NODES, ONE_EDGES + [{"from": "a1", "to": "orph", "type": "EXTRACTED"}])
    assert resolved.total < base

    # degrading: drop an intra-cluster edge (lower cohesion) → E up
    weaker = _E(ONE_NODES, [e for e in ONE_EDGES if {e["from"], e["to"]} != {"a1", "a2"}])
    assert weaker.total > base

    # degrading: add an isolated note (new orphan) → E up
    orphaned = _E(ONE_NODES + [{"id": "orph2", "label": "orph2"}], ONE_EDGES)
    assert orphaned.total > base


def _vr(**over) -> VaultReport:
    base = dict(generated_at="", scope="", totals={}, god_nodes=[], bridges=[],
                orphans=[], dangling=[], clusters=[])
    base.update(over)
    return VaultReport(**base)


def test_injected_terms_have_correct_sign():
    """dangling/contested/deficit have no override-path graph representation;
    inject them straight onto a report and confirm each raises E."""
    empty = vault_energy(_vr()).total
    assert empty == 0.0
    for field in (
        {"dangling": [{"target": "ghost", "refs": 1}]},
        {"contested": [ContestedNote("c", [])]},
        {"integration_deficits": [IntegrationDeficit("n", 9, 1, score=4.5)]},
    ):
        assert vault_energy(_vr(**field)).total > empty


def test_gap_term_no_longer_size_pathological():
    """FIXED (docs IV.1 calibration): E sums the bounded gap_density, so with a
    structural gap present, resolving an orphan by linking it into a cluster now
    LOWERS E — enlarging the cluster barely moves the density while the orphan
    term drops by 1. Reverting to gap_score trips this."""
    base = _E(TWO_NODES, TWO_EDGES).total
    resolved = _E(
        TWO_NODES, TWO_EDGES + [{"from": "a1", "to": "orph", "type": "EXTRACTED"}]
    ).total
    assert resolved < base  # improving edit now reads as improving

    # Adding a real bridge still lowers the gap (density falls).
    bridged = _E(TWO_NODES, TWO_EDGES + [{"from": "a4", "to": "b4", "type": "EXTRACTED"}]).total
    assert bridged < base


def test_unit_weights_gap_commensurate_with_cohesion():
    """At unit weights the bounded gap term (∈[0,1) per gap) is now the same
    order of magnitude as cohesion, so E's per-term decomposition is meaningful
    without hand-tuned weights."""
    e = _E(TWO_NODES, TWO_EDGES)
    assert 0 < e.gaps < 1.0          # one gap, bounded density
    assert e.cohesion < 0
    assert abs(e.gaps) <= abs(e.cohesion)  # no longer swamps


def _report_row(nodes, edges):
    e = _E(nodes, edges)
    return (f"E={e.total:+.3f} [coh {e.cohesion:+.2f} orph {e.orphans:+.2f} "
            f"dang {e.dangling:+.2f} gap {e.gaps:+.2f} def {e.deficits:+.2f} "
            f"con {e.contested:+.2f}]")


if __name__ == "__main__":
    print("Base1 (one clique + orphan): ", _report_row(ONE_NODES, ONE_EDGES))
    print("Base2 (two cliques + gap):   ", _report_row(TWO_NODES, TWO_EDGES))
    resolved = TWO_EDGES + [{"from": "a1", "to": "orph", "type": "EXTRACTED"}]
    print("Base2 + resolve orphan:      ", _report_row(TWO_NODES, resolved),
          " <-- improving edit, E now FALLS (gap term fixed)")
