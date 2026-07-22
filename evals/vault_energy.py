# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""E(vault): the lattice energy of the vault as a single scalar.

Formalises Part IV.1 of docs/Silica_x_chemistry.md. Every term already exists
as a field on VaultReport; this module only composes them. No new metric, no
new dependency, no LLM.

    E = -w1 * Σ cohesion(c)      # enthalpy: bonds formed, intra-cluster density
        + w2 * orphans          # entropic cost: unlinked matter
        + w3 * dangling         # broken bonds (links to non-existent notes)
        + w4 * Σ gap_density     # structural holes: absent-link fraction ∈ [0,1) per gap
        + w5 * Σ deficit.score  # concept-rich, weakly-linked notes (concepts/(1+degree))
        + w6 * contested        # unresolved polymorphs

Lower E = more coherent vault. The one negative term (cohesion) is the only
force that pulls E down; every other term is an entropic penalty.

E is a THERMOMETER, not an objective function (guardrail IV.5.1). Read it to
compare runs and to decide which gate lowered which term; never descend it with
a global optimiser. The vault is a dissipative steady state (Part III.3): it
improves per local event, not by chasing a global minimum.

Comparability caveat: cohesion and structural_gaps are only populated when the
report is built with full=True (compute_report skips them in the cheap nucleate
path). Compare E only across reports computed at the same depth.
"""
from __future__ import annotations

from dataclasses import dataclass

from silica.kernel.graph_report.models import VaultReport


@dataclass(frozen=True)
class Weights:
    """The six tuning knobs of E. Defaults = the doc's baseline (all unit).

    These are the common tuning surface the seven independent gate thresholds
    (0.85, 0.65, 0.80, 100, 0.6, 0.25, ...) lacked. Raw term scales differ
    wildly (cohesion ∈ [0, #clusters], gap_score can reach size_a*size_b), so
    balancing them is exactly what these weights are for — but that is a
    measured calibration, not a default. Ship unit weights; tune on the bench.
    """

    cohesion: float = 1.0
    orphans: float = 1.0
    dangling: float = 1.0
    gaps: float = 1.0
    deficits: float = 1.0
    contested: float = 1.0


@dataclass(frozen=True)
class VaultEnergy:
    """E and its six signed contributions. The contributions sum to `total`,
    so ΔE between two runs decomposes per term: which force moved the vault."""

    total: float
    cohesion: float   # negative contribution (enthalpy)
    orphans: float
    dangling: float
    gaps: float
    deficits: float
    contested: float


def vault_energy(report: VaultReport, weights: Weights = Weights()) -> VaultEnergy:
    """Compose a VaultReport into E(vault). Pure: reads fields, sums, returns."""
    cohesion = -weights.cohesion * sum(c.cohesion for c in report.clusters)
    orphans = weights.orphans * len(report.orphans)
    dangling = weights.dangling * len(report.dangling)
    gaps = weights.gaps * sum(g.gap_density for g in report.structural_gaps)
    deficits = weights.deficits * sum(d.score for d in report.integration_deficits)
    contested = weights.contested * len(report.contested)
    total = cohesion + orphans + dangling + gaps + deficits + contested
    return VaultEnergy(
        total=total,
        cohesion=cohesion,
        orphans=orphans,
        dangling=dangling,
        gaps=gaps,
        deficits=deficits,
        contested=contested,
    )
