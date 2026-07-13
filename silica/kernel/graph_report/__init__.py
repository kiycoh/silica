# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L1 Graph Report — deterministic structural audit of the vault.

Builds a VaultReport from the driver's wikilink graph using only networkx
and the existing graph_export helpers. No LLM calls, no network access.

Principle: "embeddings PROPOSE, graph DISPOSES" — the report is authoritative
over vault structure; missing_links (embeddings) are clearly separated and
labelled as proposed candidates.

Paper-inspired signals (Marwitz et al., Nature Mach. Intell. 2026):
  - common_neighbors: structural likelihood boost (2-length path count) so
    proposed links sharing more neighbours with the source rank higher.
  - d_prev: shortest path distance before a link is predicted, annotated per
    MissingLink so downstream consumers can differentiate likely (d=2) from
    novel (d≥3) candidates.
  - Temporal decay: EmbedStore timestamps boost recent note pairs.
  - Cosine-band filtering: autolink candidates semantically too similar or
    too alien are suppressed when embeddings are available.

Package layout: models (dataclasses), compute (compute_report core),
embed_signals (embedding proposals), cooccur_delta (co-occurrence vs
wikilink delta — ADR-0013 CORRELATE lands there), render (output).
This façade re-exports the full historical module surface; consumers
import from silica.kernel.graph_report only.
"""
from __future__ import annotations

from silica.kernel.graph_report.compute import _empty_report, compute_report
from silica.kernel.graph_report.cooccur_delta import _compute_cooccur_delta
from silica.kernel.graph_report.embed_signals import (
    _compute_duplicate_pairs,
    _compute_missing_links,
)
from silica.kernel.graph_report.models import (
    AutolinkCandidate,
    BridgeStat,
    ClusterStat,
    ContestedNote,
    DuplicatePair,
    IntegrationDeficit,
    MissingHub,
    MissingLink,
    NodeStat,
    SourceDrift,
    StaleLink,
    VaultReport,
)
from silica.kernel.graph_report.render import to_digest, to_facts, to_markdown, write_report

__all__ = [
    "AutolinkCandidate",
    "BridgeStat",
    "ClusterStat",
    "ContestedNote",
    "DuplicatePair",
    "IntegrationDeficit",
    "MissingHub",
    "MissingLink",
    "NodeStat",
    "SourceDrift",
    "StaleLink",
    "VaultReport",
    "compute_report",
    "to_digest",
    "to_facts",
    "to_markdown",
    "write_report",
]
