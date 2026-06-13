"""Dataclasses for the L1 graph report.

Pure data containers — no I/O, no graph computation. Authoritative
structures (NodeStat … VaultReport) and PROPOSED-signal records
(MissingLink, DuplicatePair, AutolinkCandidate, StaleLink, MissingHub)
live together because they all describe one VaultReport payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeStat:
    id: str           # vault-relative path (no .md)
    label: str        # display name
    cluster: int      # node["group"], -1 if none
    out_degree: int
    in_degree: int
    degree: int       # out+in
    pagerank: float   # rounded to 5 decimal places


@dataclass
class BridgeStat:
    source: str
    target: str
    source_cluster: int
    target_cluster: int
    weight: float     # surprise score: (deg(u)+deg(v)) / (1 + shared_neighbors)


@dataclass
class ClusterStat:
    cluster_id: int
    size: int
    hub: str | None        # highest-degree node in cluster
    members: list[str]     # capped at 25 in markdown, full in JSON
    cohesion: float        # intra-cluster edges / C(size,2)


@dataclass
class MissingLink:          # PROPOSED — not authoritative
    source: str
    target: str
    cosine: float
    d_prev: int = 0         # shortest path before prediction (0 = unreachable → highest novelty)


@dataclass
class DuplicatePair:        # PROPOSED — borderline duplicates
    source: str
    target: str
    score: float


# --- Co-occurrence vs wikilink delta (PROPOSED, embedder-free) -------------

@dataclass
class AutolinkCandidate:    # co-occurrence − wikilink: related in text, unlinked
    source: str
    target: str
    weight: float          # co-occurrence relatedness weight (higher = stronger)
    shared: list[str]      # directly shared concept labels (evidence)
    convergence: int = 0   # #8: number of god-node hubs this pair connects to


@dataclass
class StaleLink:           # wikilink − co-occurrence: linked, no textual co-presence
    source: str
    target: str


@dataclass
class MissingHub:          # central concept in the discourse with no hub note
    concept: str           # surface label of the concept
    centrality: float      # weighted degree in the co-occurrence graph


@dataclass
class VaultReport:
    generated_at: str
    scope: str
    totals: dict[str, int]
    god_nodes: list[NodeStat]
    bridges: list[BridgeStat]
    orphans: list[str]
    dangling: list[dict]   # [{"target": str, "refs": int}]
    clusters: list[ClusterStat]
    missing_links: list[MissingLink] = field(default_factory=list)
    duplicate_pairs: list[DuplicatePair] = field(default_factory=list)
    autolink_candidates: list[AutolinkCandidate] = field(default_factory=list)
    stale_links: list[StaleLink] = field(default_factory=list)
    missing_hubs: list[MissingHub] = field(default_factory=list)
    lean_notes: list[str] = field(default_factory=list)
    reformat_notes: list[str] = field(default_factory=list)
    pagerank_map: dict[str, float] = field(default_factory=dict)  # all nodes: vault-relative path (no .md) → pagerank
