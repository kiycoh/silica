# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L3 Analyst Plan — maps VaultReport anomalies to a three-tier task plan.

Translates structural anomalies into actionable TaskCandidates without executing
anything. The three confidence tiers enforce the rule:

  auto      — reversible, graph-safe by construction, unambiguous signal
  propose   — reversible but borderline/opinion-dependent → needs confirmation
  escalate  — irreversible or requires human judgment → IssueCard only

§3.2-bis invariant: capability_name in plan.auto must NEVER be an irreversible
tool (silica_merge, silica_move, silica_delete, etc.). Only silica_autolink
qualifies today — it is graph-safe by construction and fully reversible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

from silica.kernel.graph_report import AutolinkCandidate, MissingLink, VaultReport
from silica.kernel.progress import PlanStep

Tier = Literal["auto", "propose", "escalate"]

# Edge provenance vocabulary (ported from Graphify, MIT, © 2026 Safi Shamsi).
# The tier follows the confidence, so it is never a loose threshold buried in a
# call site — one mapping, one source of truth.
#   EXTRACTED  = corroborated by structure  → auto (reversible, unambiguous)
#   INFERRED   = single-signal / embedding  → propose (confirm before writing)
#   AMBIGUOUS  = conflicting / needs a human → escalate
Confidence = Literal["EXTRACTED", "INFERRED", "AMBIGUOUS"]

_MISSING_LINK_TAU_HIGH = 0.85  # mirrors CONFIG.sim_threshold_high default


def classify_missing_link(ml: "MissingLink", *, tau_high: float = _MISSING_LINK_TAU_HIGH) -> Confidence:
    """Confidence of an embedding-proposed link from its provenance.

    Embeddings propose, the graph disposes: a proposal the graph also corroborates
    (d_prev == 2 → source and target share a neighbour) at a high cosine is
    EXTRACTED; anything weaker is INFERRED. Missing links are pre-filtered ≥ τ in
    compute_report, so they never fall to AMBIGUOUS here.
    """
    corroborated = ml.d_prev == 2
    if ml.cosine >= tau_high and corroborated:
        return "EXTRACTED"
    return "INFERRED"


def classify_autolink(cand: "AutolinkCandidate") -> Confidence:
    """Confidence of an embedder-free co-occurrence autolink, from its evidence.

    Never EXTRACTED: an autolink is by construction the co-occurrence − wikilink
    delta (>2 hops, unlinked), so the graph does not corroborate it as a link.
    A directly shared concept is textual evidence the pair belongs together
    (INFERRED → propose); a pair related only through transitive/associative
    expansion has no shared concept and needs a human (AMBIGUOUS → escalate).
    """
    return "INFERRED" if cand.shared else "AMBIGUOUS"

# Capability names that are IRREVERSIBLE — never allowed in plan.auto regardless
# of confidence. Expand this set when new write tools are added.
_IRREVERSIBLE = frozenset({
    "silica_move",
    "silica_delete",
    "silica_bulk_write",
    "silica_run_injector",
    "silica_bulk_move",
    "silica_run_organizer",
})

# Threshold for orphan "linkable" heuristic: an orphan goes to `auto` only
# when its title appears as a substring of an existing note name (graph-safe
# title-match heuristic). If no title candidates are known at plan-time we use
# `propose` instead. The actual candidate lookup happens in silica_autolink.
_CLUSTER_SIZE_THRESHOLD = 40   # clusters > this → propose audit
_DANGLING_REFS_THRESHOLD = 2   # dangling targets seen >= this many times → escalate


@dataclass
class TaskCandidate:
    capability_name: str   # name of an existing tool in TOOLS
    payload: dict          # args dict for that tool
    reason: str            # human-readable explanation
    tier: Tier             # auto | propose | escalate
    priority: int = 0      # 0 = highest priority
    confidence: Confidence | None = None  # provenance that drove the tier (None = N/A)


@dataclass
class AnalystPlan:
    checkpoints: list[PlanStep]
    auto:     list[TaskCandidate] = field(default_factory=list)
    propose:  list[TaskCandidate] = field(default_factory=list)
    escalate: list[TaskCandidate] = field(default_factory=list)


def build_task_plan(report: VaultReport) -> AnalystPlan:
    """Translate a VaultReport into a three-tier AnalystPlan.

    Rules (applied in priority order, deterministic):

    1. Orphan with linkable title candidates → auto  silica_autolink
       Orphan without candidates            → propose silica_autolink
    2. Missing link (embedding, cosine ≥ τ) → propose silica_autolink on source
    3. Cluster size > threshold             → propose silica_graph_explain (audit)
    4. Dangling wikilink refs ≥ threshold   → escalate (create vs rename decision)
    """
    auto: list[TaskCandidate] = []
    propose: list[TaskCandidate] = []
    escalate: list[TaskCandidate] = []

    # Build a set of existing note paths for title-index heuristic
    # (we use god_node IDs + cluster members as a proxy for "existing titles")
    known_ids: set[str] = set()
    for n in report.god_nodes:
        known_ids.add(n.id.lower())
    for c in report.clusters:
        for m in c.members:
            known_ids.add(m.lower())

    def _has_title_candidate(orphan_path: str) -> bool:
        """Heuristic: orphan stem matches another known note's stem (substring)."""
        stem = orphan_path.rsplit("/", 1)[-1].removesuffix(".md").lower()
        if not stem:
            return False
        orphan_lower = orphan_path.lower()
        for kid in known_ids:
            if kid == orphan_lower:
                continue  # skip the orphan itself
            kid_stem = kid.rsplit("/", 1)[-1].removesuffix(".md").lower()
            if stem in kid_stem or kid_stem in stem:
                return True
        return False

    # Pre-compute cluster assignments for topology-aware chunking
    node_to_cluster: dict[str, int] = {}
    for c in report.clusters:
        for m in c.members:
            node_to_cluster[m] = c.cluster_id

    def _chunk_groups(group_map: dict[int, list[str]], max_bytes: int = 4096) -> list[list[str]]:
        chunks = []
        current_chunk = []
        current_size = 0
        for _, nodes in group_map.items():
            nodes_size = len(json.dumps(nodes))
            if current_size + nodes_size > max_bytes and current_chunk:
                chunks.append(current_chunk)
                current_chunk = list(nodes)
                current_size = nodes_size
            else:
                current_chunk.extend(nodes)
                current_size += nodes_size
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    # 1. Orphans
    auto_orphans_by_cluster: dict[int, list[str]] = {}
    propose_orphans_by_cluster: dict[int, list[str]] = {}

    for orphan in report.orphans:
        cid = node_to_cluster.get(orphan, -1)
        if _has_title_candidate(orphan):
            auto_orphans_by_cluster.setdefault(cid, []).append(orphan)
        else:
            propose_orphans_by_cluster.setdefault(cid, []).append(orphan)

    for chunk in _chunk_groups(auto_orphans_by_cluster):
        auto.append(TaskCandidate(
            capability_name="silica_autolink",
            payload={"note_paths": chunk, "use_candidates": True},
            reason=f"{len(chunk)} orphans have linkable title candidates → auto-link",
            tier="auto",
            priority=0,
        ))

    for chunk in _chunk_groups(propose_orphans_by_cluster):
        propose.append(TaskCandidate(
            capability_name="silica_autolink",
            payload={"note_paths": chunk, "use_candidates": True},
            reason=f"{len(chunk)} orphans — no clear title match, confirm before autolinking",
            tier="propose",
            priority=1,
        ))

    # 2. Missing links (embedding proposals, already filtered ≥ τ in compute_report)
    #    Tier follows the edge's confidence, not a flat propose: a proposal the
    #    graph corroborates (EXTRACTED) is auto-linkable (reversible, unambiguous);
    #    an embedding-only proposal (INFERRED) needs confirmation.
    conf_by_source: dict[str, Confidence] = {}
    for ml in report.missing_links:
        c = classify_missing_link(ml)
        prev = conf_by_source.get(ml.source)
        # EXTRACTED wins: a source with any corroborated link is auto-linkable.
        if prev is None or (c == "EXTRACTED" and prev != "EXTRACTED"):
            conf_by_source[ml.source] = c

    missing_by_tier: dict[Confidence, dict[int, list[str]]] = {"EXTRACTED": {}, "INFERRED": {}}
    for source, c in conf_by_source.items():
        cid = node_to_cluster.get(source, -1)
        missing_by_tier[c].setdefault(cid, []).append(source)

    for chunk in _chunk_groups(missing_by_tier["EXTRACTED"]):
        auto.append(TaskCandidate(
            capability_name="silica_autolink",
            payload={"note_paths": chunk, "use_candidates": True},
            reason=f"Embedding + graph corroborate links for {len(chunk)} source note(s) → auto-link",
            tier="auto",
            confidence="EXTRACTED",
            priority=2,
        ))

    for chunk in _chunk_groups(missing_by_tier["INFERRED"]):
        propose.append(TaskCandidate(
            capability_name="silica_autolink",
            payload={"note_paths": chunk, "use_candidates": True},
            reason=f"Embedding proposes links for {len(chunk)} source note(s) — confirm before writing",
            tier="propose",
            confidence="INFERRED",
            priority=2,
        ))

    # 2.6 Autolink candidates — the embedder-free co-occurrence twin of step 2.
    #     Works even when the embedder is down (missing_links would be empty).
    #     Tier follows evidence via classify_autolink: a shared concept is INFERRED
    #     → propose; an associative-only pair is AMBIGUOUS → escalate (human review).
    autolink_inferred_by_cluster: dict[int, list[str]] = {}
    seen_inferred: set[str] = set()
    for cand in getattr(report, "autolink_candidates", []):
        if classify_autolink(cand) == "INFERRED":
            if cand.source in seen_inferred:
                continue
            seen_inferred.add(cand.source)
            cid = node_to_cluster.get(cand.source, -1)
            autolink_inferred_by_cluster.setdefault(cid, []).append(cand.source)
        else:
            escalate.append(TaskCandidate(
                capability_name="",  # review-only — a human decides if the association is real
                payload={"source": cand.source, "target": cand.target},
                reason=(
                    f"'{cand.source}' and '{cand.target}' co-occur only associatively "
                    f"(no shared concept) — confirm relevance before linking"
                ),
                tier="escalate",
                confidence="AMBIGUOUS",
                priority=4,
            ))

    for chunk in _chunk_groups(autolink_inferred_by_cluster):
        propose.append(TaskCandidate(
            capability_name="silica_autolink",
            payload={"note_paths": chunk, "use_candidates": True},
            reason=f"Co-occurrence proposes links for {len(chunk)} source note(s) — confirm before writing",
            tier="propose",
            confidence="INFERRED",
            priority=2,
        ))

    # 2.5 Duplicate pairs (dedup suggestions)
    if hasattr(report, "duplicate_pairs") and report.duplicate_pairs:
        dup_graph = nx.Graph()
        for dp in report.duplicate_pairs:
            dup_graph.add_edge(dp.source, dp.target, score=dp.score)

        dedup_components = list(nx.connected_components(dup_graph))
        component_pairs = []
        for comp in dedup_components:
            comp_pairs = []
            for dp in report.duplicate_pairs:
                if dp.source in comp and dp.target in comp:
                    comp_pairs.append({"source": dp.source, "target": dp.target, "score": dp.score})
            if comp_pairs:
                component_pairs.append(comp_pairs)

        def _chunk_components(components: list[list[dict]], max_bytes: int = 4096) -> list[list[dict]]:
            chunks = []
            current_chunk = []
            current_size = 0
            for comp in components:
                comp_size = len(json.dumps(comp))
                if current_size + comp_size > max_bytes and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = list(comp)
                    current_size = comp_size
                else:
                    current_chunk.extend(comp)
                    current_size += comp_size
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        for chunk in _chunk_components(component_pairs):
            propose.append(TaskCandidate(
                capability_name="silica_dedup_pairs",
                payload={"pairs": chunk},
                reason=f"Embedding proposes {len(chunk)} duplicate pairs for merge — confirm before executing",
                tier="propose",
                priority=2,
            ))

    # 3. Refiner & Enricher (from OFM triage) → propose
    if getattr(report, "lean_notes", []):
        lean_chunks = _chunk_groups({1: report.lean_notes})
        for chunk in lean_chunks:
            propose.append(TaskCandidate(
                capability_name="silica_enrich_batch",
                payload={"note_paths": chunk},
                reason=f"Identified {len(chunk)} lean or empty note(s) → propose semantic enrichment",
                tier="propose",
                priority=3,
            ))
            
    if getattr(report, "reformat_notes", []):
        ref_chunks = _chunk_groups({1: report.reformat_notes})
        for chunk in ref_chunks:
            propose.append(TaskCandidate(
                capability_name="silica_refine_batch",
                payload={"note_paths": chunk},
                reason=f"Identified {len(chunk)} note(s) with missing or invalid tags → propose stylistic refinement",
                tier="propose",
                priority=3,
            ))

    # 4. Oversized clusters → propose a read-only audit
    for c in report.clusters:
        if c.size > _CLUSTER_SIZE_THRESHOLD and c.hub:
            propose.append(TaskCandidate(
                capability_name="silica_graph_explain",
                payload={"note": c.hub, "depth": 1},
                reason=(
                    f"Cluster {c.cluster_id} is large (size={c.size} > {_CLUSTER_SIZE_THRESHOLD}) "
                    f"— audit hub '{c.hub}' to decide if refactoring is needed"
                ),
                tier="propose",
                priority=4,
            ))

    # 5. Recurring dangling wikilinks → escalate (create vs rename is irreversible)
    for d in report.dangling:
        if d["refs"] >= _DANGLING_REFS_THRESHOLD:
            escalate.append(TaskCandidate(
                capability_name="",  # no automatic capability — human decides
                payload={"target": d["target"], "refs": d["refs"]},
                reason=(
                    f"Unresolved wikilink '{d['target']}' appears {d['refs']} time(s) "
                    f"— decide: create note, rename existing, or ignore"
                ),
                tier="escalate",
                priority=5,
            ))

    # §3.2-bis safety check: strip any irreversible capability that leaked into auto
    auto = [c for c in auto if c.capability_name not in _IRREVERSIBLE]

    # Sort by priority within each tier
    auto.sort(key=lambda c: (c.priority, c.reason))
    propose.sort(key=lambda c: (c.priority, c.reason))
    escalate.sort(key=lambda c: (c.priority, c.reason))

    checkpoints = [
        PlanStep(id="audit",    kind="mechanical", objective="silica_vault_report"),
        PlanStep(id="remediate", kind="gate",      objective="silica_autolink"),
        PlanStep(id="report",   kind="mechanical", objective="silica_ledger_digest"),
    ]

    return AnalystPlan(
        checkpoints=checkpoints,
        auto=auto,
        propose=propose,
        escalate=escalate,
    )
