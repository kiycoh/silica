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

from silica.kernel.graph_report import VaultReport
from silica.planner.progress import CheckpointSpec

Tier = Literal["auto", "propose", "escalate"]

# Capability names that are IRREVERSIBLE — never allowed in plan.auto regardless
# of confidence. Expand this set when new write tools are added.
_IRREVERSIBLE = frozenset({
    "silica_move",
    "silica_delete",
    "silica_bulk_write",
    "silica_run_injector",
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


@dataclass
class AnalystPlan:
    checkpoints: list[CheckpointSpec]
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
    seen_autolink_propose: set[str] = set()
    propose_missing_by_cluster: dict[int, list[str]] = {}
    for ml in report.missing_links:
        if ml.source not in seen_autolink_propose:
            seen_autolink_propose.add(ml.source)
            cid = node_to_cluster.get(ml.source, -1)
            propose_missing_by_cluster.setdefault(cid, []).append(ml.source)

    for chunk in _chunk_groups(propose_missing_by_cluster):
        propose.append(TaskCandidate(
            capability_name="silica_autolink",
            payload={"note_paths": chunk, "use_candidates": True},
            reason=f"Embedding proposes links for {len(chunk)} source notes — confirm before writing",
            tier="propose",
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
        CheckpointSpec(id="audit",    kind="mechanical", objective="silica_vault_report"),
        CheckpointSpec(id="remediate", kind="gate",      objective="silica_autolink"),
        CheckpointSpec(id="report",   kind="mechanical", objective="silica_ledger_digest"),
    ]

    return AnalystPlan(
        checkpoints=checkpoints,
        auto=auto,
        propose=propose,
        escalate=escalate,
    )
