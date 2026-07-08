# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Curator composer — project graph_report findings into a typed CurationPlan.

The vault already has every remediation *mechanism* (/dedup, /refine, orphan
repair, autolink) but they are all *pull*: a vault is only curated when the
user remembers to. This module is the *initiative* half — a pure projection
from the L1 VaultReport onto a plan of typed items the dispatch layer
(silica.tools.curate) then enqueues on the existing capability seam. No new
power: every item maps onto a WorkItem kind (or the mechanical autolink path)
that already exists.

Finding → item (spec-hermes-coherence §5):
    strong autolink candidate  → autolink  (mechanical, no LLM, direct commit)
    orphan                     → orphan    WorkItem
    high-similarity pair       → dedup     WorkItem (ternary verdict incl.
                                            contradicts → contested sweep)
    oversized / lean note      → refine    WorkItem

Pure & kernel-legal: reads only graph_report dataclasses, no I/O, no
router/capabilities import (import-linter boundary). It is the curator's twin
of kernel.analyst_plan — that one seeds the analyst ledger, this one drives the
background policy.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from silica.kernel.graph_report import VaultReport

# The four item kinds the curator can emit. "autolink" is the mechanical,
# LLM-free direct commit; the rest are WorkItem kinds on the capability seam.
Kind = str  # "autolink" | "orphan" | "dedup" | "refine"

# Closed set of emittable kinds — single source of truth for CurationPlan.filtered
# validation, so a typo like "dedups" is rejected loudly instead of silently
# filtering the plan to empty.
VALID_KINDS = frozenset({"autolink", "orphan", "dedup", "refine"})


def _norm_note_path(path: str) -> str:
    """Forward-slash, `.md`-suffixed note path for suffix matching.

    Local to the kernel (curator.py can't import silica.agent.bounds across the
    import-linter boundary); named apart from bounds._norm_path to avoid confusion.
    """
    p = path.strip().replace("\\", "/")
    return p if p.endswith(".md") else p + ".md"

# Silica's own generated artifacts, living at the VAULT ROOT: the driver
# indexes them like any other note (in-degree 0 -> orphan finding, no
# frontmatter -> reformat finding), but the curator must never plan work
# against them — --apply would LLM-rewrite the journal or the report on
# every vault with >=1 ingest. Matched by root-relative stem only, so a
# genuine note in a subfolder sharing the name (e.g. "Concepts/log.md")
# stays curatable.
_VAULT_ROOT_ARTIFACT_STEMS = {"log", "GRAPH_REPORT"}


def _is_vault_artifact(note_id: str) -> bool:
    """True if `note_id` is a Silica-generated file at the vault root.

    Id form varies by caller (graph node ids carry `.md`; other callers may
    not), so this matches on the `.md`-stripped stem and requires no path
    separator — i.e. vault-root only.
    """
    stem = note_id.removesuffix(".md")
    return "/" not in stem and stem in _VAULT_ROOT_ARTIFACT_STEMS


@dataclass
class CurationItem:
    kind: Kind
    target: str            # primary note id — graph node id form, carries `.md`
    partner: str = ""      # dedup/autolink: the other note in the pair
    score: float = 0.0     # similarity / co-occurrence weight, when available
    reason: str = ""       # human-readable provenance


@dataclass
class CurationPlan:
    items: list[CurationItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def is_empty(self) -> bool:
        return not self.items

    def by_kind(self, kind: Kind) -> list[CurationItem]:
        return [i for i in self.items if i.kind == kind]

    def counts(self) -> dict[str, int]:
        """Item count per kind, omitting kinds with zero items."""
        return dict(Counter(i.kind for i in self.items))

    def filtered(
        self,
        kinds: list[str] | None = None,
        targets: list[str] | None = None,
    ) -> "CurationPlan":
        """Return a new plan keeping items that satisfy both predicates.

        - `kinds`: keep items whose kind is in `kinds` (case-insensitive).
          Validated against VALID_KINDS — an unknown kind raises ValueError
          rather than silently filtering to empty. Empty/None ⇒ all kinds.
        - `targets`: keep items where a requested path suffix-matches
          `item.target` OR `item.partner` on segment boundaries (a request `r`
          matches path `p` when `p == r` or `p.endswith("/" + r)`, both
          normalized to forward-slash + `.md`). Empty/None ⇒ all targets.
        - Both predicates AND; both empty ⇒ identity. Pure, no mutation.
        """
        kset: set[str] | None = None
        if kinds:
            kset = {k.strip().lower() for k in kinds}
            unknown = kset - VALID_KINDS
            if unknown:
                raise ValueError(
                    "unknown curation kind(s): "
                    + ", ".join(sorted(unknown))
                    + "; valid kinds: "
                    + ", ".join(sorted(VALID_KINDS))
                )
        reqs = [_norm_note_path(t) for t in (targets or []) if t.strip()]

        def keep(item: CurationItem) -> bool:
            if kset is not None and item.kind not in kset:
                return False
            if reqs:
                # ponytail: bare-stem `x.md` suffix-matches every folder's x.md
                # (intended convenience); the escape hatch is a folder-qualified
                # path (Concepts/x.md), which the segment-boundary rule narrows.
                paths = [_norm_note_path(item.target)]
                if item.partner:
                    paths.append(_norm_note_path(item.partner))
                if not any(
                    p == r or p.endswith("/" + r) for p in paths for r in reqs
                ):
                    return False
            return True

        return CurationPlan(items=[i for i in self.items if keep(i)])


def compose_curation_plan(report: VaultReport) -> CurationPlan:
    """Project a VaultReport into a deterministic plan of typed curation items.

    Deterministic and side-effect-free: the same report always yields the same
    plan. Duplicate dedup pairs (a pair present in both the confirmed and the
    borderline band, in either orientation) collapse to a single item.
    """
    items: list[CurationItem] = []

    # 1. Strong autolink candidates → mechanical direct commit.
    #    "Strong" == corroborated by a directly shared concept (INFERRED, per
    #    analyst_plan.classify_autolink). Associative-only pairs (no shared
    #    concept) are AMBIGUOUS — a human decides, so the curator leaves them.
    for cand in report.autolink_candidates:
        if _is_vault_artifact(cand.source) or _is_vault_artifact(cand.target):
            continue
        if cand.shared:
            items.append(CurationItem(
                kind="autolink",
                target=cand.source,
                partner=cand.target,
                score=cand.weight,
                reason="co-occurrence: " + ", ".join(cand.shared),
            ))

    # 2. Orphans (in-degree 0) → orphan-connector WorkItem.
    for orphan in report.orphans:
        if _is_vault_artifact(orphan):
            continue
        items.append(CurationItem(
            kind="orphan",
            target=orphan,
            reason="orphan (no inbound links)",
        ))

    # 3. High-similarity pairs → dedup WorkItem. Confirmed (>= tau_high) first,
    #    then the borderline band; the dedup worker itself returns the ternary
    #    verdict (duplicate / distinct / contradicts), so feeding both bands
    #    also yields the contested-notes sweep for free.
    seen_pairs: set[tuple[str, str]] = set()
    for dp in list(report.confirmed_duplicate_pairs) + list(report.duplicate_pairs):
        if _is_vault_artifact(dp.source) or _is_vault_artifact(dp.target):
            continue
        key = tuple(sorted((dp.source, dp.target)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        items.append(CurationItem(
            kind="dedup",
            target=dp.source,
            partner=dp.target,
            score=dp.score,
            reason=f"similarity {dp.score:.3f}",
        ))

    # 4. Oversized / lean notes → refine WorkItem. reformat_notes is the
    #    report's "Stylistic Refinement" bucket; lean_notes fold in per the spec
    #    row "oversized / lean → refine".
    seen_refine: set[str] = set()
    for note in list(report.reformat_notes) + list(report.lean_notes):
        if _is_vault_artifact(note) or note in seen_refine:
            continue
        seen_refine.add(note)
        items.append(CurationItem(
            kind="refine",
            target=note,
            reason="needs stylistic refinement",
        ))

    return CurationPlan(items=items)
