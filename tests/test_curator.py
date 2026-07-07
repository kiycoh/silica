"""Tests for the Curator — vault maintenance as a background policy.

Two layers under test:
  * silica.kernel.curator.compose_curation_plan — the PURE composer that
    projects graph_report findings into a typed CurationPlan (no I/O).
  * silica.tools.curate — the dispatch layer: dry-run (print, enqueue/write
    nothing) vs --apply (enqueue WorkItems on the existing capability seam,
    mechanical autolink direct-commit, one idempotent journal line).

The composer is deterministic over a synthetic VaultReport, so the acceptance
case (1 orphan + 1 near-dup pair + 1 oversized/lean note → 3 items) is a pure
unit test with no live driver.
"""
from __future__ import annotations

import pytest

from silica.kernel.graph_report import (
    AutolinkCandidate,
    DuplicatePair,
    VaultReport,
)
from silica.kernel.curator import (
    CurationItem,
    CurationPlan,
    compose_curation_plan,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _report(**overrides) -> VaultReport:
    base = dict(
        generated_at="2026-07-02T00:00:00Z",
        scope="",
        totals={},
        god_nodes=[],
        bridges=[],
        orphans=[],
        dangling=[],
        clusters=[],
    )
    base.update(overrides)
    return VaultReport(**base)


# ---------------------------------------------------------------------------
# composer — the acceptance case
# ---------------------------------------------------------------------------

def test_compose_one_orphan_one_dup_one_oversized_gives_three_items():
    """Spec acceptance: 1 orphan + 1 near-dup pair + 1 oversized note → 3 items."""
    report = _report(
        orphans=["Concepts/Lonely"],
        confirmed_duplicate_pairs=[
            DuplicatePair(source="Concepts/A", target="Concepts/B", score=0.91),
        ],
        reformat_notes=["Concepts/Bloated"],
    )

    plan = compose_curation_plan(report)

    assert len(plan) == 3
    counts = plan.counts()
    assert counts.get("orphan") == 1
    assert counts.get("dedup") == 1
    assert counts.get("refine") == 1


def test_compose_maps_each_finding_to_the_right_kind():
    report = _report(
        orphans=["Orphan One"],
        confirmed_duplicate_pairs=[DuplicatePair(source="A", target="B", score=0.9)],
        reformat_notes=["Reformat Me"],
        lean_notes=["Lean One"],
        autolink_candidates=[
            AutolinkCandidate(source="X", target="Y", weight=3.2, shared=["neural nets"]),
        ],
    )

    plan = compose_curation_plan(report)

    kinds = {i.kind for i in plan.items}
    assert kinds == {"orphan", "dedup", "refine", "autolink"}
    # orphan carries the note; dedup carries both sides
    dedup = plan.by_kind("dedup")[0]
    assert dedup.target == "A" and dedup.partner == "B"
    assert dedup.score == 0.9
    # lean + reformat both become refine work
    assert {i.target for i in plan.by_kind("refine")} == {"Reformat Me", "Lean One"}


def test_compose_only_strong_autolinks_become_mechanical_items():
    """A candidate with shared-concept evidence is 'strong' → autolink item;
    an associative-only candidate (no shared concept) is skipped (needs a human)."""
    report = _report(
        autolink_candidates=[
            AutolinkCandidate(source="Strong", target="Partner", weight=4.0, shared=["topic"]),
            AutolinkCandidate(source="Weak", target="Other", weight=4.0, shared=[]),
        ],
    )

    plan = compose_curation_plan(report)
    al = plan.by_kind("autolink")
    assert len(al) == 1
    assert al[0].target == "Strong" and al[0].partner == "Partner"


def test_compose_dedup_pairs_are_deduplicated_across_bands():
    """The same pair appearing in both confirmed and borderline bands yields one item."""
    report = _report(
        confirmed_duplicate_pairs=[DuplicatePair(source="A", target="B", score=0.9)],
        duplicate_pairs=[DuplicatePair(source="B", target="A", score=0.7)],
    )
    plan = compose_curation_plan(report)
    assert len(plan.by_kind("dedup")) == 1


def test_compose_empty_report_is_empty_plan():
    plan = compose_curation_plan(_report())
    assert len(plan) == 0
    assert plan.is_empty()
    assert plan.counts() == {}


# ---------------------------------------------------------------------------
# Silica-artifact exclusion — the curator must never target its own
# generated files (vault-root log.md / GRAPH_REPORT.md). The driver indexes
# them like any other note (in-degree 0 -> orphan, no frontmatter -> reformat)
# but --apply LLM-rewriting the journal or the report is never correct.
# ---------------------------------------------------------------------------

def test_compose_excludes_vault_root_silica_artifacts_from_orphan_and_refine():
    report = _report(
        orphans=["log.md", "GRAPH_REPORT.md", "Concepts/Lonely"],
        reformat_notes=["log.md", "GRAPH_REPORT.md", "Concepts/Bloated"],
    )

    plan = compose_curation_plan(report)

    assert {i.target for i in plan.by_kind("orphan")} == {"Concepts/Lonely"}
    assert {i.target for i in plan.by_kind("refine")} == {"Concepts/Bloated"}


def test_compose_excludes_vault_root_artifacts_regardless_of_md_suffix():
    """Id form may or may not carry `.md` depending on the caller — exclude
    both forms of the vault-root artifact name."""
    report = _report(orphans=["log", "GRAPH_REPORT"])

    plan = compose_curation_plan(report)

    assert plan.by_kind("orphan") == []


def test_compose_does_not_exclude_subfolder_notes_sharing_the_artifact_name():
    """Only VAULT-ROOT log.md/GRAPH_REPORT.md are Silica artifacts — a note
    in a subfolder that happens to share the name is a real note."""
    report = _report(orphans=["Concepts/log.md", "Archive/GRAPH_REPORT.md"])

    plan = compose_curation_plan(report)

    assert {i.target for i in plan.by_kind("orphan")} == {
        "Concepts/log.md", "Archive/GRAPH_REPORT.md",
    }


# ---------------------------------------------------------------------------
# journal line shape
# ---------------------------------------------------------------------------

def test_format_curate_event_shape():
    from silica.kernel.run_log import format_curate_event

    event = format_curate_event({"dedup": 2, "refine": 1, "orphan": 3, "autolink": 4})
    assert event == "curate → 10 item (2 dedup, 1 refine, 3 orphan, 4 autolink)"


def test_format_curate_event_omits_zero_types():
    from silica.kernel.run_log import format_curate_event

    event = format_curate_event({"orphan": 1})
    assert event == "curate → 1 item (1 orphan)"


# ---------------------------------------------------------------------------
# CurationPlan.filtered — selective apply (pure, deterministic, no I/O).
# A user, after seeing the plan, drives a subset in natural language ("apply
# only the dedups", "refine only x.md"); the agent maps that onto kind/target
# filter args. The filter shapes both the dry-run preview and the apply path.
# ---------------------------------------------------------------------------

def _mixed_plan() -> CurationPlan:
    return CurationPlan(items=[
        CurationItem(kind="orphan", target="Concepts/Lonely"),
        CurationItem(kind="dedup", target="Concepts/A", partner="Concepts/B", score=0.9),
        CurationItem(kind="refine", target="Concepts/Bloated"),
        CurationItem(kind="autolink", target="Concepts/X", partner="Concepts/Y", score=3.0),
    ])


def test_filtered_kinds_only_keeps_matching_kinds():
    plan = _mixed_plan().filtered(kinds=["dedup"])
    assert [i.kind for i in plan.items] == ["dedup"]


def test_filtered_kinds_is_case_insensitive():
    plan = _mixed_plan().filtered(kinds=["DEDUP"])
    assert [i.kind for i in plan.items] == ["dedup"]


def test_filtered_targets_only_keeps_by_path_regardless_of_kind():
    plan = _mixed_plan().filtered(targets=["Lonely.md", "Bloated.md"])
    assert {i.kind for i in plan.items} == {"orphan", "refine"}


def test_filtered_kinds_and_targets_both_predicates_must_hold():
    # targets alone would keep dedup (A) AND orphan (Lonely); kinds narrows to dedup.
    plan = _mixed_plan().filtered(kinds=["dedup"], targets=["A.md", "Lonely.md"])
    assert [i.kind for i in plan.items] == ["dedup"]


def test_filtered_empty_args_is_identity():
    src = _mixed_plan()
    assert [i.kind for i in src.filtered().items] == [i.kind for i in src.items]
    assert [i.kind for i in src.filtered(kinds=[], targets=[]).items] == [
        i.kind for i in src.items
    ]


def test_filtered_matches_on_partner_side_item_unchanged():
    # dedup target=A partner=B; requesting the partner's stem keeps the pair intact.
    plan = _mixed_plan().filtered(targets=["B.md"])
    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.kind == "dedup"
    assert item.target == "Concepts/A" and item.partner == "Concepts/B"


def test_filtered_bare_stem_matches_any_folder_qualified_narrows():
    plan = CurationPlan(items=[
        CurationItem(kind="refine", target="Concepts/x.md"),
        CurationItem(kind="refine", target="Other/x.md"),
    ])
    # a bare stem matches every folder's x.md (the intended convenience)
    assert len(plan.filtered(targets=["x.md"]).items) == 2
    # a folder-qualified path narrows to that one note (the escape hatch)
    narrowed = plan.filtered(targets=["Concepts/x.md"]).items
    assert [i.target for i in narrowed] == ["Concepts/x.md"]


def test_filtered_no_match_returns_empty_plan():
    plan = _mixed_plan().filtered(targets=["does-not-exist.md"])
    assert plan.is_empty()


def test_filtered_unknown_kind_raises_listing_valid_kinds():
    with pytest.raises(ValueError) as exc:
        _mixed_plan().filtered(kinds=["dedups"])
    msg = str(exc.value)
    assert "dedups" in msg
    for valid in ("autolink", "orphan", "dedup", "refine"):
        assert valid in msg
