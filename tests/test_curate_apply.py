"""Tests for the Curator dispatch layer (silica.tools.curate).

The composer is pure (see test_curator.py). Here we pin behaviour:
  * apply_curation_plan enqueues the right WorkItem kinds on the seam,
    fires the mechanical autolink direct-commit, and appends exactly one
    idempotent journal line per run.
  * silica_curate defaults to dry-run: it composes + returns the plan but
    enqueues NOTHING and writes NOTHING; --apply routes through the dispatch.
  * an empty plan is a no-op: no dispatch, no journal line.

The seam and the I/O helpers are patched so no LLM / driver / embed index is
touched — the same faking style as tests/test_subagent.py.
"""
from __future__ import annotations

import json

import silica.tools.curate as curate
from silica.kernel.curator import CurationItem, CurationPlan
from silica.kernel.graph_report import VaultReport
from silica.kernel.run_log import DEFAULT_LOG_FILENAME


def _plan(*items: CurationItem) -> CurationPlan:
    return CurationPlan(items=list(items))


def test_dedup_workitems_collapses_confirmed_family_to_largest(monkeypatch):
    """A confirmed duplicate FAMILY (A-B-C chain) collapses to its single largest
    note, not one survivor per local top-1 hub. Borderline pairs stay per-pair."""
    bodies = {"A": "a" * 10, "B": "b" * 100, "C": "c" * 50, "D": "d" * 30, "E": "e" * 20}
    monkeypatch.setattr(curate, "_read_body", lambda p: bodies.get(p, ""))
    monkeypatch.setattr("silica.config.CONFIG.sim_threshold_high", 0.85, raising=False)

    plan = _plan(
        CurationItem(kind="dedup", target="A", partner="B", score=0.90),  # confirmed
        CurationItem(kind="dedup", target="B", partner="C", score=0.90),  # confirmed, chains
        CurationItem(kind="dedup", target="D", partner="E", score=0.70),  # borderline
    )
    items = curate._dedup_workitems(plan)

    # A and C both merge INTO B (the largest of the confirmed component) — one
    # survivor, and the family ships as ONE batch item (one judge call).
    family = next(w for w in items if w.target_path == "B")
    assert {c["concept"] for c in family.context["concepts"]} == {"A", "C"}
    # Borderline D-E untouched by the closure: larger (D) is the target, per-pair.
    single = next(w for w in items if w.target_path == "D")
    assert single.context["concept"] == "E"
    assert len(items) == 2


def _report(**overrides) -> VaultReport:
    base = dict(
        generated_at="2026-07-02T00:00:00Z",
        scope="", totals={}, god_nodes=[], bridges=[],
        orphans=[], dangling=[], clusters=[],
    )
    base.update(overrides)
    return VaultReport(**base)


class _Capture:
    """Stand-in for run_subagent_batch that records the items it received."""

    def __init__(self):
        self.items = None

    def __call__(self, items, config=None, **kw):
        self.items = list(items)
        return {"items": len(items), "summary": {"committed": len(items)}, "results": []}


# ---------------------------------------------------------------------------
# apply_curation_plan
# ---------------------------------------------------------------------------

def test_apply_enqueues_correct_workitem_kinds(monkeypatch, tmp_path):
    cap = _Capture()
    monkeypatch.setattr(curate, "run_subagent_batch", cap)
    monkeypatch.setattr(curate, "_orphan_candidates", lambda p, k=5: [{"name": "N", "path": "N.md"}])
    monkeypatch.setattr(curate, "_read_body", lambda p: "body of " + p)
    autolinked = {}

    def _fake_autolink(srcs):
        autolinked["srcs"] = srcs
        return {"notes_processed": 1, "total_links_added": 1}

    monkeypatch.setattr(curate, "_run_autolink", _fake_autolink)

    plan = _plan(
        CurationItem(kind="orphan", target="Lonely"),
        CurationItem(kind="dedup", target="A", partner="B", score=0.9),
        CurationItem(kind="refine", target="Bloated"),
        CurationItem(kind="autolink", target="X", partner="Y", score=3.0),
    )

    res = curate.apply_curation_plan(plan, run_id="feedface1234", vault_path=str(tmp_path))

    kinds = sorted(it.kind for it in cap.items)
    assert kinds == ["dedup", "orphan", "refine"]  # autolink is NOT a WorkItem
    assert autolinked["srcs"] == ["X"]             # mechanical autolink fired on source
    assert res["status"] == "applied"

    # the dedup item is hydrated with an excerpt + concept from the partner
    dedup = next(it for it in cap.items if it.kind == "dedup")
    assert dedup.context["excerpt"]
    assert dedup.context["score"] == 0.9
    # the orphan item carries the offered candidates
    orphan = next(it for it in cap.items if it.kind == "orphan")
    assert orphan.context["candidates"] == [{"name": "N", "path": "N.md"}]


def test_apply_appends_one_journal_line_idempotent_per_run(monkeypatch, tmp_path):
    monkeypatch.setattr(curate, "run_subagent_batch", _Capture())
    monkeypatch.setattr(curate, "_orphan_candidates", lambda p, k=5: [])
    monkeypatch.setattr(curate, "_read_body", lambda p: "body")
    monkeypatch.setattr(curate, "_run_autolink", lambda srcs: {})

    plan = _plan(
        CurationItem(kind="orphan", target="Lonely"),
        CurationItem(kind="dedup", target="A", partner="B", score=0.8),
    )

    curate.apply_curation_plan(plan, run_id="cafebabe0001", vault_path=str(tmp_path))
    # a re-run under the SAME run id must not duplicate the line
    curate.apply_curation_plan(plan, run_id="cafebabe0001", vault_path=str(tmp_path))

    log = (tmp_path / DEFAULT_LOG_FILENAME).read_text(encoding="utf-8")
    lines = [ln for ln in log.splitlines() if "curate" in ln]
    assert len(lines) == 1
    assert "run cafebabe" in lines[0]
    assert "2 item" in lines[0]


def test_apply_journal_reports_real_outcomes_not_planned_counts(monkeypatch, tmp_path):
    """A batch where every dedup came back 'no_merge' (distinct — the worker
    declined to merge) must not be journalled as though the planned item
    succeeded. The line must reflect run_subagent_batch's REAL per-item
    outcome, not plan.counts()."""
    def _fake_batch(items, config=None, **kw):
        return {
            "items": len(items),
            "summary": {"no_merge": len(items)},
            "results": [{"target": it.target_path, "status": "no_merge"} for it in items],
        }

    monkeypatch.setattr(curate, "run_subagent_batch", _fake_batch)
    monkeypatch.setattr(curate, "_read_body", lambda p: "body")
    monkeypatch.setattr(curate, "_run_autolink", lambda srcs: {})

    plan = _plan(CurationItem(kind="dedup", target="A", partner="B", score=0.7))

    res = curate.apply_curation_plan(plan, run_id="dead0001beef", vault_path=str(tmp_path))

    assert res["outcome_counts"] == {"no_merge": 1}

    log = (tmp_path / DEFAULT_LOG_FILENAME).read_text(encoding="utf-8")
    lines = [ln for ln in log.splitlines() if "curate" in ln]
    assert len(lines) == 1
    assert "no_merge" in lines[0]
    assert "1 dedup" not in lines[0]  # not the planned-kind phrasing


def test_apply_outcome_counts_include_real_autolink_result(monkeypatch, tmp_path):
    """The mechanical autolink item count in the plan is a candidate count,
    not what actually happened — outcome_counts must use the real
    links-added figure from silica_autolink's return value. The mock uses
    silica_autolink's REAL return shape {"notes_processed", "total_links_added"}
    (silica/tools/graph.py) — "added" is silica_backlink's key, not autolink's."""
    monkeypatch.setattr(curate, "run_subagent_batch", _Capture())
    monkeypatch.setattr(
        curate, "_run_autolink",
        lambda srcs: {"notes_processed": 1, "total_links_added": 3},
    )

    plan = _plan(CurationItem(kind="autolink", target="X", partner="Y", score=3.0))

    res = curate.apply_curation_plan(plan, run_id="beadfeed0002", vault_path=str(tmp_path))

    assert res["outcome_counts"] == {"autolink": 3}


def test_apply_empty_plan_no_dispatch_no_journal(monkeypatch, tmp_path):
    cap = _Capture()
    monkeypatch.setattr(curate, "run_subagent_batch", cap)

    def _boom(_srcs):
        raise AssertionError("autolink must not fire on an empty plan")

    monkeypatch.setattr(curate, "_run_autolink", _boom)

    res = curate.apply_curation_plan(_plan(), run_id="deadbeef0000", vault_path=str(tmp_path))

    assert res["status"] == "nothing_to_do"
    assert cap.items is None                              # nothing enqueued
    assert not (tmp_path / DEFAULT_LOG_FILENAME).exists()  # nothing journalled


# ---------------------------------------------------------------------------
# silica_curate tool — dry-run vs --apply
# ---------------------------------------------------------------------------

def test_dry_run_composes_but_does_not_enqueue_or_write(monkeypatch, tmp_path):
    report = _report(
        orphans=["Lonely"],
        reformat_notes=["Bloated"],
    )
    monkeypatch.setattr(curate, "compute_report", lambda **kw: report)

    def _boom_apply(*a, **k):
        raise AssertionError("dry-run must not call apply_curation_plan")

    monkeypatch.setattr(curate, "apply_curation_plan", _boom_apply)

    res = curate.silica_curate(apply=False)

    assert res["status"] == "dry_run"
    assert res["total"] == 2
    assert res["counts"] == {"orphan": 1, "refine": 1}
    assert not (tmp_path / DEFAULT_LOG_FILENAME).exists()


def test_apply_flag_routes_through_dispatch(monkeypatch):
    report = _report(orphans=["Lonely"])
    monkeypatch.setattr(curate, "compute_report", lambda **kw: report)

    seen = {}

    def _fake_apply(plan, **kw):
        seen["n"] = len(plan)
        return {"status": "applied", "run_id": "x"}

    monkeypatch.setattr(curate, "apply_curation_plan", _fake_apply)

    res = curate.silica_curate(apply=True)

    assert res["status"] == "applied"
    assert seen["n"] == 1


def test_filter_matching_nothing_reports_no_matches_not_nothing_to_do(monkeypatch):
    """A non-empty pre-filter plan emptied by the filter is `no_matches`, not
    `nothing_to_do` — the vault has work, the filter (not the vault) produced the
    emptiness. `available` exposes the pre-filter counts so the agent self-corrects."""
    report = _report(orphans=["Lonely"], reformat_notes=["Bloated"])
    monkeypatch.setattr(curate, "compute_report", lambda **kw: report)

    def _boom_apply(*a, **k):
        raise AssertionError("an emptied plan must not dispatch")

    monkeypatch.setattr(curate, "apply_curation_plan", _boom_apply)

    res = curate.silica_curate(apply=True, targets=["does-not-exist.md"])

    assert res["status"] == "no_matches"
    assert res["available"] == {"orphan": 1, "refine": 1}
    assert res["total"] == 0


def test_unknown_kind_surfaces_as_tool_error(monkeypatch):
    """A kind typo is a loud ValueError, wrapped by the @tool runner as an
    `error` — never a silent empty filter."""
    import silica.tools as tools

    monkeypatch.setattr(curate, "compute_report", lambda **kw: _report(orphans=["Lonely"]))

    result = json.loads(tools.TOOLS["silica_curate"].run(kinds=["dedups"]))
    assert "error" in result
    assert "dedups" in result["error"]


def test_empty_report_tool_reports_nothing_to_do(monkeypatch):
    monkeypatch.setattr(curate, "compute_report", lambda **kw: _report())

    def _boom_apply(*a, **k):
        raise AssertionError("empty report must not dispatch")

    monkeypatch.setattr(curate, "apply_curation_plan", _boom_apply)

    res = curate.silica_curate(apply=True)
    assert res["total"] == 0
    assert res["status"] == "nothing_to_do"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_cli_curate_dry_run_invokes_tool():
    from unittest.mock import MagicMock, patch
    from silica import cli

    fake_tool = MagicMock()
    fake_tool.run.return_value = json.dumps({
        "status": "dry_run", "total": 1, "counts": {"orphan": 1},
        "items": [{"kind": "orphan", "target": "Lonely", "partner": "", "reason": ""}],
    })
    with patch.dict("silica.tools.TOOLS", {"silica_curate": fake_tool}, clear=False):
        handled = cli._handle_direct_shortcut("/curate Concepts", [])
    assert handled is True
    fake_tool.run.assert_called_once_with(apply=False, folder="Concepts")


def test_cli_curate_apply_flag_passed_through():
    from unittest.mock import MagicMock, patch
    from silica import cli

    fake_tool = MagicMock()
    fake_tool.run.return_value = json.dumps({
        "status": "applied", "total": 2, "counts": {"orphan": 1, "refine": 1},
        "execution": {"outcome_counts": {"committed": 1, "no_link": 1}},
    })
    with patch.dict("silica.tools.TOOLS", {"silica_curate": fake_tool}, clear=False):
        handled = cli._handle_direct_shortcut("/curate --apply", [])
    assert handled is True
    fake_tool.run.assert_called_once_with(apply=True, folder="")


def test_cli_curate_apply_prints_real_outcomes_not_planned_counts(monkeypatch):
    """A dedup batch that came back all 'no_merge' must not be printed as
    'Applied 1 item(s): 1 dedup' — that's the plan, not what happened."""
    from unittest.mock import MagicMock, patch
    from silica import cli
    from silica.ui.console import CONSOLE

    fake_tool = MagicMock()
    fake_tool.run.return_value = json.dumps({
        "status": "applied", "total": 1, "counts": {"dedup": 1},
        "execution": {
            "outcome_counts": {"no_merge": 1},
            "batch": {"items": 1, "summary": {"no_merge": 1}, "results": []},
            "autolink": {},
        },
    })
    buf: list[str] = []
    monkeypatch.setattr(CONSOLE, "print", lambda *a, **kw: buf.append(" ".join(str(x) for x in a)))
    with patch.dict("silica.tools.TOOLS", {"silica_curate": fake_tool}, clear=False):
        handled = cli._handle_direct_shortcut("/curate --apply", [])
    assert handled is True
    printed = "\n".join(buf)
    assert "no_merge" in printed
    assert "1 dedup" not in printed
