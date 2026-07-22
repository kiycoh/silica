"""Tests for the contested-claims layer (spec-hermes-coherence §1).

Kernel: mark_contested / contested_callout / contested_refs are pure
(frontmatter in, frontmatter out). Integration: a patch Op carrying
`contested_by` marks the target note through the single write executor,
idempotently (re-running the same op adds nothing).
"""
from __future__ import annotations

import pytest

from silica.kernel.contested import (
    clear_contested,
    contested_callout,
    contested_refs,
    mark_contested,
)
from silica.kernel import frontmatter
from silica.kernel.ofm import ofm_lint


NOTE = """---
parent note: "[[Farmacologia]]"
related:
  - "[[Farmacologia]]"
tags:
  - farmacologia
last modified: 2026, 07, 02
AI: true
---

# Dosaggio Warfarin

Il dosaggio raccomandato è 5mg/die con monitoraggio INR.
"""

REF = "fonte: appunti-cardiologia-2026.md"
REF2 = "[[Warfarin (linee guida 2027)]]"


# --- mark_contested (pure) ---------------------------------------------------

def test_mark_contested_sets_flag_and_ref():
    out = mark_contested(NOTE, REF)
    data, _, body = frontmatter.split(out)
    assert data["contested"] is True
    assert data["contradictions"] == [REF]
    assert "Il dosaggio raccomandato" in body  # body untouched


def test_mark_contested_idempotent_same_ref():
    once = mark_contested(NOTE, REF)
    twice = mark_contested(once, REF)
    assert twice == once


def test_mark_contested_appends_second_ref():
    out = mark_contested(mark_contested(NOTE, REF), REF2)
    data, _, _ = frontmatter.split(out)
    assert data["contradictions"] == [REF, REF2]


def test_mark_contested_creates_frontmatter_when_missing():
    out = mark_contested("# Nota nuda\n\nSolo body.\n", REF)
    data, _, body = frontmatter.split(out)
    assert data is not None and data["contested"] is True
    assert data["contradictions"] == [REF]
    assert "Solo body." in body


def test_mark_contested_broken_yaml_untouched():
    broken = "---\ntags: [unclosed\n---\n\ncorpo\n"
    assert mark_contested(broken, REF) == broken


def test_mark_contested_preserves_existing_frontmatter():
    out = mark_contested(NOTE, REF)
    data, _, _ = frontmatter.split(out)
    assert data["parent note"] == "[[Farmacologia]]"
    assert data["tags"] == ["farmacologia"]
    assert data["AI"] is True


def test_marked_note_stays_lint_clean():
    before = ofm_lint(NOTE)["violations"]
    after = ofm_lint(mark_contested(NOTE, REF))["violations"]
    assert after == before == []


def test_contested_refs_roundtrip():
    assert contested_refs(NOTE) == []
    assert contested_refs(mark_contested(NOTE, REF)) == [REF]


# --- clear_contested (pure) --------------------------------------------------

def test_clear_contested_removes_flag_and_refs():
    cleared = clear_contested(mark_contested(NOTE, REF))
    data, _, body = frontmatter.split(cleared)
    assert not data.get("contested")
    assert "contradictions" not in data
    assert "Il dosaggio raccomandato" in body  # note content intact


def test_clear_contested_noop_on_clean_note():
    assert clear_contested(NOTE) == NOTE


def test_clear_contested_broken_yaml_untouched():
    broken = "---\ntags: [unclosed\n---\n\ncorpo\n"
    assert clear_contested(broken) == broken


# --- contested_callout (pure) ------------------------------------------------

def test_contested_callout_is_warning_and_quotes_claim():
    out = contested_callout("Il dosaggio è 50mg/die.", "appunti.md")
    assert out.startswith("> [!warning]")
    assert "appunti.md" in out
    assert "Il dosaggio è 50mg/die." in out


def test_contested_callout_multiline_claim_stays_in_callout():
    out = contested_callout("riga uno\nriga due", "x.md")
    for line in out.splitlines():
        assert line.startswith(">")


def test_contested_callout_lints_clean():
    body = "# T\n\ntesto\n\n" + contested_callout("claim", "x.md") + "\n"
    note = f"---\nAI: true\ntags:\n  - t\nlast modified: 2026, 07, 02\nrelated:\n  - \"[[H]]\"\n---\n\n{body}"
    assert ofm_lint(note)["violations"] == []


# --- patch executor integration ----------------------------------------------

@pytest.fixture()
def contested_op():
    from silica.kernel.ops import Op, OpType

    return Op(
        op=OpType.patch,
        heading="Dosaggio Warfarin",
        source_basename="appunti-cardiologia-2026.md",
        path="Farmacologia/Dosaggio Warfarin.md",
        snippet=contested_callout(
            "Il dosaggio raccomandato è 50mg/die.", "appunti-cardiologia-2026.md"
        ),
        contested_by=REF,
    )


def test_execute_patch_contested_marks_note(tmp_vault, contested_op):
    from silica.kernel.bulk import execute_one

    path = tmp_vault.note("Farmacologia/Dosaggio Warfarin.md", NOTE)
    res = execute_one(contested_op)
    assert res["success"]

    content = tmp_vault.read(path)
    assert contested_refs(content) == [REF]
    assert "> [!warning]" in content
    assert "Il dosaggio raccomandato è 5mg/die" in content  # original claim intact


def test_execute_patch_contested_idempotent(tmp_vault, contested_op):
    from silica.kernel.bulk import execute_one

    path = tmp_vault.note("Farmacologia/Dosaggio Warfarin.md", NOTE)
    execute_one(contested_op)
    res2 = execute_one(contested_op)
    assert res2.get("skipped") == "duplicate"

    content = tmp_vault.read(path)
    assert contested_refs(content) == [REF]           # one entry, not two
    assert content.count("> [!warning]") == 1          # one block, not two


def test_run_dedup_contradicts_end_to_end(tmp_vault):
    """Full path through the REAL commit_ops micro-gate (leash → validate →
    snapshot → write → lint): contested_by must survive the ops-file roundtrip
    and land on disk as callout + frontmatter."""
    from unittest.mock import patch

    from silica.capabilities.dedup import run_dedup, DedupDecision
    from silica.config import SilicaConfig
    from silica.kernel.workqueue import WorkItem

    tmp_vault.note("Farmacologia/Farmacologia.md", "# Farmacologia\n")
    path = tmp_vault.note("Farmacologia/Dosaggio Warfarin.md", NOTE)

    item = WorkItem(
        kind="dedup",
        target_path="Farmacologia/Dosaggio Warfarin.md",
        context={
            "concept": "Dosaggio Warfarin",
            "excerpt": "Il dosaggio raccomandato è 50mg/die.",
            "candidate": "Dosaggio Warfarin",
            "inbox_file": "Inbox/appunti-cardiologia-2026.md",
            "hub": "Farmacologia",
        },
        reason="borderline_similarity score=0.78",
    )
    decision = DedupDecision(
        verdict="contradicts",
        rationale="conflicting dosage",
        addition="Il dosaggio raccomandato è 50mg/die.",
    )
    with patch("silica.capabilities.dedup._decide_dedup", return_value=decision):
        res = run_dedup(item, SilicaConfig())

    assert res["status"] == "committed", res
    content = tmp_vault.read(path)
    assert contested_refs(content) == ["fonte: appunti-cardiologia-2026.md"]
    assert "> [!warning] Contradiction — from appunti-cardiologia-2026.md" in content
    assert "50mg/die" in content
    assert "5mg/die" in content  # original claim never touched


def test_execute_patch_without_contested_by_unchanged(tmp_vault):
    """Plain patches don't grow contested frontmatter."""
    from silica.kernel.bulk import execute_one
    from silica.kernel.ops import Op, OpType

    path = tmp_vault.note("Farmacologia/Dosaggio Warfarin.md", NOTE)
    execute_one(Op(
        op=OpType.patch,
        heading="Dosaggio Warfarin",
        source_basename="altra-fonte.md",
        path="Farmacologia/Dosaggio Warfarin.md",
        snippet="Dettaglio aggiuntivo.",
    ))
    assert contested_refs(tmp_vault.read(path)) == []
