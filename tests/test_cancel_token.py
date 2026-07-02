"""Tests for cancel_token integration in the capability run() functions.

All LLM calls and commit_ops are patched so tests run without any external
dependencies.  The key invariants tested:

1. Pre-set token → run() returns {"status": "cancelled"} without calling the LLM.
2. Mid-run token → run() returns {"status": "cancelled"} at the next checkpoint.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from silica.kernel.workqueue import WorkItem
from silica.config import SilicaConfig
from silica.capabilities.dedup import run_dedup, DedupDecision
from silica.capabilities.refine import run_refine
from silica.capabilities.enrich import run_enrich
from silica.capabilities.orphan import run_orphan, OrphanLinkDecision
from silica.capabilities._base import NoteContent


def _item(kind: str, *, cancelled: bool = False) -> WorkItem:
    item = WorkItem(kind=kind, target_path="Notes/Test.md", context={"hub": "Hub"})
    if cancelled:
        item.cancel_token.set()
    return item


@pytest.fixture()
def config():
    return SilicaConfig()


class _MockNote:
    content = "existing body"


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def test_dedup_cancelled_before_llm(config):
    item = _item("dedup", cancelled=True)
    with patch("silica.driver.DRIVER.read_note", return_value=_MockNote()):
        result = run_dedup(item, config)
    assert result["status"] == "cancelled"


def test_dedup_cancelled_between_llm_and_commit(config):
    """Token is set after _decide_dedup returns (mid-run cancellation)."""
    item = _item("dedup")

    def fake_decide(*a, **k):
        item.cancel_token.set()   # simulate token being set during LLM call
        return DedupDecision(verdict="duplicate", rationale="dup", addition="new info")

    with patch("silica.driver.DRIVER.read_note", return_value=_MockNote()), \
         patch("silica.capabilities.dedup._decide_dedup", side_effect=fake_decide):
        result = run_dedup(item, config)

    assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------

def test_refine_cancelled_before_llm(config):
    item = _item("refine", cancelled=True)
    with patch("silica.driver.DRIVER.read_note", return_value=_MockNote()):
        result = run_refine(item, config)
    assert result["status"] == "cancelled"


def test_refine_cancelled_between_llm_and_commit(config):
    item = _item("refine")

    def fake_refine(*a, **k):
        item.cancel_token.set()
        return NoteContent(content="refined body")

    with patch("silica.driver.DRIVER.read_note", return_value=_MockNote()), \
         patch("silica.capabilities.refine._refine_note", side_effect=fake_refine):
        result = run_refine(item, config)

    assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Enrich
# ---------------------------------------------------------------------------

def test_enrich_cancelled_before_llm(config):
    item = _item("enrich", cancelled=True)
    with patch("silica.driver.DRIVER.read_note", return_value=_MockNote()):
        result = run_enrich(item, config)
    assert result["status"] == "cancelled"


def test_enrich_cancelled_between_llm_and_commit(config):
    item = _item("enrich")

    def fake_enrich(*a, **k):
        item.cancel_token.set()
        return NoteContent(content="enriched body")

    with patch("silica.driver.DRIVER.read_note", return_value=_MockNote()), \
         patch("silica.capabilities.enrich._enrich_note", side_effect=fake_enrich):
        result = run_enrich(item, config)

    assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Orphan
# ---------------------------------------------------------------------------

def test_orphan_cancelled_before_llm(config):
    item = _item("orphan", cancelled=True)
    item.context["candidates"] = [{"name": "Hub", "path": "Hub.md"}]
    with patch("silica.driver.DRIVER.read_note", return_value=_MockNote()):
        result = run_orphan(item, config)
    assert result["status"] == "cancelled"


def test_orphan_cancelled_between_llm_and_commit(config):
    item = _item("orphan")
    item.context["candidates"] = [{"name": "Hub", "path": "Hub.md"}]

    def fake_links(*a, **k):
        item.cancel_token.set()
        return OrphanLinkDecision(links=["Hub"], rationale="ok")

    with patch("silica.driver.DRIVER.read_note", return_value=_MockNote()), \
         patch("silica.capabilities.orphan._decide_links", side_effect=fake_links):
        result = run_orphan(item, config)

    assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Phase feedback events are published
# ---------------------------------------------------------------------------

def test_feedback_events_published_on_normal_run(config):
    """WorkFeedbackEvent is published to BUS for each phase on a normal run."""
    import silica.agent.bus as bus_mod
    received = []
    bus_mod.BUS.subscribe("work/feedback", received.append)

    item = _item("refine")

    with patch("silica.driver.DRIVER.read_note", return_value=_MockNote()), \
         patch("silica.capabilities.refine._refine_note", return_value=NoteContent(content="x")), \
         patch("silica.capabilities.refine.commit_ops", return_value={"status": "committed"}):
        run_refine(item, config)

    phases = [e.phase for e in received]
    assert phases == ["reading", "calling_llm", "committing"]
