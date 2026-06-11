"""Smoke tests for the /organize pipeline FSM and classifier.

These tests run without a live vault driver — they use monkey-patching
and in-memory data to validate the pipeline mechanics.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from silica.kernel.classify import Classification, classify_notes
from silica.kernel.taxonomy import FolderRule, Taxonomy


# ---------------------------------------------------------------------------
# Classifier unit tests (L1 only, no driver, no LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_taxonomy() -> Taxonomy:
    return Taxonomy(
        rules=[
            FolderRule(folder="Concepts/AI", themes=["machine learning"], keywords=["GPT"]),
            FolderRule(folder="Life/Food", themes=["cooking", "food"], keywords=["risotto"]),
        ],
        uncategorized="Misc",
    )


class TestClassifyNotes:
    def test_keyword_match_classifies_correctly(self, simple_taxonomy: Taxonomy):
        """A note whose title contains a keyword should be classified to the matching folder."""
        results = classify_notes(
            ["Concepts/GPT-4 review.md"],
            simple_taxonomy,
            cooccur_store=None,    # no index → keyword-only
            llm_arbiter=False,
        )
        assert len(results) == 1
        c = results[0]
        assert c.target_folder == "Concepts/AI"
        assert c.evidence == "keyword"

    def test_no_match_goes_to_uncategorized(self, simple_taxonomy: Taxonomy):
        results = classify_notes(
            ["random-note.md"],
            simple_taxonomy,
            cooccur_store=None,
            llm_arbiter=False,
        )
        assert results[0].target_folder == "Misc"

    def test_needs_move_false_when_already_in_target(self, simple_taxonomy: Taxonomy):
        """A note already in the target folder should have needs_move=False."""
        results = classify_notes(
            ["Concepts/AI/my note.md"],
            simple_taxonomy,
            cooccur_store=None,
            llm_arbiter=False,
        )
        # Even if it matches Concepts/AI, target == current → needs_move should be False
        c = results[0]
        if c.target_folder == "Concepts/AI":
            assert not c.needs_move

    def test_empty_note_list(self, simple_taxonomy: Taxonomy):
        results = classify_notes([], simple_taxonomy, cooccur_store=None, llm_arbiter=False)
        assert results == []

    def test_empty_taxonomy_all_uncategorized(self):
        empty = Taxonomy(uncategorized="Other")
        results = classify_notes(
            ["note-a.md", "note-b.md"],
            empty,
            cooccur_store=None,
            llm_arbiter=False,
        )
        assert all(c.target_folder == "Other" for c in results)

    def test_risotto_keyword(self, simple_taxonomy: Taxonomy):
        results = classify_notes(
            ["Risotto al parmigiano.md"],
            simple_taxonomy,
            cooccur_store=None,
            llm_arbiter=False,
        )
        assert results[0].target_folder == "Life/Food"

    def test_classification_count_matches_input(self, simple_taxonomy: Taxonomy):
        paths = ["a.md", "b.md", "c.md"]
        results = classify_notes(paths, simple_taxonomy, cooccur_store=None, llm_arbiter=False)
        assert len(results) == len(paths)

    def test_metadata_filter_year_equals(self):
        from silica.kernel.taxonomy import FolderRule, Taxonomy, MetadataFilter
        tax = Taxonomy(
            rules=[
                FolderRule(
                    folder="Archive/2026",
                    themes=[],
                    keywords=["report"],
                    metadata_filters=[
                        MetadataFilter(key="date", operator="year_equals", value=2026)
                    ]
                )
            ],
            uncategorized="Misc"
        )
        # Note matching both keyword AND year filter
        results = classify_notes(
            ["report_a.md"],
            tax,
            cooccur_store=None,
            llm_arbiter=False,
            props_map={"report_a.md": {"date": "2026-06-08"}}
        )
        assert results[0].target_folder == "Archive/2026"

        # Note matching keyword but WRONG year (should be excluded/Misc)
        results2 = classify_notes(
            ["report_b.md"],
            tax,
            cooccur_store=None,
            llm_arbiter=False,
            props_map={"report_b.md": {"date": "2025-12-31"}}
        )
        assert results2[0].target_folder == "Misc"

    def test_metadata_filter_other_operators(self):
        from silica.kernel.taxonomy import FolderRule, Taxonomy, MetadataFilter
        tax = Taxonomy(
            rules=[
                FolderRule(
                    folder="Recent",
                    themes=[],
                    keywords=["doc"],
                    metadata_filters=[
                        MetadataFilter(key="year", operator="year_greater_than", value=2025)
                    ]
                ),
                FolderRule(
                    folder="Recipe",
                    themes=[],
                    keywords=["pie"],
                    metadata_filters=[
                        MetadataFilter(key="tags", operator="contains", value="dessert")
                    ]
                )
            ],
            uncategorized="Misc"
        )

        # year > 2025 (e.g. 2026) -> Recent
        r1 = classify_notes(
            ["doc1.md"], tax, cooccur_store=None, llm_arbiter=False,
            props_map={"doc1.md": {"year": 2027}}
        )
        assert r1[0].target_folder == "Recent"

        # year <= 2025 (e.g. 2024) -> Misc
        r2 = classify_notes(
            ["doc2.md"], tax, cooccur_store=None, llm_arbiter=False,
            props_map={"doc2.md": {"year": 2024}}
        )
        assert r2[0].target_folder == "Misc"

        # tags contains dessert -> Recipe
        r3 = classify_notes(
            ["apple pie.md"], tax, cooccur_store=None, llm_arbiter=False,
            props_map={"apple pie.md": {"tags": ["sweet", "dessert"]}}
        )
        assert r3[0].target_folder == "Recipe"

    def test_metadata_filter_fs_fallback(self):
        from silica.kernel.taxonomy import FolderRule, Taxonomy, MetadataFilter
        tax = Taxonomy(
            rules=[
                FolderRule(
                    folder="Archive/2026",
                    themes=[],
                    keywords=["report"],
                    metadata_filters=[
                        MetadataFilter(key="created", operator="year_equals", value=2026)
                    ]
                )
            ],
            uncategorized="Misc"
        )

        # Key is missing, but fallback to fs returning 2026 -> Archive/2026
        with patch("silica.kernel.classify._get_file_fs_year", return_value=2026):
            results = classify_notes(
                ["report_a.md"],
                tax,
                cooccur_store=None,
                llm_arbiter=False,
                props_map={"report_a.md": {}}  # no 'created' property
            )
            assert results[0].target_folder == "Archive/2026"

    def test_uncategorized_stays_put_by_default(self, simple_taxonomy: Taxonomy):
        """A note matching no rule keeps its place — no move toward uncategorized."""
        with patch("silica.driver.DRIVER"):
            results = classify_notes(
                ["Inbox/random-note.md"],
                simple_taxonomy,
                cooccur_store=None,
                llm_arbiter=False,
            )
        c = results[0]
        assert c.target_folder == "Misc"
        assert not c.needs_move

    def test_move_uncategorized_opt_in(self, simple_taxonomy: Taxonomy):
        """With move_uncategorized=True the legacy collect-into-uncategorized behavior returns."""
        with patch("silica.driver.DRIVER"):
            results = classify_notes(
                ["Inbox/random-note.md"],
                simple_taxonomy,
                cooccur_store=None,
                llm_arbiter=False,
                move_uncategorized=True,
            )
        c = results[0]
        assert c.target_folder == "Misc"
        assert c.needs_move

    def test_body_fallback_when_index_empty(self, simple_taxonomy: Taxonomy):
        """A note absent from the cooccur index is classified by tokenizing its body."""
        mock_driver = MagicMock()
        mock_driver.read_note.return_value = MagicMock(
            content=(
                "Machine learning and deep learning. "
                "Machine learning models. "
                "Machine learning everywhere."
            )
        )
        with patch("silica.driver.DRIVER", mock_driver):
            results = classify_notes(
                ["untitled-note.md"],
                simple_taxonomy,
                cooccur_store=None,    # no index → body tokenization fallback
                llm_arbiter=False,
            )
        c = results[0]
        assert c.target_folder == "Concepts/AI"
        assert c.evidence == "cooccur"

    def test_body_fallback_read_failure_degrades_to_uncategorized(self, simple_taxonomy: Taxonomy):
        """If the body can't be read either, the note degrades to uncategorized in place."""
        mock_driver = MagicMock()
        mock_driver.read_note.side_effect = RuntimeError("vault unreachable")
        with patch("silica.driver.DRIVER", mock_driver):
            results = classify_notes(
                ["untitled-note.md"],
                simple_taxonomy,
                cooccur_store=None,
                llm_arbiter=False,
            )
        c = results[0]
        assert c.target_folder == "Misc"
        assert not c.needs_move

    def test_metadata_filter_case_insensitivity(self):
        from silica.kernel.taxonomy import FolderRule, Taxonomy, MetadataFilter
        tax = Taxonomy(
            rules=[
                FolderRule(
                    folder="Archive/2026",
                    themes=[],
                    keywords=["report"],
                    metadata_filters=[
                        MetadataFilter(key="Date", operator="year_equals", value=2026)
                    ]
                )
            ],
            uncategorized="Misc"
        )
        # Note matching both keyword AND year filter but property is lowercase 'date'
        results = classify_notes(
            ["report_a.md"],
            tax,
            cooccur_store=None,
            llm_arbiter=False,
            props_map={"report_a.md": {"date": "2026-06-08"}}
        )
        assert results[0].target_folder == "Archive/2026"


# ---------------------------------------------------------------------------
# OrganizerFSM smoke test (dry_run=True — no driver writes)
# ---------------------------------------------------------------------------

class TestOrganizerFSMDryRun:
    """Smoke test: dry_run=True should return plan without any DRIVER.move() calls."""

    @pytest.fixture
    def taxonomy(self) -> Taxonomy:
        return Taxonomy(
            rules=[
                FolderRule(folder="AI", themes=[], keywords=["gpt"]),
                FolderRule(folder="Food", themes=[], keywords=["risotto"]),
            ],
            uncategorized="Misc",
        )

    def test_dry_run_returns_plan(self, taxonomy: Taxonomy):
        from silica.router.organize_fsm import OrganizerFSM

        mock_refs = [
            MagicMock(path="GPT notes.md", name="GPT notes"),
            MagicMock(path="Random.md", name="Random"),
        ]

        with patch("silica.router.organize_fsm.DRIVER") as mock_driver:
            mock_driver.list_files.return_value = mock_refs

            fsm = OrganizerFSM(taxonomy=taxonomy, dry_run=True, llm_arbiter=False)
            result = fsm.run()

        assert result.get("final_status") == "DryRun"
        assert "plan_summary" in result
        assert isinstance(result["plan_summary"]["moves_planned"], int)
        # DRIVER.move must NOT have been called
        mock_driver.move.assert_not_called()

    def test_dry_run_no_moves_when_all_uncategorized(self, taxonomy: Taxonomy):
        from silica.router.organize_fsm import OrganizerFSM

        # "xyzabc-completely-unrelated.md" has no keyword ("gpt", "risotto") in title
        # → L1 keyword-only classification sends it to Misc
        mock_refs = [MagicMock(path="xyzabc-completely-unrelated.md", name="xyzabc")]

        with patch("silica.router.organize_fsm.DRIVER") as mock_driver:
            mock_driver.list_files.return_value = mock_refs

            fsm = OrganizerFSM(taxonomy=taxonomy, dry_run=True, llm_arbiter=False)
            result = fsm.run()

        # The key invariant: DRIVER.move is NEVER called in dry_run mode
        mock_driver.move.assert_not_called()
        assert "plan_summary" in result


    def test_dry_run_uncategorized_stays_put(self, taxonomy: Taxonomy):
        """Unmatched notes produce no planned moves by default."""
        from silica.router.organize_fsm import OrganizerFSM

        mock_refs = [MagicMock(path="xyzabc-completely-unrelated.md", name="xyzabc")]

        with patch("silica.router.organize_fsm.DRIVER") as mock_driver, \
             patch("silica.driver.DRIVER"):
            mock_driver.list_files.return_value = mock_refs

            fsm = OrganizerFSM(taxonomy=taxonomy, dry_run=True, llm_arbiter=False)
            result = fsm.run()

        assert result["plan_summary"]["moves_planned"] == 0

    def test_dry_run_move_uncategorized_opt_in(self, taxonomy: Taxonomy):
        """With move_uncategorized=True the unmatched note is planned into Misc."""
        from silica.router.organize_fsm import OrganizerFSM

        mock_refs = [MagicMock(path="xyzabc-completely-unrelated.md", name="xyzabc")]

        with patch("silica.router.organize_fsm.DRIVER") as mock_driver, \
             patch("silica.driver.DRIVER"):
            mock_driver.list_files.return_value = mock_refs

            fsm = OrganizerFSM(
                taxonomy=taxonomy, dry_run=True, llm_arbiter=False, move_uncategorized=True
            )
            result = fsm.run()

        assert result["plan_summary"]["moves_planned"] == 1
        assert result["plan_summary"]["plan"][0]["to"] == "Misc/xyzabc-completely-unrelated.md"

    def test_fsm_scan_error_transitions_to_error(self, taxonomy: Taxonomy):
        from silica.router.organize_fsm import OrganizerFSM, OrganizerState

        with patch("silica.router.organize_fsm.DRIVER") as mock_driver:
            mock_driver.list_files.side_effect = RuntimeError("vault unreachable")

            fsm = OrganizerFSM(taxonomy=taxonomy, dry_run=True, llm_arbiter=False)
            fsm.run()

        assert fsm.state == OrganizerState.ERROR


class TestOrganizerFSMRollback:
    def test_rollback_only_moves_successful_ones_back(self):
        from silica.router.organize_fsm import OrganizerFSM, OrganizerState
        from silica.driver.base import NoteRef
        from silica.kernel.taxonomy import Taxonomy, FolderRule

        taxonomy = Taxonomy(
            rules=[
                FolderRule(folder="AI", themes=[], keywords=["gpt"]),
            ],
            uncategorized="Misc",
        )

        mock_refs = [
            NoteRef(name="gpt-success", path="gpt-success.md"),
            NoteRef(name="gpt-fail", path="gpt-fail.md"),
        ]

        with patch("silica.router.organize_fsm.DRIVER") as mock_driver:
            mock_driver.list_files.return_value = mock_refs
            
            # Mock move to succeed for "gpt-success.md" and fail for "gpt-fail.md"
            def mock_move(src, dst):
                if "gpt-fail" in src:
                    raise RuntimeError("Permission denied / disk full")
                return None
            mock_driver.move.side_effect = mock_move
            
            # Setup graph snapshot mocks to return empty/basic structures
            from silica.driver.base import GraphSnapshot
            mock_driver.graph_snapshot.return_value = GraphSnapshot()

            fsm = OrganizerFSM(taxonomy=taxonomy, dry_run=False, llm_arbiter=False)
            result = fsm.run()

        # The failure rate is 1/2 = 50% > move_failure_max (10% default), so it must abort and rollback.
        assert "Rolled Back:" in result.get("final_status", "")

        # Check all move calls
        move_calls = mock_driver.move.call_args_list
        assert len(move_calls) == 3
        
        # Call 1: move gpt-success to AI/gpt-success
        assert move_calls[0][0] == ("gpt-success.md", "AI/gpt-success.md")
        # Call 2: move gpt-fail to AI/gpt-fail
        assert move_calls[1][0] == ("gpt-fail.md", "AI/gpt-fail.md")
        # Call 3: rollback gpt-success from AI/gpt-success back to gpt-success
        assert move_calls[2][0] == ("AI/gpt-success.md", "gpt-success.md")



