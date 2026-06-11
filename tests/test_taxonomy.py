"""Tests for kernel/taxonomy.py — schema validation and best_folder() scoring."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from silica.kernel.taxonomy import FolderRule, Taxonomy


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestTaxonomyValidation:
    def test_empty_taxonomy(self):
        t = Taxonomy()
        assert t.rules == []
        assert t.uncategorized == "Uncategorized"
        assert t.scope == ""

    def test_from_dict_basic(self):
        data = {
            "version": 1,
            "uncategorized": "Misc",
            "rules": [
                {"folder": "Concepts/AI", "themes": ["machine learning"], "keywords": ["LLM"]},
            ],
        }
        t = Taxonomy.from_dict(data)
        assert len(t.rules) == 1
        assert t.rules[0].folder == "Concepts/AI"
        assert t.rules[0].themes == ["machine learning"]
        assert t.rules[0].keyword_set() == {"llm"}

    def test_from_yaml_roundtrip(self, tmp_path: Path):
        t = Taxonomy(
            rules=[
                FolderRule(folder="A/B", themes=["alpha", "beta"], keywords=["AB"]),
                FolderRule(folder="C/D", themes=["gamma"]),
            ],
            uncategorized="Other",
        )
        p = tmp_path / "taxonomy.yaml"
        t.to_yaml(p)
        loaded = Taxonomy.from_yaml(p)
        assert loaded.uncategorized == "Other"
        assert len(loaded.rules) == 2
        assert loaded.rules[0].folder == "A/B"

    def test_invalid_missing_folder_raises(self):
        with pytest.raises(Exception):
            Taxonomy.from_dict({"rules": [{"themes": ["x"]}]})  # folder missing → pydantic error

    def test_keyword_set_lowercased(self):
        r = FolderRule(folder="X", themes=[], keywords=["Foo", "BAR", "baz"])
        assert r.keyword_set() == {"foo", "bar", "baz"}

    def test_metadata_filter_validation(self):
        data = {
            "version": 1,
            "rules": [
                {
                    "folder": "Archive/2026",
                    "themes": ["notes"],
                    "metadata_filters": [
                        {"key": "date", "operator": "year_equals", "value": 2026}
                    ]
                }
            ]
        }
        t = Taxonomy.from_dict(data)
        assert len(t.rules) == 1
        assert len(t.rules[0].metadata_filters) == 1
        filt = t.rules[0].metadata_filters[0]
        assert filt.key == "date"
        assert filt.operator == "year_equals"
        assert filt.value == 2026

    def test_taxonomy_auto_scoping_rules_and_uncategorized(self):
        data = {
            "version": 1,
            "scope": "Agenti Autonomi",
            "uncategorized": "Uncategorized",
            "rules": [
                {"folder": "DeepSeek", "themes": ["deepseek"]},
                {"folder": "Agenti Autonomi/OpenAI", "themes": ["openai"]},
            ]
        }
        t = Taxonomy.from_dict(data)
        assert t.uncategorized == "Agenti Autonomi/Uncategorized"
        assert t.rules[0].folder == "Agenti Autonomi/DeepSeek"
        assert t.rules[1].folder == "Agenti Autonomi/OpenAI"


# ---------------------------------------------------------------------------
# best_folder() scoring
# ---------------------------------------------------------------------------

class TestBestFolder:
    @pytest.fixture
    def taxonomy(self) -> Taxonomy:
        return Taxonomy(
            rules=[
                FolderRule(
                    folder="Concepts/AI",
                    themes=["machine learning", "deep learning", "neural networks"],
                    keywords=["LLM", "GPT"],
                ),
                FolderRule(
                    folder="Concepts/Math",
                    themes=["linear algebra", "calculus", "statistics"],
                    keywords=[],
                ),
                FolderRule(
                    folder="Life/Cooking",
                    themes=["cooking", "recipe", "food"],
                    keywords=["risotto"],
                ),
            ],
            uncategorized="Misc",
        )

    def test_keyword_hit_wins(self, taxonomy: Taxonomy):
        folder, score = taxonomy.best_folder(themes=[], title="My LLM experiments")
        assert folder == "Concepts/AI"
        assert score >= 0.4

    def test_theme_overlap_wins(self, taxonomy: Taxonomy):
        folder, score = taxonomy.best_folder(
            themes=["machine learning", "neural networks"], title="A study note"
        )
        assert folder == "Concepts/AI"
        assert score > 0.0

    def test_uncategorized_when_no_match(self, taxonomy: Taxonomy):
        folder, score = taxonomy.best_folder(themes=[], title="random note")
        assert folder == "Misc"
        assert score == 0.0

    def test_multiple_keyword_hits_capped(self, taxonomy: Taxonomy):
        folder, score = taxonomy.best_folder(themes=[], title="GPT and LLM papers")
        assert folder == "Concepts/AI"
        assert score <= 1.0

    def test_cooking_theme_match(self, taxonomy: Taxonomy):
        folder, score = taxonomy.best_folder(themes=["cooking", "food"], title="dinner ideas")
        assert folder == "Life/Cooking"
        assert score > 0.0

    def test_empty_taxonomy_returns_uncategorized(self):
        empty = Taxonomy(uncategorized="Other")
        folder, score = empty.best_folder(themes=["ai", "ml"])
        assert folder == "Other"
        assert score == 0.0

    def test_folder_for_convenience(self, taxonomy: Taxonomy):
        folder = taxonomy.folder_for(themes=[], title="LLM benchmark")
        assert folder == "Concepts/AI"


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

class TestGenerateTaxonomyTool:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        # Ensure CONFIG.vault_path doesn't interfere with relative paths in tests
        from silica.config import CONFIG
        monkeypatch.setattr(CONFIG, "vault_path", "")

    def test_silica_generate_taxonomy_lists_files(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from silica.driver.base import NoteRef
        from silica.tools.composed import silica_generate_taxonomy

        with patch("silica.agent.llm.call_llm") as mock_call_llm, \
             patch("silica.driver.DRIVER") as mock_driver:
            
            mock_driver.list_files.return_value = [
                NoteRef(name="FIPA-ACL", path="Agenti Autonomi/FIPA-ACL.md"),
                NoteRef(name="GAIA", path="Agenti Autonomi/GAIA.md"),
                NoteRef(name="SomeImage", path="Agenti Autonomi/SomeImage.png"),  # Should be ignored
            ]
            
            mock_response = MagicMock()
            mock_response.text = textwrap.dedent("""\
                version: 1
                scope: "Agenti Autonomi"
                uncategorized: "Agenti Autonomi/Uncategorized"
                rules:
                  - folder: "Agenti Autonomi/Communication"
                    themes: ["FIPA-ACL"]
                    keywords: ["FIPA-ACL"]
            """)
            mock_call_llm.return_value = mock_response

            save_file = tmp_path / "taxonomy.yaml"
            res = silica_generate_taxonomy(
                user_intent="Organize agents",
                scope="Agenti Autonomi",
                save_path=str(save_file),
            )

            mock_driver.list_files.assert_called_once_with("Agenti Autonomi")
            assert mock_call_llm.called
            called_args = mock_call_llm.call_args[1]
            called_messages = called_args["messages"]
            user_content = called_messages[1]["content"]
            
            assert "- FIPA-ACL" in user_content
            assert "- GAIA" in user_content
            assert "- SomeImage" not in user_content
            
            assert "taxonomy" in res
            assert res["success"] is True
            assert save_file.exists()

    def test_silica_generate_taxonomy_merge_includes_existing(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from silica.tools.composed import silica_generate_taxonomy

        save_file = tmp_path / "taxonomy.yaml"
        Taxonomy(
            rules=[FolderRule(folder="Aziende/X", themes=["pratiche"], keywords=["acme"])],
            uncategorized="Misc",
        ).to_yaml(save_file)

        with patch("silica.agent.llm.call_llm") as mock_call_llm, \
             patch("silica.driver.DRIVER") as mock_driver:
            mock_driver.list_files.return_value = []

            mock_response = MagicMock()
            mock_response.text = textwrap.dedent("""\
                version: 1
                uncategorized: "Misc"
                rules:
                  - folder: "Aziende/X"
                    themes: ["pratiche"]
                    keywords: ["acme"]
                  - folder: "Commesse/2026"
                    themes: ["commesse"]
                    keywords: []
            """)
            mock_call_llm.return_value = mock_response

            res = silica_generate_taxonomy(
                user_intent="organizza per anno di commissione",
                save_path=str(save_file),
                merge=True,
            )

        user_content = mock_call_llm.call_args[1]["messages"][1]["content"]
        # The existing taxonomy and the merge directives must reach the LLM
        assert "Merge instructions" in user_content
        assert "Aziende/X" in user_content

        assert res["success"] is True
        assert res["rules_count"] == 2

    def test_silica_generate_taxonomy_merge_without_existing_file(self, tmp_path):
        """merge=True with no existing taxonomy behaves like a plain generation."""
        from unittest.mock import MagicMock, patch
        from silica.tools.composed import silica_generate_taxonomy

        save_file = tmp_path / "taxonomy.yaml"  # does not exist yet

        with patch("silica.agent.llm.call_llm") as mock_call_llm, \
             patch("silica.driver.DRIVER") as mock_driver:
            mock_driver.list_files.return_value = []

            mock_response = MagicMock()
            mock_response.text = textwrap.dedent("""\
                version: 1
                uncategorized: "Misc"
                rules:
                  - folder: "Aziende/X"
                    themes: ["pratiche"]
                    keywords: ["acme"]
            """)
            mock_call_llm.return_value = mock_response

            res = silica_generate_taxonomy(
                user_intent="metti le pratiche di Acme in Aziende/X",
                save_path=str(save_file),
                merge=True,
            )

        user_content = mock_call_llm.call_args[1]["messages"][1]["content"]
        assert "Merge instructions" not in user_content
        assert res["success"] is True
        assert save_file.exists()
