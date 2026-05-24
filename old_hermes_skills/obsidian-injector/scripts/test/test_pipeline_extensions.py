import unittest
import sys
import os
import json
import tempfile
import datetime
from pathlib import Path

# --- hermes_common bootstrap (uniform across all hermes skills) ---
import os, sys
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "hermes_common")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
# --- end bootstrap ---

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Import modules to test
import parse_distiller_output
import validate_operations
from hermes_common import templates
import distiller_payload

# Import dedup finder and gather payload
sys.path.insert(0, os.path.join(_p, "obsidian-dedup", "scripts"))
import find_duplicates
import gather_merge_payload



class TestParseDistillerOutput(unittest.TestCase):
    def test_clean_json(self):
        raw = '{"updates": [{"heading": "A", "op": "skip"}]}'
        parsed, was_clean = parse_distiller_output.parse_json(raw, strict=False)
        self.assertTrue(was_clean)
        self.assertEqual(parsed["updates"][0]["heading"], "A")

    def test_json_with_fences(self):
        raw = '```json\n{"updates": [{"heading": "A", "op": "skip"}]}\n```'
        parsed, was_clean = parse_distiller_output.parse_json(raw, strict=False)
        self.assertFalse(was_clean)
        self.assertEqual(parsed["updates"][0]["heading"], "A")

        # In strict mode, this should raise ValueError
        with self.assertRaises(ValueError):
            parse_distiller_output.parse_json(raw, strict=True)

    def test_json_with_prose_noise(self):
        raw = 'Here is the JSON you requested:\n```json\n{"updates": [{"heading": "A", "op": "skip"}]}\n```\nHope this helps!'
        parsed, was_clean = parse_distiller_output.parse_json(raw, strict=False)
        self.assertFalse(was_clean)
        self.assertEqual(parsed["updates"][0]["heading"], "A")

        # In strict mode, this should raise ValueError
        with self.assertRaises(ValueError):
            parse_distiller_output.parse_json(raw, strict=True)


class TestTemplatesFrontmatterFallback(unittest.TestCase):
    def test_patch_snippet_with_frontmatter(self):
        existing = "---\nrelated:\n  - \"[[Old Hub]]\"\n---\nSome text"
        result = templates.patch_snippet(
            heading="Concept",
            snippet="New info",
            source_basename="inbox.md",
            hub="New Hub",
            existing_content=existing
        )
        self.assertIn("[[New Hub]]", result)
        self.assertIn("## Note aggiuntive — Concept", result)

    def test_patch_snippet_without_frontmatter_fallback(self):
        existing = "# Concept Title\nSome content without YAML."
        result = templates.patch_snippet(
            heading="Concept",
            snippet="New info",
            source_basename="inbox.md",
            hub="New Hub",
            existing_content=existing
        )
        
        # Verify it has created a valid YAML frontmatter
        self.assertTrue(result.startswith("---"))
        self.assertIn("parent note: \"[[New Hub]]\"", result)
        self.assertIn("related:", result)
        self.assertIn("- \"[[New Hub]]\"", result)
        self.assertIn("AI: true", result)
        self.assertIn("last modified:", result)
        
        # Verify the date is today
        today = datetime.date.today().strftime("%Y, %m, %d")
        self.assertIn(today, result)
        
        # Verify it still contains original text and patched snippet
        self.assertIn("# Concept Title\nSome content without YAML.", result)
        self.assertIn("## Note aggiuntive — Concept", result)


class TestValidateOperations(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.target = Path(self.tmp_dir.name) / "vault"
        self.target.mkdir()
        self.inbox_dir = Path(self.tmp_dir.name) / "inbox"
        self.inbox_dir.mkdir()

        # Create collision file in target vault
        self.collision_file = self.target / "Backpropagation.md"
        self.collision_file.write_text("Old content", encoding="utf-8")

        # Prepare dummy payload
        self.payload = {
            "schema_version": 1,
            "batches": [
                {
                    "inbox_file": str(self.inbox_dir / "Lezione 04.md"),
                    "concepts": [
                        {
                            "name": "Backpropagation",
                            "action_hint": "enrich",
                            "inbox_excerpt": "Backprop details...",
                            "vault_collision": {
                                "path": str(self.collision_file),
                                "match_type": "title"
                            }
                        },
                        {
                            "name": "Adam Optimizer",
                            "action_hint": "create",
                            "inbox_excerpt": "Adam details...",
                            "vault_collision": None
                        }
                    ]
                }
            ]
        }
        self.payload_file = Path(self.tmp_dir.name) / "payload.json"
        self.payload_file.write_text(json.dumps(self.payload), encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def run_validator(self, operations_list):
        ops_file = Path(self.tmp_dir.name) / "operations.json"
        ops_file.write_text(json.dumps({"updates": operations_list}), encoding="utf-8")

        validated_file = Path(self.tmp_dir.name) / "operations.validated.json"
        rejected_file = Path(self.tmp_dir.name) / "operations.rejected.json"

        # Mock sys.argv to call main()
        sys.argv = [
            "validate_operations.py",
            "--operations", str(ops_file),
            "--payload", str(self.payload_file),
            "--target", str(self.target),
            "--out", str(validated_file),
            "--rejected-out", str(rejected_file)
        ]

        try:
            validate_operations.main()
            exit_code = 0
        except SystemExit as e:
            exit_code = e.code

        validated = json.loads(validated_file.read_text(encoding="utf-8")) if validated_file.exists() else []
        rejected = json.loads(rejected_file.read_text(encoding="utf-8")) if rejected_file.exists() else []

        return exit_code, validated, rejected

    def test_valid_operations(self):
        # 1 valid patch, 1 valid write, 1 valid skip (should be ignored in validated.json)
        ops = [
            {
                "heading": "Backpropagation",
                "op": "patch",
                "path": str(self.collision_file),
                "source_basename": "Lezione 04.md",
                "snippet": "New backprop facts"
            },
            {
                "heading": "Adam Optimizer",
                "op": "write",
                "path": str(self.target / "Adam Optimizer.md"),
                "source_basename": "Lezione 04.md",
                "snippet": "New Adam facts"
            },
            {
                "heading": "Backpropagation",
                "op": "skip",
                "source_basename": "Lezione 04.md"
            }
        ]
        
        exit_code, validated, rejected = self.run_validator(ops)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(validated), 2)
        self.assertEqual(len(rejected), 0)
        
        # Verify skips are not in validated.json
        ops_headings = [o["heading"] for o in validated]
        self.assertIn("Backpropagation", ops_headings)
        self.assertIn("Adam Optimizer", ops_headings)

    def test_invalid_operations_rejected(self):
        # 1 valid, 1 invalid (hallucinated heading) -> 50% rejection (should exit with 2)
        ops = [
            {
                "heading": "Backpropagation",
                "op": "patch",
                "path": str(self.collision_file),
                "source_basename": "Lezione 04.md",
                "snippet": "New backprop facts"
            },
            {
                "heading": "Hallucinated Concept",
                "op": "write",
                "path": str(self.target / "Hallucinated.md"),
                "source_basename": "Lezione 04.md",
                "snippet": "Fake concept info"
            }
        ]

        exit_code, validated, rejected = self.run_validator(ops)
        self.assertEqual(exit_code, 2)
        self.assertEqual(len(validated), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["op"]["heading"], "Hallucinated Concept")
        self.assertIn("not present in payload concepts", rejected[0]["reason"])

    def test_write_outside_target_rejected(self):
        ops = [
            {
                "heading": "Adam Optimizer",
                "op": "write",
                "path": str(Path(self.tmp_dir.name) / "Outside.md"),
                "source_basename": "Lezione 04.md",
                "snippet": "New Adam facts"
            }
        ]
        exit_code, validated, rejected = self.run_validator(ops)
        self.assertEqual(exit_code, 2)
        self.assertEqual(len(validated), 0)
        self.assertEqual(len(rejected), 1)
        self.assertIn("is not within the target folder", rejected[0]["reason"])

    def test_patch_path_mismatch_rejected(self):
        # Patching to a different path than the vault_collision in payload
        ops = [
            {
                "heading": "Backpropagation",
                "op": "patch",
                "path": str(self.target / "WrongPath.md"),
                "source_basename": "Lezione 04.md",
                "snippet": "New backprop facts"
            }
        ]
        exit_code, validated, rejected = self.run_validator(ops)
        self.assertEqual(exit_code, 2)
        self.assertEqual(len(validated), 0)
        self.assertEqual(len(rejected), 1)
        self.assertIn("does not match expected collision path", rejected[0]["reason"])

    def test_inbox_path_segment_rejected(self):
        # Targeting inside the inbox folder is forbidden
        ops = [
            {
                "heading": "Adam Optimizer",
                "op": "write",
                "path": str(self.inbox_dir / "Adam.md"),
                "source_basename": "Lezione 04.md",
                "snippet": "New Adam facts"
            }
        ]
        exit_code, validated, rejected = self.run_validator(ops)
        self.assertEqual(exit_code, 2)
        self.assertEqual(len(validated), 0)
        self.assertEqual(len(rejected), 1)
        self.assertIn("contains or points to a forbidden inbox directory segment", rejected[0]["reason"])

    def test_coercion_write_to_patch(self):
        # 1. Coerce write to patch if path exists on disk (with collision in payload)
        # target / "Backpropagation.md" exists, so write should be coerced to patch.
        ops = [
            {
                "heading": "Backpropagation",
                "op": "write",
                "path": str(self.collision_file),
                "source_basename": "Lezione 04.md",
                "snippet": "Coerced write to patch"
            }
        ]
        exit_code, validated, rejected = self.run_validator(ops)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["op"], "patch")

        # 1b. Coerce write to patch if path exists on disk but has NO collision in payload
        # Create an untracked existing file in the target
        untracked_file = self.target / "Adam Optimizer.md"
        untracked_file.write_text("Existing Adam content", encoding="utf-8")
        ops_untracked = [
            {
                "heading": "Adam Optimizer",
                "op": "write",
                "path": str(untracked_file),
                "source_basename": "Lezione 04.md",
                "snippet": "Coerced untracked write to patch"
            }
        ]
        exit_code, validated, rejected = self.run_validator(ops_untracked)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["op"], "patch")
        
        # Clean up untracked file
        if untracked_file.exists():
            untracked_file.unlink()

        # 2. Coerce patch to write if path does NOT exist on disk
        # target / "Adam Optimizer.md" does not exist, so patch should be coerced to write.
        ops2 = [
            {
                "heading": "Adam Optimizer",
                "op": "patch",
                "path": str(self.target / "Adam Optimizer.md"),
                "source_basename": "Lezione 04.md",
                "snippet": "Coerced patch to write"
            }
        ]
        exit_code, validated, rejected = self.run_validator(ops2)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["op"], "write")

    def test_global_dedup_cross_batch(self):
        # Multiple operations targeting same path. One with longer snippet wins.
        # Winner is kept, other is degraded to skip.
        ops = [
            {
                "heading": "Adam Optimizer",
                "op": "write",
                "path": str(self.target / "Adam Optimizer.md"),
                "source_basename": "Lezione 04.md",
                "snippet": "Short snippet"
            },
            {
                "heading": "Adam Optimizer",
                "op": "write",
                "path": str(self.target / "Adam Optimizer.md"),
                "source_basename": "Lezione 04.md",
                "snippet": "Longer and richer snippet content"
            }
        ]
        exit_code, validated, rejected = self.run_validator(ops)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["snippet"], "Longer and richer snippet content")


class TestDistillerPayload(unittest.TestCase):
    def test_expand_to_double_newline(self):
        content = "Para 1.\n\nPara 2 (concept match).\n\nPara 3."
        # Match starts at index 9 ('Para 2') and ends at 32
        start, end = 9, 32
        ns, ne = distiller_payload.expand_to_double_newline(content, start, end)
        self.assertEqual(ns, 9)
        self.assertEqual(ne, 32)
        self.assertEqual(content[ns:ne], "Para 2 (concept match).")

    def test_safe_truncate(self):
        text = "This is a paragraph.\n\nThis is a code block:\n```python\nprint(1)\n```\n\nFinal block."
        # Truncating with limit that lands around code block end (char index ~65)
        truncated = distiller_payload.safe_truncate(text, 68)
        self.assertTrue(truncated.endswith("```"))
        self.assertNotIn("Final block.", truncated)
        
        # Fallback to hard limit if max_chars is extremely short
        short_truncated = distiller_payload.safe_truncate(text, 10)
        self.assertEqual(short_truncated, "This is a")


class TestFindDuplicates(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp_dir.name) / "vault"
        self.vault.mkdir()
        self.sub1 = self.vault / "sub1"
        self.sub1.mkdir()
        self.sub2 = self.vault / "sub2"
        self.sub2.mkdir()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_find_semantic_duplicates(self):
        # Create "Sub-word Tokenization" and "Subword Tokenization (NLP)"
        f1 = self.sub1 / "Sub-word Tokenization.md"
        f1.write_text("content 1", encoding="utf-8")
        f2 = self.sub2 / "Subword Tokenization (NLP).md"
        f2.write_text("content 2", encoding="utf-8")

        # Create "Word form" and "Wordform (NLP)"
        f3 = self.sub1 / "Word form.md"
        f3.write_text("content 3", encoding="utf-8")
        f4 = self.sub2 / "Wordform (NLP).md"
        f4.write_text("content 4", encoding="utf-8")

        # Create a non-duplicate file
        f5 = self.sub1 / "Unique Note.md"
        f5.write_text("content 5", encoding="utf-8")

        dupes = find_duplicates.find_duplicates(str(self.vault))
        
        self.assertEqual(len(dupes), 2)
        
        # Verify "Sub-word Tokenization" is grouped
        self.assertIn("Sub-word Tokenization", dupes)
        self.assertEqual(set(dupes["Sub-word Tokenization"]), {str(f1.resolve()), str(f2.resolve())})
        
        # Verify "Word form" is grouped
        self.assertIn("Word form", dupes)
        self.assertEqual(set(dupes["Word form"]), {str(f3.resolve()), str(f4.resolve())})

    def test_find_duplicates_with_leading_numbers_and_typos(self):
        # Create "5. L'ambiente e le sue propietà" and "L'ambiente e le sue proprietà"
        f1 = self.sub1 / "5. L'ambiente e le sue propietà.md"
        f1.write_text("content 1", encoding="utf-8")
        f2 = self.sub2 / "L'ambiente e le sue proprietà.md"
        f2.write_text("content 2", encoding="utf-8")

        # Create "02. La cellula" and "2. La cellula"
        f3 = self.sub1 / "02. La cellula.md"
        f3.write_text("content 3", encoding="utf-8")
        f4 = self.sub2 / "2. La cellula.md"
        f4.write_text("content 4", encoding="utf-8")

        # Create date files that should not be duplicates
        f5 = self.sub1 / "2026-05-24.md"
        f5.write_text("content 5", encoding="utf-8")
        f6 = self.sub2 / "2025-05-24.md"
        f6.write_text("content 6", encoding="utf-8")

        # Create two identical date files that should be duplicates
        f7 = self.sub1 / "2026-05-25.md"
        f7.write_text("content 7", encoding="utf-8")
        f8 = self.sub2 / "2026-05-25.md"
        f8.write_text("content 8", encoding="utf-8")

        # Create non-duplicates with different acronyms (e.g. "Metodologia AAII" vs "Metodologia GAIA")
        f9 = self.sub1 / "Metodologia AAII.md"
        f9.write_text("content 9", encoding="utf-8")
        f10 = self.sub2 / "Metodologia GAIA.md"
        f10.write_text("content 10", encoding="utf-8")

        dupes = find_duplicates.find_duplicates(str(self.vault))

        # We should find both groups
        self.assertIn("L'ambiente e le sue proprietà", dupes)
        self.assertEqual(set(dupes["L'ambiente e le sue proprietà"]), {str(f1.resolve()), str(f2.resolve())})

        self.assertIn("2. La cellula", dupes)
        self.assertEqual(set(dupes["2. La cellula"]), {str(f3.resolve()), str(f4.resolve())})

        # Check date files behavior
        self.assertNotIn("2026-05-24", dupes)
        self.assertNotIn("2025-05-24", dupes)
        self.assertIn("2026-05-25", dupes)
        self.assertEqual(set(dupes["2026-05-25"]), {str(f7.resolve()), str(f8.resolve())})

        # Check that different acronyms are not matched
        self.assertNotIn("Metodologia AAII", dupes)
        self.assertNotIn("Metodologia GAIA", dupes)

    def test_gather_payload_with_dates(self):
        # Create a duplicate with frontmatter date
        f1 = self.sub1 / "Date Note.md"
        f1.write_text("---\nlast modified: 2026-05-23\ntags:\n  - test\n---\nBody 1", encoding="utf-8")
        
        f2 = self.sub2 / "Date Note.md"
        f2.write_text("---\nlast modified: 2026-05-23\ntags:\n  - test\n---\nBody 2", encoding="utf-8")
        
        dupes = find_duplicates.find_duplicates(str(self.vault))
        self.assertIn("Date Note", dupes)
        
        payload = gather_merge_payload.build(dupes)
        
        # This will raise TypeError if datetime.date is not JSON serializable
        serialized = json.dumps(payload, default=str)
        deserialized = json.loads(serialized)
        
        self.assertEqual(len(deserialized["groups"]), 1)
        self.assertEqual(deserialized["groups"][0]["basename"], "Date Note")


if __name__ == "__main__":
    unittest.main()
