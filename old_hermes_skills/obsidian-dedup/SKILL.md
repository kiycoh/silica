---
name: obsidian-dedup
description: "Deduplication & Unification Pipeline — identifies duplicate notes of the same name located in different folders across the vault, and merges their contents smoothly into a single canonical note without losing technical density."
---

# Obsidian Dedup — Deduplication Pipeline

The Deduplication pipeline is used to find and resolve duplicate note files within the vault. In Obsidian, while you cannot have files with the exact same name in the same directory, you can have duplicates in different folders (e.g. `1.1 Informazione/MLP.md` and `1.2 Calcolo/MLP.md`).

## Inputs

- `<VAULT_ROOT>`: path to the root of the Obsidian vault.
- `<FOLDER_PATH>` (Optional): specific subdirectory to restrict the duplicate scan.

## Required Tools

This skill requires:
- **`find_duplicates.py`** (executed via `execute_code`) to locate identical note basenames across the vault, optionally scoped using the `--folder` parameter.
- **`web_search` & `web_extract`** (native tools called directly by the model) to verify facts, definitions, or correct formulas.
- **`write_file` & `patch`** (native file operation tools) to commit updates to the vault.

## Deduplication Workflow

**Warning:** Vault paths frequently contain spaces. When constructing commands in `execute_code`, always use `shell_quote()` for all path arguments to prevent `argparse` errors.

- **Phase 1 — Locate Duplicates**:
  Run the duplicate locator script using `execute_code`:
  ```bash
  python3 ~/.hermes/skills/note-taking/obsidian-dedup/scripts/find_duplicates.py --vault "<VAULT_ROOT>" [--folder "<SUBDIRECTORY_PATH>"] > /tmp/dupes.json
  ```
- **Phase 1→2 Bridge — Gather Payload**:
  Bundle all duplicate note contents into a single payload using the gather script:
  ```bash
  python3 ~/.hermes/skills/note-taking/obsidian-dedup/scripts/gather_merge_payload.py --duplicates /tmp/dupes.json --out /tmp/merge_payload.json
  ```
- **Phase 2 — Semantic Unification**:
  Read `/tmp/merge_payload.json` in one shot. Select a single canonical target path (usually the most relevant subdirectory). Integrate all facts, definitions, and formatting from the duplicates into a single, cohesive canonical note body without losing technical details or references. **Crucially, synthesize the YAML frontmatter by merging tags, related notes, and parent references from all variants, and ensure `AI: true` is preserved if present.** Plan bulk writer operations: one `overwrite` operation for the canonical note, and one `delete` operation per obsolete duplicate file.
- **Phase 3 — Execution & Cleanup**:
  Apply the plan via the bulk writer:
  ```bash
  python3 ~/.hermes/skills/note-taking/hermes_common/bulk_writer.py --operations /tmp/dedup_ops.json
  ```
- **Phase 4 — Validate**:
  Run the common linter to ensure the merged note meets atomicity and YAML frontmatter rules:
  ```bash
  python3 ~/.hermes/skills/note-taking/hermes_common/linter.py --operations /tmp/dedup_ops.json
  ```
  **[EMOTION PROMPT: Semantic unification is a high-risk operation. Verify that absolutely no factual density was lost in the merge and that the linter passes flawlessly. If there is any doubt about data loss, abort the merge immediately.]**

## Content Preservation & Deletion Rules

- **Strict Anti-Deletion Policy**: Deleting existing information during merge consolidation is **strictly discouraged** unless:
  1. The information is pure semantic/formatting noise.
  2. The model is rewriting/expanding that same concept in a more thorough, detailed, and academically rigorous manner.
  3. The model has verified via `web_search` that the original phrase, definition, or formula is factually incorrect.
- **Frontmatter Preservation**: Failure to synthesize and include the YAML frontmatter (especially `AI: true` and `parent note`) will cause `linter.py` to fail. Always merge the metadata from all variants into the canonical note.
- **Atomicity Risk**: Merging multiple notes may create a "monolith" that exceeds the vault's length limits. If the resulting note is too long, the linter will fail; in such cases, treat the canonical note as a candidate for the `obsidian-refiner` pipeline.
- **Enrichment Trigger**: If the unified note contains **fewer than 600 characters**, it must be enriched with external web sources.
