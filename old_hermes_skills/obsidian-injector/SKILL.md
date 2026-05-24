---
name: obsidian-injector
description: "Inject Pipeline — ingests source markdown from an inbox into the Obsidian vault. Performs mechanical recon via execute_code, reasoning in-context, and file writes directly from the Router."
---

# Obsidian Injector — Inject Pipeline

## Curation Standards
The curation process must adhere strictly to these principles:
- **Factual Density**: Extract all concrete details, definitions, schemas, and examples from the source files. Avoid generalizations or hand-wavy summaries.
- **Modular Atomicity**: Avoid monolithic files. Split information into specific, granular concepts (Spoke notes) of roughly ~40 lines to ensure high-resolution modularity.
- **YAML Frontmatter**: Maintain consistent tagging style: lowercase, hyphen-separated tags describing the semantic areas (e.g. `intelligenza-artificiale`, `machine-learning`, `reti-neurali`).
- **Scholarly Readability**: Write in formal Italian, using bold keywords, clear structures, lists, and callout blocks (`> [!TIP]`) to make content highly usable for scholars and researchers.
- **Content Preservation Guardrail**: Deleting entire notes, sentences, or words without a logical and heavily weighed reason is strictly discouraged. Rather than deleting information, prioritize unifying and merging the incoming inbox content smoothly into the target vault note without losing any factual density.

## Optimization: Bulk Execution
Use `execute_code` for all mechanical tasks:
- **Phase 1**: Execute `scripts/recon.py` to iterate inbox and search vault.
- **Phase 2.0**: Execute `scripts/distiller_payload.py` to pre-distill the payload context json.
- **Phase 3**: Execute mutations programmatically via `<COMMON_DIR>/bulk_writer.py` using the validated operations JSON.
- **Phase 4**: Static linting of modified files using `<COMMON_DIR>/linter.py` (targeting operations via `--operations`).
- **Phase 5**: Move successfully processed/written inbox files to `<INBOX>/done/` immediately after each successful validation to ensure idempotency.
**[EMOTION PROMPT: This cleanup step is the backbone of idempotency. Do not assume the script completed perfectly; verify its success. Treat file tracking with absolute strictness to prevent duplicates on resume.]**

## Inputs

- `<INBOX>`: folder with source .md files
- `<TARGET>`: destination folder inside the vault
- `<HUB_NAME>`: the Hub note that Spokes link back to (e.g. "Computer Vision")

## Required bundled skill

The Router must have access to the bundled `obsidian` skill. Confirm via:
`skills_list | grep -i obsidian`. If missing, install:
`hermes skills install official/note-taking/obsidian`.

## Scripts & References

- `scripts/recon.py` — Phase 1 engine. Run via `execute_code`: `python3 ~/.hermes/skills/note-taking/obsidian-injector/scripts/recon.py --inbox "<INBOX>" --vault "<VAULT_ROOT>"`
- `scripts/distiller_payload.py` — Phase 2.0 pre-distiller. Run via `execute_code`: `python3 ~/.hermes/skills/note-taking/obsidian-injector/scripts/distiller_payload.py --recon-report /tmp/recon.json --out /tmp/distiller_payload.json`
- `<COMMON_DIR>/bulk_writer.py` — Phase 3 bulk writer. Run via `execute_code`: `python3 ~/.hermes/skills/note-taking/hermes_common/bulk_writer.py --operations "<PATH_TO_OPS_JSON>"`
- `<COMMON_DIR>/linter.py` — Phase 4 validator. Run via `execute_code`: `python3 ~/.hermes/skills/note-taking/hermes_common/linter.py [--target "<TARGET>" | --operations "<PATH_TO_OPS_JSON>"] --hub "<HUB_NAME>"`
- `<COMMON_DIR>/templates.py` — markdown templates (template_spoke, patch_snippet).


## Ambient Discovery

To discover the directory structure of the `<INBOX>` or `<TARGET>` folders cleanly:
- **Do not** use `search_files` with `*` or generic patterns without a path scope, as it will return workspace root internals and `.git` repository objects.
- **Do** list files in the target directory using shell commands:
  ```bash
  find "/path/to/dir" -maxdepth 2 -not -path '*/.*' -name '*.md'
  ```
- **Do** list files programmatically inside `execute_code` using Python:
  ```python
  from pathlib import Path
  print([str(p) for p in Path("/path/to/dir").glob("**/*.md")])
  ```

## Pitfalls
- **read_file Deduplication**: When using `read_file` inside an `execute_code` loop, if a file was already read in the conversation, the tool returns a dedup message instead of content. For reliable bulk reading in scripts, use `terminal(f"cat {shell_quote(path)}")`.
- **hermes_tools.read_file Output Format**: `hermes_tools.read_file` returns file content as `"LINE|CONTENT"` per line. When loading JSON written by a subagent or Router via python, strip these line numbers (e.g., `"\n".join(l.split("|", 1)[1] for l in raw.split("\n") if "|" in l)`) before calling `json.loads()`, or use `terminal(f"cat {shell_quote(path)}")` instead.
- **Semantic Noise**: `recon.py` can produce false positive collisions for generic terms (e.g., 'PIL', 'TABLE', 'ZERO'). Refer to the `NOISE_PATTERNS` filter inside `recon.py` for common noise terms and filtering strategies.
- **Path Quoting**: Vault paths containing spaces or apostrophes (e.g., "Alex's Second Brain") must be handled with `shell_quote` when passed to `terminal()`. Do **NOT** wrap the `{shell_quote(...)}` block in extra single or double quotes (e.g. `TARGET='{shell_quote(folder)}'`), as this results in nested matching errors in the shell.
- **bulk_writer.py Input Format**: The `--operations` flag expects a JSON **list** of operation objects (e.g. `[{...}, {...}]`), NOT a dictionary containing an `"operations"` key (e.g. `{"operations": [...]}`). Passing a dictionary will cause an `AttributeError` during iteration.
- **delegate_task Batch Limits**: The `max_concurrent_children` limit (typically 5) restricts the number of tasks per `delegate_task` call. For large distillation batches, partition tasks across multiple `delegate_task` calls to avoid tool errors.
- **Subagent Output Reliability**: Subagents may sometimes return the distilled JSON in their summary text rather than writing it to the requested file. Always verify the output file exists and contains valid JSON; if not, extract the JSON from the subagent's summary.
- **recon.py stderr behaviour**: In JSON mode, `recon.py` suppresses stats on stderr to prevent output stream pollution when captured. Do NOT redirect stderr to a file (like `2>/tmp/recon.stderr`) in JSON mode as it creates a confusing empty file.
- **prep_delegation.py Substitutions**: The `--substitute` flag only supports specific keys (e.g., `TARGET`). Attempting to pass unsupported keys like `HUB_NAME` will cause the script to crash with an unrecognized arguments error.
- **bulk_writer.py Requirements**: For `write` operations, the operations JSON **must** include both `heading` and `hub` keys. The Router is responsible for ensuring these are populated (e.g., deriving `heading` from the filename and using the provided `<HUB_NAME>`) before executing the bulk write.
- **Patch-to-Write Fallback**: Coercion between `patch` and `write` is handled automatically by the validator `validate_operations.py` (which coerces `patch` to `write` if the target file does not exist, and `write` to `patch` if it already exists). Do NOT use the native `patch` primitive tool to mutate note content directly; all mutational updates must be routed through `bulk_writer.py`.

