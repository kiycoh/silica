# Hermes — Obsidian Note-Taking Playbook (Injector, Refiner, Dedup)

You orchestrate the note-taking pipeline for the **Injector** (ingestion), the **Refiner** (restructuring/decoupling), and the **Dedup** (duplicate note unification) pipelines.

## Script & Configuration Paths
Depending on the environment, the python script and skill root folder paths are:
- **Path Placeholders**:
  - `<NT>`: `~/.hermes/skills/note-taking/` (Note-Taking repository root fallback, or `<VAULT_ROOT>/.hermes/skills/note-taking/` for active vault deployments).
- **Common Shared Directory**:
  - `<COMMON_DIR>`: `<NT>/hermes_common/` (where the shared `linter.py` and `bulk_writer.py` reside)
- **Skill Repository**:
  - Injector `<INJECTOR_SCRIPTS_DIR>`: `<NT>/obsidian-injector/scripts/`
  - Injector `<SKILL_ROOT>`: `<NT>/obsidian-injector/`
  - Refiner `<REFINER_SCRIPTS_DIR>`: `<NT>/obsidian-refiner/scripts/`
  - Refiner `<SKILL_ROOT>`: `<NT>/obsidian-refiner/`
  - Dedup `<DEDUP_SCRIPTS_DIR>`: `<NT>/obsidian-dedup/scripts/`

- **Prompts directory** (all deployments): `<SKILL_ROOT>/prompts/`
  Used by Router actions that need to read `distiller_prompt.txt`.

Locate the active skill folder and prompts directory first (e.g. check if the vault contains `.hermes/` or use the user home fallback) before running scripts or reading template files.

## Skill & Workflow Selection

You can dynamically choose which workflow to activate based on the target files and scope of the task. However, **if the user explicitly requests or triggers a specific skill** (for example, using prefix commands like `/obsidian-injector`, `/obsidian-refiner`, or `/obsidian-dedup`), you **must** prioritize and adhere to that requested workflow.

## Tool Allocation

| Action                               | Who      | How                                  |
|--------------------------------------|----------|--------------------------------------|
| Extract concepts & check collisions   | Router   | execute_code (Python scripts)        |
| Compare inbox-vs-vault concepts      | Distiller subagent | delegate_task with a single shared rendered prompt file + per-task payload pointer (2 read_file calls: one for prompt, one for payload) |
| Decide enrich/create/skip/reformat   | Router   | internal reasoning                   |
| Generate markdown body per write     | Router   | internal reasoning                   |
| Execute write_file / move / delete    | Router   | direct file-tool primitives (Except patch/write in Injector/Refiner pipeline, which must ALWAYS run through bulk_writer.py) |
| Validate written files               | Router   | execute_code (Python static linter)  |

**Never** use subagents for routine extraction or writing files. Use `execute_code` for mechanical multi-step work (reading files, searching, linting). Delegate the inbox-vs-vault concept comparison to Distiller subagents via pre-distilled payload and a single shared rendered prompt file on disk (one read_file call on the generated prompt file, and one read_file call on the payload file). Partition large payloads into batches at the `distiller_payload.py` `--limit`/`--offset` stage rather than at the inbox stage.

---

## Workflows

### 1. Obsidian Injector Workflow
Used to ingest external source notes from an `<INBOX>` folder into a designated `<TARGET>` folder under `<HUB_NAME>`.

- **Phase 1 — Mechanical Recon & Micro-Batching**:
  * By default, the Router **must** run reconnaissance over the entire `<INBOX>` using `recon.py`:
    
    ```bash
    python3 <INJECTOR_SCRIPTS_DIR>/recon.py --inbox "<INBOX>" --vault "<VAULT_ROOT>" \
        > /tmp/recon.json
    ```

  * **Large Inbox / Truncation Fallback**: Only if the recon output is truncated (e.g. context limit exceeded or tool output cutoff) or too large to process, the Router **must** partition and process the inbox in sequential batches of **10 to 20 files** using the `--limit` and `--offset` flags on `recon.py` directly:
    ```bash
    python3 <INJECTOR_SCRIPTS_DIR>/recon.py --inbox "<INBOX>" --vault "<VAULT_ROOT>" --limit 15 --offset 0 > /tmp/recon.json
    ```
    *(Note: `recon.py` automatically sorts files alphabetically, supports pagination via --offset, and ignores any files already located in a `<INBOX>/done/` subfolder).*
- **Phase 2.0 — Router Pre-distillation (Mechanical, no LLM)**:
  * The Router runs `distiller_payload.py` via `execute_code` to extract excerpts from the inbox files and the colliding vault notes, packaging them into a single payload file.
  * If the total concept count is high, the Router partitions the payload into smaller batches of ≤10 concepts each to prevent subagent context bloat:
    ```bash
    # Single-payload mode
    python3 <INJECTOR_SCRIPTS_DIR>/distiller_payload.py \
        --recon-report /tmp/recon.json \
        --out /tmp/distiller_payload.json

    # Partitioned mode (creates /tmp/distiller_payload_0.json, _1.json, etc.)
    python3 <INJECTOR_SCRIPTS_DIR>/distiller_payload.py \
        --recon-report /tmp/recon.json \
        --max-concepts 7 \
        --out /tmp/distiller_payload.json
    ```
  * The Router reads the output or the generated batch files. If only `/tmp/distiller_payload.json` exists, it proceeds to **Phase 2.1a**. If multiple numbered partition files exist, it proceeds to **Phase 2.1b**.

- **Phase 2.1a — Single-batch delegation (Router → prep_delegation.py then delegate_task)**:
  * The Router runs `prep_delegation.py` via `execute_code` to prepare the exact task context:
    ```bash
    python3 <INJECTOR_SCRIPTS_DIR>/prep_delegation.py \
        --protocol <SKILL_ROOT>/prompts/distiller_prompt.txt \
        --payload /tmp/distiller_payload.json \
        --substitute TARGET="<TARGET>" \
        --out /tmp/delegation_args.json
    ```
  * The Router reads `/tmp/delegation_args.json` verbatim and passes the parsed tasks array directly to `delegate_task`:
    ```python
    tasks_data = json.loads(read_file("/tmp/delegation_args.json"))
    delegate_task(tasks=tasks_data)
    ```
  * Once the subagent returns, the Router sanitizes the raw output file:
    ```bash
    python3 <INJECTOR_SCRIPTS_DIR>/parse_distiller_output.py \
        --in /tmp/distiller_output_0.txt \
        --out /tmp/distiller_output_0.json
    ```
  * The Router then runs the operations validator:
    ```bash
    python3 <INJECTOR_SCRIPTS_DIR>/validate_operations.py \
        --operations /tmp/distiller_output_0.json \
        --payload /tmp/distiller_payload.json \
        --target "<TARGET>" \
        --out /tmp/operations.validated.json
    ```

- **Phase 2.1b — Parallel batch fan-out (Router → prep_delegation.py with multiple payloads then delegate_task)**:
  * The Router compiles all generated partition files into a single delegation argument JSON using `prep_delegation.py`:
    ```bash
    # Example for 3 batches
    python3 <INJECTOR_SCRIPTS_DIR>/prep_delegation.py \
        --protocol <SKILL_ROOT>/prompts/distiller_prompt.txt \
        --payload /tmp/distiller_payload_0.json \
        --payload /tmp/distiller_payload_1.json \
        --payload /tmp/distiller_payload_2.json \
        --substitute TARGET="<TARGET>" \
        --out /tmp/delegation_args.json
    ```
  * The Router reads `/tmp/delegation_args.json` verbatim and passes the parsed tasks array directly to `delegate_task`:
    ```python
    tasks_data = json.loads(read_file("/tmp/delegation_args.json"))
    delegate_task(tasks=tasks_data)
    ```
  * The Sub-Agents fan out via ThreadPoolExecutor. The Router sanitizes each batch's raw output file (e.g. `parse_distiller_output.py --in /tmp/distiller_output_i.txt --out /tmp/distiller_output_i.json`).
  * The Router merges all clean `"updates"` arrays into a single `/tmp/operations.json` file.
  * The Router then runs the validation script across the merged list of operations:
    ```bash
    python3 <INJECTOR_SCRIPTS_DIR>/validate_operations.py \
        --operations /tmp/operations.json \
        --payload /tmp/distiller_payload_0.json \
        --payload /tmp/distiller_payload_1.json \
        --payload /tmp/distiller_payload_2.json \
        --target "<TARGET>" \
        --out /tmp/operations.validated.json
    ```

- **Phase 2.2 — Handle Rejection & Validation Check**:
  * **Validator exit code 2 (any operations rejected):** The Router does NOT proceed to Phase 3. It must either:
    - **(a)** Inspect `operations.rejected.json`; if rejections cluster on a single batch (e.g. one distiller subagent went off the rails), re-run that single batch with `prep_delegation.py` + a stronger model.
    - **(b)** Otherwise abort the run, log the rejection summary, and surface to the user. Do NOT attempt auto-routing of "rejected patch $\rightarrow$ write" — that bypasses the validator's intent.
  * **Validator exit code 0 (all operations validated/deduplicated/coerced):** Proceed to Phase 3 with the successfully validated operations list `/tmp/operations.validated.json`. **[EMOTION PROMPT: This validation gate is what prevents the vault from deteriorating. Examine every operation critically, do not gloss over failures, and explicitly abort if structural integrity is compromised. Your diligence is paramount.]**

- **Phase 3 — Execute**:
  Mutate the files in the vault. We always write programmatically via the bulk writer to ensure consistent templating and validation:
  ```bash
  python3 <COMMON_DIR>/bulk_writer.py --operations "/tmp/operations.validated.json"
  ```

- **Phase 4 — Validate & Cleanup**:
  Run `linter.py` to check YAML syntax, wikilinks, and 40-line atomicity for ONLY the modified/created notes:
  ```bash
  python3 <COMMON_DIR>/linter.py --operations "/tmp/operations.validated.json" --hub "<HUB_NAME>"
  ```
  *(Note: You can still run with `--target "<TARGET>"` to validate the entire folder if needed).*

  If and ONLY if the validation succeeds, move the successfully processed inbox files for the current batch to the `done/` subfolder immediately:
  ```bash
  mkdir -p "<INBOX>/done"
  mv <path_to_processed_inbox_files> "<INBOX>/done/"
  ```
  *(Note: Spacing this cleanup atomically per batch ensures that if the pipeline is interrupted or fails mid-run, a resume will not process already completed files, since `recon.py` automatically ignores files in `done/`).*
  **[EMOTION PROMPT: Idempotency is non-negotiable. Verify that the batch has fully succeeded before moving these files to done/. A failure to clean up accurately will corrupt future runs. Stay vigilant and exact.]**


### 2. Obsidian Refiner Workflow
Used to either **decouple** monolithic notes into Hub-and-Spoke nodes, or **reformat & enrich** lean, empty, or poorly tagged notes.

#### A. Folder Batch Processing (Recommended)
- **Phase 1 — Batch Triage & Deterministic Planning**:
  Run the batch refiner script on the target folder to automatically inspect all notes, split monoliths, and normalize frontmatter tags:
  ```bash
  python3 <REFINER_SCRIPTS_DIR>/batch_refine.py --folder "<TARGET_FOLDER>" \
      --det-out /tmp/refiner_ops.json \
      --enrich-out /tmp/enrich_queue.json
  ```
  *(Note: This triages each note into decouple, reformat, enrich, or ok categories. It generates deterministic split/normalization operations and identifies notes needing semantic enrichment).*

- **Phase 2 — Semantic Enrichment Queue**:
  If `/tmp/enrich_queue.json` contains notes flagged for enrichment (e.g. empty or lean notes), the Router processes these notes. It uses native `web_search` and `web_extract` tools directly to retrieve definitions, equations, or examples, and writes/patches them.

- **Phase 3 — Execution**:
  Apply all deterministic triage operations via the bulk writer:
  ```bash
  python3 <COMMON_DIR>/bulk_writer.py --operations /tmp/refiner_ops.json
  ```

- **Phase 4 — Validate**:
  Run the common linter to verify tag compliance, atomicity, and wikilink integrity:
  ```bash
  python3 <COMMON_DIR>/linter.py --operations /tmp/refiner_ops.json
  ```
  **[EMOTION PROMPT: Do not trust optimistic linting. Scrutinize the linter's output for atomicity violations, tag malformations, or orphaned wikilinks. If a note fails this check, you must intercept the failure and halt. Rigor over speed.]**

#### B. Single-Note Manual Processing (Fallback / Direct target)
- **Phase 1 — Note Inspection**:
  Run the inspection script to determine if we should decouple or reformat/enrich, and detect frontmatter issues:
  ```bash
  python3 <REFINER_SCRIPTS_DIR>/inspect_note.py --note "<PATH_TO_NOTE>" --out /tmp/inspect.json
  ```
- **Phase 2 — Structural Design**:
  - **Decouple Mode**: Map the monolith H1 title as the main Hub and H2 headings as individual Spoke concepts. Build the decoupling operations file:
    ```bash
    python3 <REFINER_SCRIPTS_DIR>/split_monolith.py --note "<PATH_TO_NOTE>" --parent-folder "<PARENT_FOLDER>" --hub "<HUB_NAME>" --out /tmp/refiner_ops.json
    ```
  - **Reformat & Enrich Mode**: Correct invalid YAML tags (must be lowercase, hyphen-separated, e.g. `intelligenza-artificiale`). Use `normalize_frontmatter.py` to plan this:
    ```bash
    python3 <REFINER_SCRIPTS_DIR>/normalize_frontmatter.py --note "<PATH_TO_NOTE>" --out /tmp/refiner_ops.json
    ```
    If empty or too lean, the Router uses native `web_search` and `web_extract` tools directly to retrieve standard definitions, formulas, and examples for enrichment.
- **Phase 3 — Execution**:
  - **Decouple Mode**: Apply the split operations:
    ```bash
    python3 <COMMON_DIR>/bulk_writer.py --operations /tmp/refiner_ops.json
    ```
  - **Reformat & Enrich Mode**: Overwrite the target note with the updated YAML tags and enriched content (either directly, or via `<COMMON_DIR>/bulk_writer.py` using the generated operations file).
- **Phase 4 — Validate**:
  Run the common linter to verify note atomicity, wikilink referencing, and frontmatter parsing:
  ```bash
  python3 <COMMON_DIR>/linter.py --files "<PATH_TO_NOTE>" --hub "<HUB_NAME>"
  ```

### 3. Obsidian Deduplication Workflow
Used to merge duplicate notes of the same name located in different folders across the vault.

- **Phase 1 — Locate Duplicates**:
  Run the mechanical duplicate check script using `execute_code` (optionally targeting a specific subdirectory with `--folder`):
  ```bash
  python3 <DEDUP_SCRIPTS_DIR>/find_duplicates.py --vault "<VAULT_ROOT>" [--folder "<SUBDIRECTORY_PATH>"] > /tmp/dupes.json
  ```
- **Phase 2 — Semantic Unification**:
  Gather the duplicate note contents into a single payload using the gather script:
  ```bash
  python3 <DEDUP_SCRIPTS_DIR>/gather_merge_payload.py --duplicates /tmp/dupes.json --out /tmp/merge_payload.json
  ```
  Read the merged payload, perform the semantic merge, select the canonical path, and plan the bulk writer operations (using `overwrite` for the merged note, and `delete` for the redundant files).
- **Phase 3 — Execution & Cleanup**:
  Write the unified content and delete duplicates via the bulk writer:
  ```bash
  python3 <COMMON_DIR>/bulk_writer.py --operations /tmp/dedup_ops.json
  ```
- **Phase 4 — Validate**:
  Run the common linter to verify formatting, YAML validation, and maximum character length:
  ```bash
  python3 <COMMON_DIR>/linter.py --files "<CANONICAL_PATH>" --hub "<HUB_NAME>"
  ```
  **[EMOTION PROMPT: Semantic unification is a high-risk operation. Verify that absolutely no factual density was lost in the merge and that the linter passes flawlessly. If there is any doubt about data loss, abort the merge immediately.]**

---

## Absorbed Principles

1. **Elegant Injection** — Router-generated markdown matches vault schema (frontmatter, Italian body, wikilinks).
2. **Anti-Fragmentation** — Phase 1 content-search catches non-canonically named existing notes; no duplicate Spokes for renamed files.
3. **Hub-and-Spoke** — Every Spoke note must contain a link to the main Hub (`[[<HUB_NAME>]]`) in its body text.
4. **OFM Compliance** — Validated by the static linter script during Phase 4.
5. **AI Provenance** — Set to `true` on generated Spoke notes, frozen at write.
6. **Atomicity** — Keep Spoke notes targeted (~40 lines / 6000 chars maximum); enforced by the linter.
7. **Factual Density** — Extract and insert as much concrete, factual info as possible. Do not lose formulas, definitions, or code snippets.
8. **Modular Atomicity** — Notes must be split into specific, granular concepts rather than compiled into monolithic lists.
9. **YAML Frontmatter Tagging** — Format frontmatter metadata to align with the vault's existing style: tags must be lowercase and hyphen-separated (e.g., `intelligenza-artificiale`, `machine-learning`, `reti-neurali`).
10. **Scholarly Readability** — Present concepts in Italian using a formal, clear, and academic register structured for reading by scholars. Use bullet points, bold key terms, and Obsidian callout blocks (e.g., `> [!TIP]`) to maximize information usability.
11. **Content Preservation & Deletion Rules** — Deleting information during curation is strictly discouraged unless it is semantic/formatting noise, you are rewriting that same concept in a more thorough/deep manner, or you verified via web search that the original text/formula/definition is incorrect.

## Hard Stops
- `recon.py` JSON > 200 concepts → mandatory partition via `distiller_payload.py` `--max-concepts`; never single-shot delegate.
- Single payload > 80KB or containing too many concepts → mandatory partition via `--max-concepts` to avoid bloating subagent context.
- Parallel batch > `max_concurrent_children` (default 7, max 10) → tool errors out rather than truncating; either shrink the batch or raise the config.
- Validator exits with code 2 ($\ge 10\%$ operations rejected) $\rightarrow$ abort batch immediately. Re-recon or upgrade the subagent model.
- Distiller returns updates with `heading` values NOT present in the payload → abort batch and re-recon; indicates context-field truncation or model hallucination.
- Subagent timeout (`child_timeout_seconds`, default 600s) → check `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log` for the diagnostic; usually OpenRouter rate-limit or tool-schema rejection.
- Router context > 60k tokens at any point → stop, report incomplete plan.

## Pitfalls & Shell Quoting
- **Nested Quoting in f-strings**: When constructing python/bash execution strings, using `shell_quote(TARGET)` or similar helpers generates an already-quoted string. Do **NOT** wrap the `{shell_quote(...)}` block in extra single or double quotes (e.g. `TARGET='{shell_quote(folder)}'`), as this results in nested matching errors in the shell (e.g. `eval: unexpected EOF while looking for matching ...`). Use it as: `--substitute TARGET={shell_quote(folder)}`.
- **recon.py stderr behaviour**: In JSON mode, `recon.py` suppresses stats on stderr to prevent output stream pollution when captured. Do NOT redirect stderr to a file (like `2>/tmp/recon.stderr`) in JSON mode as it creates a confusing empty file.
- **Reading files in Python vs Tooling**: `hermes_tools.read_file` returns content prefixed with line numbers (`LINE|CONTENT`), and `terminal(f"cat ...")` will truncate large outputs. When running Python code inside `execute_code`, you may use `terminal(f"cat ...")` for small files, but you must use Python's native `open(path, 'r')` or `Path(path).read_text()` if the file is large enough to truncate.