# Golden Pipeline Run — Obsidian Injector Reference

This file documents a canonical end-to-end execution of the Obsidian Injector pipeline, incorporating all validation and sanitization steps.

---

## Phase 1 — Reconnaissance

Run the mechanical recon script to discover concept names and vault collisions.

```bash
python3 ~/.hermes/skills/note-taking/obsidian-injector/scripts/recon.py \
    --inbox "/path/to/inbox" \
    --vault "/path/to/vault" \
    > /tmp/recon.json
```

### Sample Output (`/tmp/recon.json`)
```json
[
  {
    "file": "/path/to/inbox/Lezione 04.md",
    "collisions": [
      {
        "name": "Backpropagation",
        "total_hits": 5,
        "best_match": "title",
        "hits": [
          {
            "path": "/path/to/vault/Scrittura/Backpropagation.md",
            "count": 5,
            "in_title": true
          }
        ]
      }
    ],
    "new_concepts": [
      "Adam Optimizer"
    ]
  }
]
```

---

## Phase 2.0 — Payload Generation

Extract text excerpts of inbox files and colliding vault notes for the Distiller subagent.

```bash
python3 ~/.hermes/skills/note-taking/obsidian-injector/scripts/distiller_payload.py \
    --recon-report /tmp/recon.json \
    --max-concepts 10 \
    --out /tmp/distiller_payload.json
```

### Sample Output (`/tmp/distiller_payload_0.json`)
```json
{
  "schema_version": 1,
  "batches": [
    {
      "inbox_file": "/path/to/inbox/Lezione 04.md",
      "concepts": [
        {
          "name": "Backpropagation",
          "action_hint": "enrich",
          "inbox_excerpt": "La retropropagazione dell'errore (backpropagation)...",
          "vault_collision": {
            "path": "/path/to/vault/Scrittura/Backpropagation.md",
            "match_type": "title",
            "total_hits": 5,
            "excerpt": "# Backpropagation\nMetodo per calcolare..."
          }
        },
        {
          "name": "Adam Optimizer",
          "action_hint": "create",
          "inbox_excerpt": "L'ottimizzatore Adam combina i gradienti...",
          "vault_collision": null
        }
      ]
    }
  ]
}
```

---

## Phase 2.1 — Delegation Preparation & Run

Prepare the prompt and delegation JSON.

```bash
python3 ~/.hermes/skills/note-taking/obsidian-injector/scripts/prep_delegation.py \
    --protocol ~/.hermes/skills/note-taking/obsidian-injector/prompts/distiller_prompt.txt \
    --payload /tmp/distiller_payload_0.json \
    --substitute TARGET="/path/to/vault/Scrittura" \
    --out /tmp/delegation_args.json
```
This generates `/tmp/delegation_args.json` and a SHA-256 checksum in `/tmp/delegation_args.checksum`.

The Router reads the JSON and calls the subagent directly:
```python
tasks_data = json.loads(read_file("/tmp/delegation_args.json"))
delegate_task(tasks=tasks_data)
```

The subagent responds with its output in `/tmp/distiller_output_0.txt`.

---

## Phase 2.2 — Output Sanitization

Strip markdown fences and prose preambles from the subagent's raw output.

```bash
python3 ~/.hermes/skills/note-taking/obsidian-injector/scripts/parse_distiller_output.py \
    --in /tmp/distiller_output_0.txt \
    --out /tmp/distiller_output_0.json
```

---

## Phase 2.3 — Validation

Validate the parsed operation list against the payload to catch path deviations, hallucinated headings, or incorrect attribution basenames.

```bash
python3 ~/.hermes/skills/note-taking/obsidian-injector/scripts/validate_operations.py \
    --operations /tmp/distiller_output_0.json \
    --payload /tmp/distiller_payload_0.json \
    --target "/path/to/vault/Scrittura" \
    --out /tmp/operations.validated.json \
    --rejected-out /tmp/operations.rejected.json
```

### Validator Exit Code Meanings:
- `0`: Validated successfully. Rejections (if any) are $< 10\%$. Clean operations written to `/tmp/operations.validated.json`.
- `2`: Rejection rate is $\ge 10\%$. **ABORT** current run or re-run the failed subagent.

---

## Phase 3 — Execution

Write and patch files programmatically.

```bash
python3 ~/.hermes/skills/note-taking/obsidian-injector/scripts/bulk_writer.py \
    --operations /tmp/operations.validated.json
```

---

## Phase 4 — Linting

Verify frontmatter syntax and outgoing hub wikilinks for modified files.

```bash
python3 ~/.hermes/skills/note-taking/obsidian-injector/scripts/linter.py \
    --operations /tmp/operations.validated.json \
    --hub "Deep Learning"
```
