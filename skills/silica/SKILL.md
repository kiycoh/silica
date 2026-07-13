---
name: silica
description: Use the Silica vault as persistent memory over MCP — recall before answering, capture after learning. Trigger when the user asks what they know/decided/wrote about a topic ("what do I have on X", "cosa so su", "avevo deciso"), wants something saved for later ("save this to the vault", "salva nel vault", "ricordati che"), references their vault or past notes, or when the session produced a decision or insight worth keeping across sessions.
---

# Silica — vault memory over MCP

Silica is a deterministic knowledge-graph engine over an Obsidian vault. This
skill is the usage loop; the `silica` MCP server carries the mechanics
(tools named `silica_*` — if they are deferred, load them with ToolSearch).
If the tools are missing entirely, say so and give the user the install line:

```bash
claude mcp add silica -- uv run --project /path/to/silica-agent --with mcp silica mcp
```

## Recall — search before answering

Any question about accumulated knowledge starts with a search, not with your
own recollection:

- By meaning: `silica_semantic_search {query, k}` — "what do I have about X".
- Exact strings (error messages, names, quotes): `silica_search_context {query}`.
- Known title: `silica_search {query}`, then read it.

Never conclude "nothing in the vault" from a single miss — try at least one
semantic and one literal probe. If `silica_semantic_search` errors with
"No index available", fall back to `silica_search` / `silica_search_context`
(grep-based, always work) and tell the user `/embed` and `/cooccur` in the
Silica REPL build the missing indexes.

## Ground — read before citing

- `silica_read_note {name}` before quoting or acting on a hit; for long notes
  `silica_outline {name}` first, to target the right section.
- `silica_links {name}` / `silica_props {name}` give the note's neighborhood
  and frontmatter when you need context around a hit.

## Capture — write what deserves to outlive the session

What belongs: decisions and their why, non-obvious constraints, distilled
understanding, hard-won references. What does not: transcripts, code the repo
already holds, anything you could regenerate. Silica's quality gates reject
low-density notes — write like you'd want to re-read.

1. Search first (dedup). If a note on the concept exists, extend it:
   `silica_patch_note {name, heading, snippet, source_basename}` —
   `source_basename` is provenance (the file or conversation the snippet
   came from).
2. New concept → `silica_write_note {path, content}`. It refuses to overwrite
   by design: an "already exists" error means patch instead.
3. Note shape: one atomic concept per note; YAML frontmatter with tags;
   `[[wikilinks]]` to the related notes your searches surfaced; write in the
   vault's language (read one existing note if unsure).

## Know the boundary

The MCP surface is the fast path: search, read, single-note writes. Bulk work
— multi-file nucleation with quality gates, dedup sweeps, taxonomy, structural
reports — lives in the Silica REPL (`uv run silica`: `/nucleate`, `/report`,
`/curate`). When the task is bulk-shaped, say so and point there instead of
simulating the pipeline note by note.
