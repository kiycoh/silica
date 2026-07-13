# Silica

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-22d3ee.svg)](https://www.python.org/)
[![Obsidian Compatible](https://img.shields.io/badge/Obsidian-Compatible-38afef.svg)](https://obsidian.md/)
[![License: AGPL 3.0](https://img.shields.io/badge/License-AGPL_v3-4d8af0.svg)](https://opensource.org/licenses/AGPL-3.0)
[![Powered by UV](https://img.shields.io/badge/package--manager-uv-6366f1.svg)](https://github.com/astral-sh/uv)

<p align="center">
  <img src="assets/sili_no_bg.png" alt="Silica Mascot Sili" width="200" />
</p>

<h3 align="center">Silica istantly knows what you're working on, helping you edit and organize your content without corrupting it.</h3>

<p align="center">
Point it at a folder of notes or a codebase. Silica builds a live model of what's inside, so the LLM answers from your actual material, and every edit it makes is checked and rolled back if it broke something. <b>Local-first</b>, Obsidian-compatible.<br/>
</p>

<p align="center">
  <video src="https://github.com/kiycoh/silica-agent/raw/main/assets/presentation.mp4" width="100%" controls></video>
</p>

> **License:** AGPL-3.0-or-later, *strong copyleft*. Copying any part obliges your whole project to AGPL, including network-only use (§13). [Details below](#license).

---

## Table of Contents
- [What you get](#what-you-get)
- [Quick Start](#quick-start)
- [Ways to use Silica](#ways-to-use-silica)
- [What you can do](#what-you-can-do)
- [See your vault](#see-your-vault)
- [How Silica actually works](#how-silica-actually-works)
- [Silica for codebases](#silica-for-codebases)
- [A promise](#a-promise)
- [Command reference](#command-reference)
- [Configuration](#configuration)
- [Status](#status)
- [References](#references)
- [Contributing](#contributing) · [License](#license)

---

## What you get

A pile of pdf, or a repository, becomes something you can *ask* and *reshape* without babysitting it:

- **Answers from what you actually have.** Before you describe anything, Silica sees the *shape* of your vault: hubs, clusters, and notes nearest your questions so you get grounded answers.
- **Edits don't rot in your vault.** When Silica nucleates, merges, or refactors, every write can be reverted. The blast radius of a bad edit is one `/undo`.
- **A graph you can navigate.** Export the whole knowledge graph to an interactive page, or grow a radial mind-map out from any single note.
- **Works offline.** Local models (LM Studio, Ollama) are supported; if no embedder is configured, relatedness degrades to a deterministic local graph instead of failing. Still, embeddings are highly recommended for better quality.

---

## Quick Start

### Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/kiycoh/silica-agent.git
cd silica-agent
uv pip install -e .
```

Optional features are installed as extras, alone or combined (`'.[gui,mcp]'`):

```bash
uv pip install -e '.[gui]'      # web GUI: silica --gui
uv pip install -e '.[mcp]'      # MCP server: silica mcp (Silica as agent memory)
uv pip install -e '.[connect]'  # Obsidian plugin bridge: silica connect
uv pip install -e '.[pdf]'      # PDF nucleation
uv pip install -e '.[dev]'      # tests and linters
```

### Setup and Execution

Run the interactive wizard to set up your `.env` (vault, backend, chat provider, embeddings):

```bash
uv run silica init
```

Re-check the environment at any time:

```bash
uv run silica doctor
```

Start the interactive REPL:

```bash
uv run silica
```

A good first move on an existing vault is a read-only structural audit. It never writes, and it shows you the hubs, bridges, and orphans before you touch anything:

```
/report
```

---

## Ways to use Silica

The same vault model serves four different drivers. What changes is who holds the write key, and whether they read or write:

1. **GUI (`silica --gui`)**
   A chat-first web interface (default `http://localhost:8765`). Query and curate the vault from the browser, watch answers stream in, and open the graph. The friendliest way in, and the best first impression.

    ![Web UI Screenshot](assets/web_gui_screenshot.png)

2. **CLI / TUI (`silica`)**
   The interactive terminal REPL. Every command in the [reference](#command-reference) lives here: nucleate, audit, search, refactor, visualize. Fastest for real work once you know the verbs.

    ![CLI Screenshot](assets/cli_screenshot.png)

3. **Obsidian plugin (`silica connect`)**
   A live bridge into the Obsidian desktop app, so Silica reads and writes the vault you already have open, with its rollback and cache backing every change. *Feature-complete, pending end-to-end hardening.*

    ![Obsidian Plugin Screenshot](assets/obsidian_plugin_screenshot.png)

4. **MCP server (`silica mcp`)**
   Silica serves the vault over stdio to any MCP client. An assistant recalls from your real notes before it answers, grounding on your real decisions instead of guessing. For Claude Code, the repo is also a plugin:
   ```bash
   claude plugin marketplace add /path/to/silica-agent
   claude plugin install silica@silica
   ```
   
    ![Claude Code Screenshot](assets/mcp_screenshot.png)
---

## What you can do

- **Clear inbox files without losing anything.** Drop raw clippings and drafts in a folder; `/nucleate Inbox/*.md` distills each into an atomic note, checks it against what you already have so you don't get a fifth copy of the same idea, and files it. Hand it 20 files at once; it never gets confused by the pile.
- **Ask your notes, not your memory.** `/explain "<concept>"`, `/compare "A" "B"`, `/summarize <folder>`, `/quiz <note>`: read-only answers grounded in the vault, with contradictions surfaced instead of smoothed over.
- **Reorganize by intent.** `/organize "group by project"` classifies and moves notes to a taxonomy. `/curate` plans the autolink, dedup, and cleanup work; add `--apply` to run it.
- **Refactor safely.** Merges and splits redirect every incoming link automatically, so you never end up with a broken reference or an orphan. Changed your mind? `/undo` a note or `/revert` a whole run.
- **Research into the vault.** `/web-search "<topic>"` pulls cited findings into the inbox; `/nucleate` them when you're ready. Nothing from the web lands in the vault without your say-so.

---

## See your vault

Silica turns the invisible structure of your notes into something on screen.

**Knowledge graph (`/graph out.html`)** exports an interactive page: notes as nodes, links as edges, communities colored and named automatically so the clusters read at a glance. Open it in any browser, no server needed.

**Mind-map (`/map <note>`)** grows a radial map rooted on one note and writes it as `maps/<stem>.canvas` (opens natively in Obsidian) plus an SVG. Layout is by community, laid out so nodes never overlap, so the picture is legible from the first render.

Both run locally and work embedder-free: even with no embedding model, the deterministic co-occurrence graph still gives you clusters and relatedness. Can be viewed in GUI or saved as file.

---

## How Silica actually works

### Guardrails, not blind trust

You already let deterministic tools rewrite and reject your work every day. You don't extend them trust; you trust the guardrail. Silica wraps an LLM's edits to your vault in the same kind of guardrail:

| You already let a deterministic tool… | to guard against… | Silica does the same for a vault by… |
| :--- | :--- | :--- |
| a **compiler** reject source that won't build | syntax and type errors | an FSM refusing to commit a note that fails its structural checks |
| a **test suite** block a merge that breaks behavior | regressions | a post-write **verify gate** that reverts any edit which breaks vault coherence |
| **git** roll back a bad commit | losing history | `/undo` and `/revert` rolling back an injection, per-note or per-run |
| a **formatter** rewrite your code without asking | drift and inconsistency | graph-safe refactors that redirect links so a merge or split never orphans a note |

<p align="center">
  <img src="assets/pipeline.svg" alt="Silica vault pipeline mapped onto a software engineering pipeline" width="880" />
</p>

### Design contracts

Silica is not a free-form agent. Every vault mutation passes through a finite-state machine that enforces these contracts:

- **Single entry point:** all nucleation flows through the Injector FSM. There is no side channel that writes to the vault.
- **Verify-or-revert:** every write is re-read and checked afterward; a mismatch (`VerifyMismatchError`) rolls the write back.
- **Graph-safe moves:** renames, merges, and splits redirect incoming links atomically. No operation leaves a broken reference or an orphan.
- **Zero-trust ingress:** external content (e.g. web search) can only land in `Inbox/`. Nothing reaches the vault without explicit human staging and FSM review.
- **Layered rollback:** `/undo` (per note), `/revert` (per run), and optional `SILICA_GIT_COMMIT=auto` stack as independent safety nets.

<p align="center">
  <img src="assets/architecture.svg" alt="Silica Architectural Schematic" width="880" />
</p>

---

## Silica for codebases

Point `SILICA_VAULT` at a repository instead of a note folder, and Silica keeps a living, human-readable map of the code under `docs/silica/`, honest against git.

- `/nucleate <file>` extracts a shallow AST skeleton with tree-sitter (signatures, structure, imports) and turns it into a markdown note that documents its source, stamped with the commit it was verified against.
- `/wiki` grows that into a behavioral wiki: an `ARCHITECTURE.md` plus one note per subsystem.
- `/stale` flags notes whose source *changed in shape* since you last documented it (a signature or control-flow change, not just a reformat); `/impact` maps a set of changed files to the notes they affect. You re-document what actually moved, not the whole tree.

Two readers, one artifact: a human reads it as an always-current map, and a coding agent reads it over the [MCP server](#ways-to-use-silica) to ground its work in the real structure instead of re-deriving it every session.

---

## A promise

Most "chat with your notes" tools hand an open-ended agent a write key and hope the edits are good. The failure is quiet: a merge orphans a note, a rewrite breaks a wikilink.

There is no free-running loop holding a key to your vault. Every change goes one way: the agent proposes, a state machine checks it, you confirm, and the write is re-read and rolled back if it broke coherence. ***You can still use git alongside Silica*** as the byte-level backstop; Silica is the coherence layer on top.

> **⚠️ Note:** It is always highly recommended to create a backup of your vault before allowing Silica to manipulate your files.

That's the whole idea. You don't have to believe anything about the model, only that the guardrail runs on every write. It does.

> Enforced today on the normal write path. Not yet crash-verified (a harness that kills the process mid-write to prove invariants survive failure is [in progress](#status)): trust it as enforced control flow, not a proof under adversarial faults.

---

## Command reference

**Ask & audit (read-only):**

| Command | What it does |
| :--- | :--- |
| `/report [folder]` | Structural audit: hubs, bridges, orphans |
| `/explain "<concept>" [--level]` | Explain a concept, grounded in the vault |
| `/summarize <note\|folder>` | Digest of one or more notes |
| `/compare "A" "B"` | Comparison table, surfaces contradictions |
| `/quiz <note> [--n=10]` | Active-recall quiz from your notes |
| `/relate <note>` · `/path A B` | How notes relate · shortest reading path |
| `/find <query>` | Semantic search |

**Bring in & reshape:**

| Command | What it does |
| :--- | :--- |
| `/nucleate <file...> [--target=DIR]` | Notes via the gate; code as skeletons |
| `/organize "<intent>" [--apply]` | Classify and move notes to a taxonomy |
| `/curate [--apply]` · `/dedup` · `/refine` | Plan / run autolink, dedup, enrichment |
| `/web-search "<topic>"` | Cited web findings into the inbox |
| `/convert <file>` | Transcode a PDF into a markdown draft |

**Visualize:** `/graph [out.html]` · `/map <note>`

**Codebase:** `/wiki` · `/stale` · `/impact [<range>]`

**Undo & inspect:** `/undo [note]` · `/revert [run]` · `/status` · `/review`

**System:** `/help` · `/model` · `/vault [path]` · `/settings [<key> <value>]` · `/clear` · `/exit`

---

## Configuration

`silica init` writes the essentials; the full list with defaults is in [`.env.example`](.env.example).

| Variable | Description |
| :--- | :--- |
| `SILICA_MODEL` | Chat model (litellm format, e.g. `openrouter/anthropic/claude-sonnet-4`) |
| `SILICA_PROVIDER` | `lmstudio` or `openrouter` |
| `SILICA_VAULT` | Vault path. Obsidian vault used verbatim; any other path is repo mode → `docs/silica/` |
| `SILICA_EMBEDDING_MODEL` | Embedding model for semantic tasks (default `qwen3-embedding-4b`) |
| `SILICA_BACKEND` | `fs` (default, headless) or `cli` (live Obsidian via CDP) |
| `SILICA_GIT_COMMIT` | Git safety net for writes (`off`, `auto`) |
| `SILICA_TAVILY_API_KEY` | Enables `/web-search` |
| `SILICA_WORKER_MODEL` | Sub-agent worker model (for dedup/refinement operations) |

---

## Status

- **Available now:** note nucleation, structural audit (`/report`), semantic and embedder-free search, graph-safe refactor / dedup / merge, graph and mind-map export, codebase skeletons with git-backed `/stale` and `/impact`, the code `/wiki`, layered `/undo` and `/revert`, git safety net, the MCP server and Claude Code plugin.
- **In progress:** richer codebase coverage across more languages, PDF/DOCX/TXT nucleation, the live Obsidian bridge (`silica connect`), and the crash harness backing the guardrail.
- **Planned:** image nucleation, MCP packaging for non-Claude agents.

What ships today is enforced; what's in-progress or planned is not yet present.

---

## References

*   **[From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution](https://arxiv.org/abs/2604.11378)** (arXiv:2604.11378, 2026)
*   **[Goal-Autopilot: A Verifiable Anti-Fabrication Firewall for Unattended Long-Horizon Agents](https://arxiv.org/abs/2606.11688)** (arXiv:2606.11688, 2026)
*   **[Is Your Agent Playing Dead? Deployed LLM Agents Exhibit Constraint-Evasive Fabrication and Thanatosis](https://arxiv.org/abs/2606.14831)** (arXiv:2606.14831, 2026)
*   **[Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs](https://arxiv.org/abs/2601.08773)** (arXiv:2601.08773, 2026)
*   **[Predicting new research directions in materials science using large language models and concept graphs](https://doi.org/10.1038/s42256-026-01206-y)** (*Nature Machine Intelligence*, 2026)

Silica's embedder-free near-duplicate detection (`/dedup`) is inspired by and ports the well-thought-out MinHash design from [Graphify](https://github.com/safishamsi/graphify).

---

## Contributing

Issues and pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and conventions (English-only, conventional commits). By contributing you license your work under AGPL-3.0-or-later. For security issues follow [SECURITY.md](SECURITY.md); do not open a public issue.

---

## License

**GNU Affero General Public License v3.0** (AGPL-3.0-or-later). Copyright (C) 2026 Alessandro Carosia.

Strong copyleft: incorporate any portion of Silica and that work becomes a derivative that must itself be AGPL-3.0, with complete corresponding source offered to everyone who uses it. **§13** extends this to network use: running a modified version as a hosted service obliges you to provide source to your users. There is no permissive fallback. See [LICENSE](LICENSE) for the full text.
