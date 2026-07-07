# Silica Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-22d3ee.svg)](https://www.python.org/)
[![Obsidian Compatible](https://img.shields.io/badge/Obsidian-Compatible-38afef.svg)](https://obsidian.md/)
[![License: AGPL 3.0](https://img.shields.io/badge/License-AGPL_v3-4d8af0.svg)](https://opensource.org/licenses/AGPL-3.0)
[![Powered by UV](https://img.shields.io/badge/package--manager-uv-6366f1.svg)](https://github.com/astral-sh/uv)

<p align="center">
  <img src="docs/assets/sili_no_bg.png" alt="Silica Mascot Sili" width="250" />
</p>

> **Silica** is a CLI-based agent designed for automated curation, organization and **safe** knowledge management. Local-first and open-source. Supports Obsidian.

---

## Table of Contents
- [Overview](#overview)
- [Use Cases](#use-cases)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Setup](#setup)
  - [Execution](#execution)
  - [REPL Commands](#repl-commands)
- [Configuration](#configuration)
- [Quirks & Features](#quirks--features)
- [References](#references)
- [License](#license)

---

## Overview

Silica is a CLI-based deterministic agent orchestrator that can manage Obsidian vaults, codebases (wip), images (wip), .pdf/.docx/.txt documents (wip) by having context of their relations (cooccurrence, hyperlinks, graph).

- Silica is ***local-first*** (LM Studio, Ollama), OpenRouter is also supported.
- **Silica prevents the risk of vault corruption and structural cluttering** by using safety-hardened tools and rollbacks.
- Silica maintains and updates **a vault index separate from your files.**
- Silica is not a free-form agent orchestrator.

---

## Use Cases

1. **Automated Inbox Ingestion** — Reads raw clippings and drafts from an inbox directory, distills them into atomic markdown concepts, resolves duplicate matches against the existing vault, and writes them safely.
2. **Conversational Vault Querying** — Allows users to query their notes, map paths across the graph, and generate outlines or synthesis documents using semantic search and graph-traversal tools in the REPL.
3. **Graph-Safe Note Refactoring** — Handles complex merges and splits of concept notes. Redirects incoming links automatically to prevent broken references or orphaned files.

---

## Quick Start

### Installation
Clone the repository and install it in editable mode:

```bash
git clone https://github.com/kiycoh/silica-agent.git
cd silica-agent
uv pip install -e .
```

### Setup

Run the interactive wizard — it writes your `.env` (vault, backend, chat provider, embeddings) and finishes with a diagnostic report:

```bash
uv run silica init
```

Re-check the environment at any time:

```bash
uv run silica doctor
```

### Execution

Start the interactive REPL:

```bash
uv run silica
```

Run the ingestion pipeline from inside the REPL:

```
/ingest Inbox/note.md --target=Concepts/AI
```

### REPL Commands

**Workflow** — agent-directed:

| Command | Usage | Description |
| :--- | :--- | :--- |
| `/report` | `[folder] [--top-k=N] [--embeddings]` | Structural audit of the vault (hubs, bridges, orphans). Pauses for confirmation. |
| `/ingest` | `<file...> [--target=DIR] [--hub=H]` | Bring files in: notes via Injector FSM, code as skeleton stubs |
| `/organize` | `"<intent>" [--scope=FOLDER] [--file=taxonomy.yaml] [--merge] [--move-uncategorized] [--apply]` | Classify and reorganize vault notes according to a taxonomy |

**Direct** — immediate, no LLM round-trip:

| Command | Usage | Description |
| :--- | :--- | :--- |
| `/status` | `[run_id]` | Progress digest of the last run |
| `/convert` | `<file...> [--target=DIR]` | Transcode a non-`.md` file (PDF) into a markdown note in the inbox |
| `/web-search` | `"<concept>" [--max-searches=N]` | Research a concept on the web → cited findings note in the Inbox (then `/ingest`) |
| `/embed` | `[folder] [--force]` | Build/update the embedding index |
| `/cooccur` | `[folder] [--force]` | Build/update the co-occurrence index (no embedder needed) |
| `/graph` | `[out.html] [folder]` | Export the knowledge graph |
| `/map` | `<nota> [--force]` | Radial mind-map rooted on a note → `maps/<stem>.canvas` |
| `/find` | `<query> [--k=N]` | Semantic search |
| `/undo` | `[note-path]` | Undo the last patch on a note |
| `/review` | `[--flush=HASH]` | Inspect the async review queue (deferred ops) |
| `/revert` | `[run-id]` | Revert a whole injection (per-run, LIFO) |
| `/dedup` | `[folder]` | Deduplicate notes (sub-agent) |
| `/curate` | `[folder] [--apply]` | Curate the vault: plan autolink/orphan/dedup/refine work (dry-run; `--apply` executes) |
| `/refine` | `[folder]` | Enrich and normalize notes (sub-agent) |
| `/enrich` | `[folder]` | Enrich note semantics (sub-agent) |
| `/stale` | | List notes whose `documents:` paths have new commits since `code_ref` |
| `/plans` | | List `plans/` notes grouped by `status:` |

**System:** `/help` · `/model` · `/tools` · `/clear` · `/verbose` · `/thinking` · `/vault [path]` (show or switch the active vault for this session) · `/exit`

---

## Configuration

Configure the agent via environment variables (e.g., in a `.env` file). `silica init` writes the essentials for you; the full list with defaults lives in [`.env.example`](.env.example).

| Variable | Description |
| :--- | :--- |
| `SILICA_MODEL` | Chat LLM model identifier (litellm format, e.g. `openrouter/anthropic/claude-sonnet-4-20250514`) |
| `SILICA_PROVIDER` | Chat provider preset: `lmstudio` or `openrouter` |
| `OPENROUTER_API_KEY` | Required when the provider is `openrouter` |
| `SILICA_VAULT` | Vault path. An Obsidian vault (`.obsidian/`) is used verbatim; any other path is repo mode → `docs/silica/` |
| `SILICA_BACKEND` | `fs` (headless filesystem, default) or `cli` (live Obsidian desktop via CDP — adds rollback + live cache) |
| `SILICA_EMBEDDING_MODEL` | Embedding model identifier used for semantic tasks |
| `SILICA_WORKER_MODEL` | Sub-agent worker model (e.g., a small local model for dedup / refinement) |
| `SILICA_GIT_COMMIT` | Git commit safety net for vault writes (`off`, `auto`) |
| `SILICA_TAVILY_API_KEY` | API key for Tavily search (enables the `/web-search` command) |

---

## Quirks & Features

* **Token-Efficient Vault Auditing (`/report`)**: Computes community detection clusters (Louvain modularity), detects god-nodes (high-degree hubs), structural bridges (inter-community connectors), and orphans. Audits and builds a full structural remediation plan for a vault of **1,000+ markdown files in under 10 seconds**.
* **Parallel Worker Sub-Agents**: Cognitive-heavy, long-running batch operations like semantic deduplication (`/dedup`) and detail refinement (`/refine` or `/enrich`) are offloaded to leashed sub-agents. These run concurrently (up to `SILICA_SUBAGENT_MAX_CONCURRENT`) on a separate worker model (e.g., a small local model like `SILICA_WORKER_MODEL`), keeping the main model's context window clean and free.
* **Embedder-Free Concept Modeling**: If an embedding model is offline or unconfigured, Silica's concept matching degrades gracefully. It utilizes a deterministic, local co-occurrence concept graph (`/cooccur`) to query relatedness and label communities in `/graph` exports without making network calls or LLM API queries.
* **Strict Zero-Trust Staging**: Web search queries (`/web-search`) write findings exclusively into the inbox directory (`Inbox/`). External web content is never injected directly into the active knowledge vault without explicit human staging and FSM ingestion review.
* **Git Safety Net**: If `SILICA_GIT_COMMIT=auto` is enabled, Silica automatically commits touched paths to Git after each successful write batch, creating a history checkpoint alongside the interactive `/undo` and `/revert` features.

---

## References

*   **[From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution](https://arxiv.org/abs/2604.11378)** (arXiv:2604.11378, 2026)
*   **[Goal-Autopilot: A Verifiable Anti-Fabrication Firewall for Unattended Long-Horizon Agents](https://arxiv.org/abs/2606.11688)** (arXiv:2606.11688, 2026)
*   **[Is Your Agent Playing Dead? Deployed LLM Agents Exhibit Constraint-Evasive Fabrication and Thanatosis](https://arxiv.org/abs/2606.14831)** (arXiv:2606.14831, 2026)
*   **[Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs](https://arxiv.org/abs/2601.08773)** (arXiv:2601.08773, 2026)
*   **[Predicting new research directions in materials science using large language models and concept graphs](https://doi.org/10.1038/s42256-026-01206-y)** (*Nature Machine Intelligence*, 2026)

Silica's embedder-free near-duplicate detection (`/dedup` command) is inspired by and ports the well-thought-out MinHash design from [Graphify](https://github.com/safishamsi/graphify).

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).

See [LICENSE](LICENSE) for details.
