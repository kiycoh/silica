# Silica Agent 🪨

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Obsidian Native](https://img.shields.io/badge/Obsidian-Native-purple.svg)](https://obsidian.md/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by UV](https://img.shields.io/badge/package--manager-uv-brightgreen.svg)](https://github.com/astral-sh/uv)

> **Silica** is a conversational CLI agent and automated curation engine designed from the ground up to be **Obsidian-native**. It operates directly on your knowledge base, managing daily logs, linking notes, and structuring concepts while preserving vault integrity through strict, deterministic quality gates.

---

## Table of Contents
- [🌟 Overview](#-overview)
- [🎯 Who is Silica for?](#-who-is-silica-for)
- [⚡ How Silica Distinguishes Itself](#-how-silica-distinguishes-itself)
- [🏗️ Layered Architecture (L0–L4)](#%EF%B8%8F-layered-architecture-l0l4)
- [🔍 The Dual-Consumer Paradigm](#-the-dual-consumer-paradigm)
- [📜 The Golden Rules of Curation](#-the-golden-rules-of-curation)
- [🚀 Quick Start](#-quick-start)
- [📂 Directory Structure](#-directory-structure)
- [⚖️ License](#%EF%B8%8F-license)

---

## 🌟 Overview

At its core, **Silica** addresses a fundamental challenge in agentic AI: *how do we let large language models organize, refactor, and enrich our personal knowledge bases without risking data corruption or structural chaos?*

Instead of treating the vault as a raw, unstructured directory of markdown files, Silica communicates directly with Obsidian. It reads live metadata caches, resolves wiki-links, audits graph structures, and applies modifications via wrapped, safety-hardened tools.

```mermaid
graph TD
    User([User Request]) --> Router{Silica Router}
    Router -->|Conversational Loop| REPL[LLM Agent REPL]
    Router -->|Deterministic Run| FSM[Pipeline Orchestrator FSM]
    
    REPL -->|Call Tool| Toolset[Obsidian-Native Toolset]
    FSM -->|Step Execution| Toolset
    
    Toolset -->|L0 Driver| Driver{Obsidian Driver}
    Driver -->|cli backend / CDP| LiveApp[Live Obsidian Electron App]
    Driver -->|fs backend / Fallback| RawFiles[Filesystem Vault]
```

---

## 🎯 Who is Silica for?

Silica is tailored for power users who treat their notes as an external brain:

*   **PKM (Personal Knowledge Management) Practitioners:** Scholars, researchers, and writers who use Obsidian as a semantic web and need automated help sorting their daily inbox.
*   **Safety-First Note Takers:** Users who want to automate tedious chores (formatting metadata, resolving tags, deduplicating notes) but demand guaranteed non-regression and transaction rollbacks.
*   **AI Curation Enthusiasts:** Anyone looking to integrate LLMs into complex workflows where accuracy, facts density, and strict Markdown compliance are non-negotiable.

---

## ⚡ How Silica Distinguishes Itself

While other plugins and tools act as simple, human-in-the-loop chat interfaces, Silica is a **production-grade curation engine with mathematical gates** built for unattended execution.

### 1. Dual-Consumer Architecture
Silica decouples conversational freedom from pipeline safety. The same underlying Obsidian toolset is shared by two distinct execution modes:
*   **The Conversational REPL:** A high-freedom agent that reasons, searches, and interacts with notes on the fly.
*   **Deterministic FSM Pipelines:** Zero-freedom state machines (like the *Injector* or *Refiner*) that process batches using fixed recipes, strict validation gates, and automatic rollbacks.

### 2. Live Graph Safety & Non-Regression
Silica does not just edit text; it protects the graph.
*   **Graph Diffing & Gates:** Silica snapshots the vault graph before running changes. The LINT stage enforces a mathematical non-regression gate:
    *   *Unplanned Orphans:* Rejects if new orphan notes are created (unless explicitly created by the current batch).
    *   *New Unresolved Links:* Rejects if new unresolved links are introduced from pre-existing notes (excluding intentional forward references between newly written notes).
    *   *No Broken Backlinks:* Rejects if incoming link counts decrease for pre-existing notes.
*   **Robust Transaction Rollbacks:** Before applying edits, Silica builds a transaction log. If any gate fails, the system executes an immediate, atomic rollback using captured `InverseOp` records containing the exact prior note content to revert changes reliably.
*   **The Golden Fallback Oracle:** The dual-backend system includes a live `cli` backend (bridged directly to Electron's live cache for graph-safe updates) and an `fs` backend (direct disk interaction). The `fs` backend serves as a "golden reference" against which the live CLI is continuously validated to prevent regression.

### 3. Safety-Hardened Wrapped Tools
Instead of relying on prompt instructions (e.g., *"never delete notes"*), Silica hardcodes invariants into the tool execution layer itself. For example, `silica_move` natively handles internal links redirection so that links never break, and `silica_delete` enforces strict anti-deletion policies.

### 4. Convergence & Loop Guards
To prevent conversational agent runs from spinning into infinite loops, the agentic loop implements a strict convergence guard. It aborts the execution run if any tool call with identical arguments fails 3 times consecutively, and injects helpful warning context at 2 consecutive failures.

### 5. Deferred Operations Staging
Operations that fail validation or gates are not discarded. They are preserved in a local content-hashed deferred store. Users can list or retry these operations later using `silica_deferred_retry` without repeating the full expensive pipeline cycles.

---

## 🏗️ Layered Architecture (L0–L4)

Silica is organized into five decoupled layers, mapping directly from low-level I/O to high-level declarative workflows:

| Layer | Component | Description |
| :--- | :--- | :--- |
| **L4** | **Recipes** | Declarative YAML files (e.g., `injector.yaml`, `refiner.yaml`) defining the routing stages of curation pipelines. |
| **L3** | **Router / Orchestrator** | Deterministic Finite State Machine that drives recipes, validates gates, and manages snapshot/rollback cycles. |
| **L2** | **Worker Semantics** | Stateless, LLM-based sub-agents (e.g., *Distiller*, *Merger*) that execute complex Chain-of-Thought tasks and return structured JSON operations. |
| **L1** | **Mechanical Kernel** | Pure, deterministic Python logic for parsing frontmatter, calculating partitions, running linters, and scoring validation rates. No LLMs here. |
| **L0** | **Obsidian Driver** | The unified, domain-specific I/O protocol. Houses the primary `cli` adapter (bridges the live Obsidian desktop app via a CDP interface) and the `fs` fallback database. |

---

## 🔍 The Dual-Consumer Paradigm

The core design guarantees that conversational flexibility and structured pipelines never compromise each other:

| Feature | Conversational Loop (`silica`) | Critical FSM Pipelines (e.g., Ingestion) |
| :--- | :--- | :--- |
| **Control** | LLM-in-the-loop, high autonomy | Finite State Machine, zero autonomy |
| **Determinism** | Non-deterministic | Fully deterministic & reproducible |
| **Human Presence** | Human-in-the-loop | Unattended / Background cron |
| **Guarantees** | Best-effort | Validation gates + automatic rollback |
| **Example Goal** | *"Clean up my notes on neural networks"* | Run the `injector` recipe on incoming PDF payload |

---

## 📜 The Golden Rules of Curation

Every operation applied by Silica adheres to the following system-wide rules:

1.  **Anti-Deletion Policy:** Never silently delete content. Prefer appending, merging, or refactoring.
2.  **Modular Atomicity (Hub-and-Spoke):** Notes should represent single atomic concepts (typically under 40 lines or 6,000 characters), linked back to a central `[[Hub]]`.
3.  **Obsidian-Flavored Markdown (OFM):** Strict usage of callouts (`> [!tip]`), block refs (`^id`), Mermaid diagrams, and LaTeX math blocks.
4.  **Factual Density:** Focus on extracting raw definitions, formulas, and visual examples rather than writing generic summaries.
5.  **AI Tracking:** Automatically tags new or updated notes with `ai_generated: true` in the frontmatter.
6.  **Tag Normalization:** All tags are forced to lowercase with hyphens (e.g., `#machine-learning` instead of `#MachineLearning`).

---

## 🚀 Quick Start

### Prerequisites
*   Python 3.11+
*   [uv](https://github.com/astral-sh/uv) (recommended package installer)
*   For the primary `cli` backend: Obsidian Desktop App (running live)

### Installation
Clone the repository and install it in editable mode:

```bash
# Clone the repository
git clone https://github.com/kiycoh/silica-agent.git
cd silica-agent

# Install dependencies and project in editable mode
uv pip install -e .
```

### Running the Agent
Start the interactive conversational REPL session:

```bash
uv run silica
```

#### CLI Slash Commands
When inside the conversational REPL, the following slash commands are available:
*   `/clear` — Clears the terminal screen, resets the model message history back to the system prompt, and initializes a clean interactive session.
*   `/verbose` — Cycles through tool progress display levels: `off` → `new` → `all` → `verbose`. The `verbose` mode activates system `DEBUG` logging with a customized, human-friendly formatting wrapper.
*   `/thinking` — Toggles the rendering of the LLM's reasoning and thought blocks.
*   `/model` — Shows the currently active LLM model.
*   `/tools` — Lists all registered Obsidian-native tools.
*   `/help` — Displays the slash command help menu.
*   `/exit` or `/quit` — Exists the conversational session.

For background pipeline runs (e.g., executing the injector recipe):

```bash
# Example command executing the ingestion pipeline directly
uv run python -m silica.router.orchestrator --recipe injector --inbox ./inbox
```

---

## 📂 Directory Structure

```
silica-agent/
├── pyproject.toml              # Dependencies & entry points
├── docs/                       # Core architectural charters and review notes
│   ├── SILICA.md               # Original project foundation charter
│   └── ...                     
├── silica/
│   ├── cli.py                  # TUI / REPL CLI entry point
│   ├── agent/                  # Agent loop, LLM call abstractions, & delegate worker pool
│   ├── driver/                 # L0: Obsidian I/O driver (CDP CLI & FS backends)
│   ├── kernel/                 # L1: Pure mechanical modules (linter, parser, graph diff)
│   ├── router/                 # L3: FSM recipe orchestrator
│   ├── recipes/                # L4: YAML-defined pipeline flows
│   ├── tools/                  # Atomic, composed, and wrapped vault tools
│   └── workers/                # L2: Stateless semantic sub-agents (Distiller prompts & logic)
└── tests/                      # System integration tests & regression golden tests
```

---

## ⚖️ License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
