# Silica Agent 🪨

> A conversational CLI agent with an Obsidian-native toolset designed for deterministic, gate-enforced note curation and vault management.

---

## What is Silica?

**Silica** is a specialized agentic framework that operates directly on an Obsidian vault. Unlike general-purpose agents that run arbitrary shell scripts or interact with standard filesystems, Silica’s entire capability set is Obsidian-native. 

Silica implements a **dual-consumer, single-toolset** architecture:
1. **Conversational REPL Loop:** An LLM-controlled agent that has the flexibility to explore, search, and edit notes conversationally.
2. **Deterministic Pipelines (e.g., Injector):** Strict, hardcoded state machines with quality gates (e.g., rollback-on-error, structural linter enforcement) that execute critical ingestion and refinement tasks unattended with a zero-fault tolerance.

---

## Architectural Layers (L0–L4)

Silica is structured into five distinct, decoupled layers:

- **L0: Obsidian Driver:** The domain-specific I/O adapter. Features a primary `cli` backend (bridging the live Obsidian desktop app via a CDP interface for graph-safe updates) and a degraded `fs` backend (direct filesystem interaction).
- **L1: Kernel:** Pure, deterministic Python logic containing OFM linters, partition calculators, collision detection, and sanitizers. No LLM dependency.
- **L2: Worker:** Stateless, CoT-intensive sub-agents (e.g., Distiller, Merger) that execute specialized semantic conversions outside the vault.
- **L3: Router/Orchestrator:** Deterministic finite state machines (FSMs) that coordinate execution across L0–L2 and enforce pre-write and post-write gates.
- **L4: Recipes:** Declarative YAML pipelines mapping routing phases dynamically.

---

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended)
- Obsidian Desktop App (running live for `cli` backend operations)

### Installation
Initialize and install the project in editable mode:

```bash
uv pip install -e .
```

### Running the Agent
Launch the conversational agent:

```bash
uv run silica
```

---

## Pipeline Gates & Rollback

The core pipeline (**Injector**) operates under strict non-regression constraints:
- **Pre-write Gate:** Rejects operations if the LLM's suggested updates have a rejection rate $\ge 10\%$.
- **Transaction Rollback:** Takes snapshot versions of target notes using Obsidian's native file history. If a gate fails, the state is immediately reverted.
- **Post-write Gate:** Runs the Obsidian-Flavored Markdown (OFM) linter to verify formatting, backlinks, and hub-and-spoke atomicity constraints.

---

## Directory Structure

```
silica-agent/
├── pyproject.toml              # Project dependencies & entry points
├── SILICA.md                   # Core architectural charter
├── silica/
│   ├── cli.py                  # REPL CLI Interface
│   ├── agent/                  # Agentic loop & LLM router
│   ├── driver/                 # L0: Obsidian Protocol, CLI & FS backends
│   ├── kernel/                 # L1: Pure mechanical scripts (Linter, Sanitizer, etc.)
│   ├── router/                 # L3: FSM Orchestrator
│   └── tools/                  # Atomic, composed, and wrapped vault tools
└── tests/                      # Smoke & golden parity tests
```

---

## License

This project is licensed under the MIT License.
