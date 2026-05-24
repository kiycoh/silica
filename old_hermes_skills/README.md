# Obsidian Refiner & Note-Taking Pipelines (Hermes)

Hermes is an automated note-taking orchestration toolset designed to manage, ingest, restructure, and unify Markdown notes within Obsidian vaults. It streamlines obsidian workflows using python-based mechanical pipelines, structural validation linting, and semantic mapping.

---

## Directory Structure

This repository is organized into self-contained modular components representing different pipelines and helper utilities:

```text
├── hermes_common/               # Shared package (linter, bulk_writer, templates, etc.)
├── obsidian/                    # Core Obsidian vault integration & cli bindings
│   └── SKILL.md                 # Basic configuration & vault resolution rules
├── obsidian-injector/           # Ingestion pipeline from inbox to target vault
│   ├── prompts/                 # Context prompts for distillation
│   ├── scripts/                 # Python scripts for recon, payload packaging, etc.
│   ├── SKILL.md                 # Configuration & setup details
│   └── README.md                # Sub-component documentation
├── obsidian-refiner/            # Monolith decoupling & note restructuring pipeline
│   ├── scripts/                 # Inspection, split monolithic planning, etc.
│   ├── templates/               # Standard curation layouts
│   └── SKILL.md                 # Curation guidelines & styling instructions
├── obsidian-dedup/              # Duplicate note resolver
│   ├── scripts/                 # Find and merge tools
│   └── SKILL.md                 # Duplicate consolidation policies
├── vault-semantic-mapping/      # Semantic classifier for incoming notes
│   ├── templates/               # JSON formatting templates
│   └── SKILL.md                 # Mapping workflow & classification guidelines
└── HERMES.md                    # Root playbook containing comprehensive pipeline architecture
```

---

## Features

- **Automated Ingestion (Injector)**: Automatically crawls target inbox folders, performs collision checks with vault notes, generates subagent prompts, validates actions, and writes output programmatically.
- **Hub-and-Spoke Decomposition (Refiner)**: Splits long, monolithic markdown files into structured, decoupled atomic Spoke notes connected back to a central Hub index note.
- **Note Enrichment & Formatting**: Detects empty or lean notes (<600 characters), performs web searches to gather definitions/equations, reformats frontmatter keys to lower-case, and rewrites notes using Obsidian Flavored Markdown (OFM) syntax.
- **Duplicate Unification (Dedup)**: Finds duplicate note basenames scattered across different directories, merges content seamlessly, and purges obsolete redundant files.
- **Linter & Verification**: A static linter enforcing Obsidian metadata schemas, cross-note wikilinks, AI provenance headers, and note character limits (max 6,000 characters).

---

## Quick Start

### 1. Ingestion (Obsidian Injector)

To run the automated injection pipeline, perform the following steps:

#### Step A: Mechanical Reconnaissance
Identify target files in the inbox and scan for collisions against your active vault:
```bash
python3 obsidian-injector/scripts/recon.py \
    --inbox "/path/to/inbox" \
    --vault "/path/to/vault" > /tmp/recon.json
```

#### Step B: Pre-distill & Package Payload
Generate the concepts excerpt mapping:
```bash
python3 obsidian-injector/scripts/distiller_payload.py \
    --recon-report /tmp/recon.json \
    --out /tmp/distiller_payload.json
```

#### Step C: Run Curation Operations
Once you generate operations through internal reasoning or subagents, parse and validate the updates:
```bash
# Validate generated changes
python3 obsidian-injector/scripts/validate_operations.py \
    --operations /tmp/distiller_output.json \
    --payload /tmp/distiller_payload.json \
    --target "TargetFolder" \
    --out /tmp/operations.validated.json

# Execute bulk write updates to vault
python3 hermes_common/bulk_writer.py \
    --operations /tmp/operations.validated.json
```

### 2. Duplicate Check (Obsidian Dedup)

Scan for matching note names located in different vault folders:
```bash
python3 obsidian-dedup/scripts/find_duplicates.py \
    --vault "/path/to/vault" \
    --folder "Optional-Subfolder"
```

---

## Configuration

The pipelines rely on the following inputs during execution:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--inbox` | Path | Folder containing source `.md` files to process | None (Required) |
| `--vault` | Path | Root folder of the active Obsidian vault | None (Required) |
| `--target` | String | Sub-folder path inside the vault to output new notes | None (Required) |
| `--limit` | Integer | Maximum files to scan in a single batch (Recon) | None |
| `--offset` | Integer | Starting index for pagination (Recon) | `0` |

---

## Styling Guidelines

Any note modified or created by the pipelines must conform to the **Obsidian Flavored Markdown (OFM)** standards:
- **YAML Frontmatter**: Every note must have standard frontmatter tags (lowercase, hyphen-separated, e.g., `machine-learning`, `reti-neurali`).
- **Atomicity**: Notes must be concise (ideally ~40 lines or maximum 6,000 characters).
- **Wikilinks**: Sub-notes (Spokes) must explicitly link back to their parent Hub page (e.g., `[[Hub Name]]`).
- **LaTeX & Mermaid**: Use block-level Math equations (`$$...$$`) and standard Mermaid syntax for relation diagrams.

---

## Documentation

- [Core Playbook](./HERMES.md) — Comprehensive explanation of workflow phases, pipeline controls, and safety fallback conditions.
- [Obsidian CLI Skills](./obsidian/SKILL.md) — Commands and configurations for interacting with Obsidian CLI tools.
- [Injector Skill](./obsidian-injector/SKILL.md) — Deep dive into the recon and ingestion pipeline internals.
- [Refiner Skill](./obsidian-refiner/SKILL.md) — Structure templates, atomicity rules, and frontmatter constraints.

---

## License

This project is licensed under the [MIT License](./LICENSE).
