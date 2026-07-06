from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Command:
    name: str
    group: str       # "workflow" | "direct" | "system"
    usage: str
    summary: str
    examples: tuple[str, ...] = ()
    home_pin: bool = False


COMMANDS: tuple[Command, ...] = (
    # Workflow — agent-directed
    Command(
        name="/report",
        group="workflow",
        usage="[folder] [--top-k=N] [--embeddings]",
        summary="structural audit of the vault → steering loop",
        examples=(
            "/report Concepts/ML",
            "/report --embeddings",
            "/report --top-k=15 --embeddings",
        ),
        home_pin=True,
    ),
    Command(
        name="/ingest",
        group="workflow",
        usage="<file...> [--target=DIR] [--hub=H]",
        summary="bring files in: notes via Injector FSM, code as skeleton stubs",
        examples=(
            "/ingest Inbox/note.md --target=Concepts/AI",
            "/ingest silica/cli.py",
            "/ingest paper.pdf --target=Concepts/AI",
        ),
        home_pin=True,
    ),
    Command(
        name="/convert",
        group="direct",
        usage="<file...> [--target=DIR]",
        summary="transcode a non-.md file (PDF) into a markdown note in the inbox",
        examples=(
            "/convert paper.pdf",
            "/convert paper.pdf --target=Concepts/AI",
        ),
    ),
    Command(
        name="/web-search",
        group="direct",
        usage='"<concept>" [--max-searches=N]',
        summary="research a concept on the web → cited findings note in the Inbox (then /ingest)",
        examples=(
            '/web-search "retrieval-augmented generation"',
            '/web-search "graph neural networks" --max-searches=6',
        ),
    ),
    Command(
        name="/organize",
        group="workflow",
        usage='"<intent>" [--scope=FOLDER] [--file=taxonomy.yaml] [--merge] [--move-uncategorized] [--apply]',
        summary="classify and reorganize vault notes according to a taxonomy",
        examples=(
            '/organize "put AI notes in Concepts/AI, cooking notes in Life"',
            '/organize "archive Acme docs under Clients/Acme" --merge',
            "/organize --file=taxonomy.yaml --apply",
            "/organize --scope=Inbox",
        ),
        home_pin=True,
    ),
    # Direct — immediate, no LLM round-trip
    Command(
        name="/status",
        group="direct",
        usage="[run_id]",
        summary="progress digest of the last run",
    ),
    Command(
        name="/embed",
        group="direct",
        usage="[folder] [--force]",
        summary="build/update embedding index",
    ),
    Command(
        name="/cooccur",
        group="direct",
        usage="[folder] [--force]",
        summary="build/update co-occurrence index (without embedder)",
    ),
    Command(
        name="/graph",
        group="direct",
        usage="[out.html] [folder]",
        summary="export knowledge graph",
    ),
    Command(
        name="/map",
        group="direct",
        usage="<nota> [--force]",
        summary="radial mind-map rooted on a note → maps/<stem>.canvas",
    ),
    Command(
        name="/find",
        group="direct",
        usage="<query> [--k=N]",
        summary="semantic search",
        home_pin=True,
    ),
    Command(
        name="/undo",
        group="direct",
        usage="[note-path]",
        summary="undo the last patch on a note",
    ),
    Command(
        name="/review",
        group="direct",
        usage="[--flush=HASH]",
        summary="inspect the async review queue (deferred ops)",
    ),
    Command(
        name="/revert",
        group="direct",
        usage="[run-id]",
        summary="revert a whole injection (per-run, LIFO)",
    ),
    Command(
        name="/dedup",
        group="direct",
        usage="[folder]",
        summary="deduplicate (sub-agent)",
    ),
    Command(
        name="/curate",
        group="direct",
        usage="[folder] [--apply]",
        summary="curate the vault: plan autolink/orphan/dedup/refine work (dry-run; --apply executes)",
    ),
    Command(
        name="/refine",
        group="direct",
        usage="[folder]",
        summary="enrich and normalize notes (sub-agent)",
        home_pin=True,
    ),
    Command(
        name="/enrich",
        group="direct",
        usage="[folder]",
        summary="enrich note semantics (sub-agent)",
        home_pin=True,
    ),
    Command(
        name="/stale",
        group="direct",
        usage="",
        summary="list notes whose documents: paths have new commits since code_ref",
    ),
    Command(
        name="/plans",
        group="direct",
        usage="",
        summary="list plans/ notes grouped by status: (todo|in-progress|blocked|done)",
    ),
    # System
    Command(
        name="/vault",
        group="system",
        usage="[path]",
        summary="show the active vault, or switch to another for this session",
    ),
    Command(
        name="/help",
        group="system",
        usage="",
        summary="show this help",
    ),
    Command(
        name="/model",
        group="system",
        usage="",
        summary="show the current LLM model",
    ),
    Command(
        name="/tools",
        group="system",
        usage="",
        summary="list registered tools",
    ),
    Command(
        name="/clear",
        group="system",
        usage="",
        summary="reset conversation history",
    ),
    Command(
        name="/verbose",
        group="system",
        usage="",
        summary="cycle tool progress: off → new → all → verbose",
    ),
    Command(
        name="/thinking",
        group="system",
        usage="",
        summary="toggle display of the reasoning block",
    ),
    Command(
        name="/exit",
        group="system",
        usage="",
        summary="exit silica",
    ),
)


def command_names() -> tuple[str, ...]:
    return tuple(c.name for c in COMMANDS)


def render_help() -> None:
    from rich.padding import Padding

    from silica.ui.console import CONSOLE
    from silica.ui.style import GROUP_STYLE, command_table

    CONSOLE.print()
    CONSOLE.print("  [bold]silica commands[/]")
    CONSOLE.print()

    workflow = [c for c in COMMANDS if c.group == "workflow"]
    direct = [c for c in COMMANDS if c.group == "direct"]
    system = [c for c in COMMANDS if c.group == "system"]

    CONSOLE.print(f"  [bold {GROUP_STYLE['workflow']}]Workflow[/]  [dim]· agent-directed[/]")
    CONSOLE.print(Padding(command_table(workflow, name_style=f"bold {GROUP_STYLE['workflow']}"), (0, 0, 0, 4)))
    CONSOLE.print()
    CONSOLE.print()

    CONSOLE.print(f"  [bold {GROUP_STYLE['direct']}]Direct[/]  [dim]· immediate, no LLM[/]")
    CONSOLE.print(Padding(command_table(direct, name_style=f"bold {GROUP_STYLE['direct']}"), (0, 0, 0, 4)))
    CONSOLE.print()
    CONSOLE.print()

    sys_line = "  ·  ".join(c.name for c in system)
    CONSOLE.print(f"  [bold {GROUP_STYLE['system']}]System[/]")
    CONSOLE.print(f"    [dim]{sys_line}[/]")
    CONSOLE.print()
