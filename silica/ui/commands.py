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
        summary="audit strutturale del vault → steering loop",
        examples=(
            "/report Concepts/ML",
            "/report --embeddings",
            "/report --top-k=15 --embeddings",
        ),
        home_pin=True,
    ),
    Command(
        name="/inject",
        group="workflow",
        usage="<file...> --target=DIR [--hub=H]",
        summary="porta una nota nel vault via Injector FSM",
        examples=("/inject Inbox/note.md --target=Concepts/AI",),
        home_pin=True,
    ),
    # Direct — immediate, no LLM round-trip
    Command(
        name="/status",
        group="direct",
        usage="[run_id]",
        summary="progress digest dell'ultimo run",
    ),
    Command(
        name="/embed",
        group="direct",
        usage="[folder] [--force]",
        summary="costruisci/aggiorna indice embedding",
    ),
    Command(
        name="/cooccur",
        group="direct",
        usage="[folder] [--force]",
        summary="costruisci/aggiorna indice co-occorrenza (senza embedder)",
    ),
    Command(
        name="/graph",
        group="direct",
        usage="[out.html] [folder]",
        summary="esporta grafo della conoscenza",
    ),
    Command(
        name="/find",
        group="direct",
        usage="<query> [--k=N]",
        summary="ricerca semantica",
        home_pin=True,
    ),
    Command(
        name="/undo",
        group="direct",
        usage="[note-path]",
        summary="annulla l'ultima patch su una nota",
    ),
    Command(
        name="/revert",
        group="direct",
        usage="[run-id]",
        summary="annulla un'intera iniezione (per-run, LIFO)",
    ),
    Command(
        name="/dedup",
        group="direct",
        usage="[folder]",
        summary="deduplica (sub-agent)",
    ),
    Command(
        name="/refine",
        group="direct",
        usage="[folder]",
        summary="arricchisci e normalizza note (sub-agent)",
        home_pin=True,
    ),
    Command(
        name="/enrich",
        group="direct",
        usage="[folder]",
        summary="arricchisci semantica note (sub-agent)",
        home_pin=True,
    ),
    # System
    Command(
        name="/help",
        group="system",
        usage="",
        summary="mostra questo aiuto",
    ),
    Command(
        name="/model",
        group="system",
        usage="",
        summary="mostra il modello LLM corrente",
    ),
    Command(
        name="/tools",
        group="system",
        usage="",
        summary="elenca i tool registrati",
    ),
    Command(
        name="/clear",
        group="system",
        usage="",
        summary="resetta la cronologia conversazione",
    ),
    Command(
        name="/verbose",
        group="system",
        usage="",
        summary="cicla tool progress: off → new → all → verbose",
    ),
    Command(
        name="/thinking",
        group="system",
        usage="",
        summary="toggle display del reasoning block",
    ),
    Command(
        name="/exit",
        group="system",
        usage="",
        summary="esci da silica",
    ),
)


def command_names() -> tuple[str, ...]:
    return tuple(c.name for c in COMMANDS)


def render_help() -> None:
    from rich.padding import Padding

    from silica.ui.console import CONSOLE
    from silica.ui.style import GROUP_STYLE, command_table

    CONSOLE.print()
    CONSOLE.print("  [bold]Comandi silica[/]")
    CONSOLE.print()

    workflow = [c for c in COMMANDS if c.group == "workflow"]
    direct = [c for c in COMMANDS if c.group == "direct"]
    system = [c for c in COMMANDS if c.group == "system"]

    CONSOLE.print(f"  [bold {GROUP_STYLE['workflow']}]Workflow[/]  [dim]· agent-directed[/]")
    CONSOLE.print(Padding(command_table(workflow, name_style=f"bold {GROUP_STYLE['workflow']}"), (0, 0, 0, 4)))
    CONSOLE.print()
    CONSOLE.print()

    CONSOLE.print(f"  [bold {GROUP_STYLE['direct']}]Diretti[/]  [dim]· immediati, senza LLM[/]")
    CONSOLE.print(Padding(command_table(direct, name_style=f"bold {GROUP_STYLE['direct']}"), (0, 0, 0, 4)))
    CONSOLE.print()
    CONSOLE.print()

    sys_line = "  ·  ".join(c.name for c in system)
    CONSOLE.print(f"  [bold {GROUP_STYLE['system']}]Sistema[/]")
    CONSOLE.print(f"    [dim]{sys_line}[/]")
    CONSOLE.print()
