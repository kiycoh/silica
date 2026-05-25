"""Obsidian Driver — L0 abstraction over the vault I/O substrate.

From SILICA.md §3 L0:
  Adapter typed by DOMAIN, not by transport. Everything else talks to the
  Driver, never to disk or CLI directly. Two interchangeable backends:
  - cli: wraps the official Obsidian CLI (requires desktop app >= 1.12.7)
  - fs:  direct filesystem + index (derived from Hermes scripts)

This module defines the Protocol (interface), domain types, and the
global DRIVER instance selected at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Domain types — the vocabulary of the Driver interface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoteRef:
    """Reference to a note in the vault."""
    name: str           # wikilink-style name (no extension)
    path: str = ""      # relative path within vault (folder/note.md)


@dataclass
class NoteContent:
    """Full content of a note."""
    ref: NoteRef
    content: str
    size: int = 0


@dataclass
class Hit:
    """Search result with context."""
    ref: NoteRef
    line: int = 0
    snippet: str = ""
    score: float = 0.0


@dataclass
class Heading:
    """A heading in a note's outline."""
    level: int          # 1-6
    text: str
    position: int = 0   # char offset
    children: list[Heading] = field(default_factory=list)


@dataclass(frozen=True)
class Link:
    """A link between notes."""
    source: NoteRef
    target: str         # wikilink target (may be unresolved)
    display: str = ""   # display text if aliased


@dataclass
class GraphSnapshot:
    """Snapshot of the vault graph for non-regression diffing."""
    orphans: list[NoteRef] = field(default_factory=list)
    unresolved: list[Link] = field(default_factory=list)
    link_counts: dict[str, int] = field(default_factory=dict)   # note -> outgoing count
    backlink_counts: dict[str, int] = field(default_factory=dict)  # note -> incoming count


@dataclass
class Txn:
    """Transaction handle for snapshot/rollback.

    Three rollback strategies (C3 / ADR-009):
      - inverses:       authoritative list of InverseOp — consumed by silica_restore and
                        the ROLLBACK state. Single source of truth.
      - versions:       derived from inverses (restore_version entries); kept for
                        DRIVER.restore() compatibility until S3.3 unifies on inverses.
      - created_paths:  derived from inverses (delete_created entries); same reason.
    """
    id: str
    refs: list[NoteRef] = field(default_factory=list)
    versions: dict[str, int] = field(default_factory=dict)   # path -> version number
    created_paths: list[str] = field(default_factory=list)   # paths created by write ops
    inverses: list = field(default_factory=list)              # list[InverseOp] — real field, not dynamic attr


# ---------------------------------------------------------------------------
# ObsidianDriver Protocol — the domain interface (SILICA.md §3 L0)
# ---------------------------------------------------------------------------

@runtime_checkable
class ObsidianDriver(Protocol):
    """Domain-typed interface to an Obsidian vault.

    Freshness contract (NORMATIVE from SILICA.md):
      The Driver MUST declare read-after-write semantics. After a create/
      set_prop/move, the Driver guarantees that the next read reflects the
      mutation. If the underlying cache updates asynchronously, the backend
      MUST wait/poll until settled. A method that doesn't respect the same
      freshness contract on both backends is a bug, not a difference.
    """

    # -- discovery / read --------------------------------------------------

    def search_names(self, query: str) -> list[NoteRef]:
        """Search vault note names matching query."""
        ...

    def search_context(self, query: str) -> list[Hit]:
        """Search vault content with line-level context snippets."""
        ...

    def read_note(self, ref: NoteRef | str) -> NoteContent:
        """Read a note's full content by name or ref."""
        ...

    def props_of(self, ref: NoteRef | str) -> dict:
        """Read frontmatter properties (~hundreds of tokens, no body)."""
        ...

    def outline(self, ref: NoteRef | str) -> list[Heading]:
        """Get the heading tree of a note."""
        ...

    # -- graph -------------------------------------------------------------

    def links(self, ref: NoteRef | str) -> list[NoteRef]:
        """Outgoing links from a note."""
        ...

    def backlinks(self, ref: NoteRef | str) -> list[NoteRef]:
        """Incoming links to a note."""
        ...

    def orphans(self) -> list[NoteRef]:
        """Notes with no incoming links."""
        ...

    def unresolved(self) -> list[Link]:
        """Unresolved wikilinks in the vault."""
        ...

    def graph_snapshot(self, refs: list[NoteRef] | None = None) -> GraphSnapshot:
        """Graph snapshot for non-regression gating.

        If refs is provided, performs an incremental snapshot covering only
        the touched notes and their 1-hop neighborhood.
        """
        ...

    # -- write (graph-safe) ------------------------------------------------

    def create(self, path: str, content: str) -> NoteRef:
        """Create a new note. Path is relative to vault root. Raises if file exists."""
        ...

    def overwrite(self, path: str, content: str) -> NoteRef:
        """Overwrite an existing note in-place, preserving history.

        Unlike delete+create, this MUST NOT destroy Obsidian's version history
        or break block-references. Use for patch and overwrite op types.
        The CLI backend uses `obsidian create path=... overwrite=true`.
        The FS backend writes the file directly.
        """
        ...

    def append(self, ref: NoteRef | str, content: str) -> None:
        """Append content to an existing note."""
        ...

    def set_prop(self, ref: NoteRef | str, name: str, value: Any, type_: str = "text") -> None:
        """Set a frontmatter property on a note."""
        ...

    def move(self, ref: NoteRef | str, to: str) -> None:
        """Move/rename a note. Updates wikilinks (graph-safe)."""
        ...

    def delete(self, ref: NoteRef | str) -> None:
        """Delete a note from the vault."""
        ...

    # -- advanced ----------------------------------------------------------

    def list_files(self, folder: str = "") -> list[NoteRef]:
        """List all markdown files, optionally filtered by folder."""
        ...

    def base_query(self, base: str, view: str) -> list[dict]:
        """Query an Obsidian Base (DB on frontmatter)."""
        ...

    # -- transactionality --------------------------------------------------

    def snapshot_versions(self, refs: list[NoteRef]) -> Txn:
        """Snapshot current versions for later rollback."""
        ...

    def restore(self, txn: Txn) -> None:
        """Rollback to a previous snapshot via history/sync restore."""
        ...
