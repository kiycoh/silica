"""Atomic tools — L0 façades, 1:1 on Obsidian CLI commands.

From SILICA.md §4.2:
  Atomic tools are single Obsidian-native operations, 1:1 on a CLI command
  or a pure kernel function. They are the base vocabulary — called by both
  the agent and the pipeline.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool


# ---------------------------------------------------------------------------
# Read / Discovery
# ---------------------------------------------------------------------------

class SearchArgs(BaseModel):
    query: str = Field(description="Text to search for in note names in the vault")

@tool(SearchArgs, cls="atomic")
def silica_search(query: str) -> list:
    """Search for notes in the vault by name. Returns the names of notes matching the query."""
    refs = DRIVER.search_names(query)
    return [{"name": r.name, "path": r.path} for r in refs]


class SearchContextArgs(BaseModel):
    query: str = Field(description="Text to search for within the content of vault notes")

@tool(SearchContextArgs, cls="atomic")
def silica_search_context(query: str) -> list:
    """Search the content of the vault with context (snippets + line numbers). Useful for finding mentions of a concept."""
    hits = DRIVER.search_context(query)
    return [
        {"name": h.ref.name, "path": h.ref.path, "line": h.line, "snippet": h.snippet}
        for h in hits
    ]


class ReadNoteArgs(BaseModel):
    name: str = Field(description="Name of the note to read (wikilink style, without file extension)")

@tool(ReadNoteArgs, cls="atomic")
def silica_read_note(name: str) -> str:
    """Reads the complete content of a note in the vault by name (wikilink-style resolution). DO NOT use paths."""
    nc = DRIVER.read_note(name)
    return nc.content


class PropsArgs(BaseModel):
    name: str = Field(description="Name of the note to read the frontmatter properties from")

@tool(PropsArgs, cls="atomic")
def silica_props(name: str) -> dict:
    """Reads the frontmatter properties of a note (saves tokens, does not read the body)."""
    return DRIVER.props_of(name)


class OutlineArgs(BaseModel):
    name: str = Field(description="Name of the note to display the heading tree of")

@tool(OutlineArgs, cls="atomic")
def silica_outline(name: str) -> list:
    """Displays the heading tree (H1-H6) of a note."""
    headings = DRIVER.outline(name)
    return [{"level": h.level, "text": h.text} for h in headings]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class LinksArgs(BaseModel):
    name: str = Field(description="Name of the note to list outgoing links from")

@tool(LinksArgs, cls="atomic")
def silica_links(name: str) -> list:
    """Lists outgoing links from a note (connected notes)."""
    refs = DRIVER.links(name)
    return [{"name": r.name, "path": r.path} for r in refs]


class BacklinksArgs(BaseModel):
    name: str = Field(description="Name of the note to list incoming links (backlinks) for")

@tool(BacklinksArgs, cls="atomic")
def silica_backlinks(name: str) -> list:
    """Lists incoming links (backlinks) pointing to a note."""
    refs = DRIVER.backlinks(name)
    return [{"name": r.name, "path": r.path} for r in refs]


class EmptyArgs(BaseModel):
    pass

@tool(EmptyArgs, cls="atomic")
def silica_orphans() -> list:
    """Lists orphan notes (notes with no incoming links) in the vault."""
    refs = DRIVER.orphans()
    return [{"name": r.name, "path": r.path} for r in refs]


@tool(EmptyArgs, cls="atomic")
def silica_unresolved() -> list:
    """Lists unresolved wikilinks in the vault (links pointing to non-existent notes)."""
    links = DRIVER.unresolved()
    return [{"target": l.target} for l in links]


# ---------------------------------------------------------------------------
# List files
# ---------------------------------------------------------------------------

class ListFilesArgs(BaseModel):
    folder: str = Field(default="", description="Optional folder path to filter results")

@tool(ListFilesArgs, cls="atomic")
def silica_files(folder: str = "") -> list:
    """Lists all markdown files in the vault, optionally filtered by folder."""
    refs = DRIVER.list_files(folder)
    return [{"name": r.name, "path": r.path} for r in refs]


class ExistsArgs(BaseModel):
    path: str = Field(description="Relative path of the note in the vault")

@tool(ExistsArgs, cls="atomic")
def silica_exists(path: str) -> bool:
    """Verifies if a note exists in the vault (including the inbox) given its relative path."""
    try:
        DRIVER.read_note(path)
        return True
    except Exception:
        return False


@tool(EmptyArgs, cls="atomic")
def silica_inbox_ls() -> list:
    """Lists all files in the Inbox folder (inbox_dir)."""
    refs = DRIVER.list_inbox_files()
    return [{"name": r.name, "path": r.path} for r in refs]

