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
    query: str = Field(description="Testo da cercare nei nomi delle note del vault")

@tool(SearchArgs, cls="atomic")
def silica_search(query: str) -> list:
    """Cerca note nel vault per nome. Restituisce i nomi delle note che corrispondono alla query."""
    refs = DRIVER.search_names(query)
    return [{"name": r.name, "path": r.path} for r in refs]


class SearchContextArgs(BaseModel):
    query: str = Field(description="Testo da cercare nel contenuto delle note del vault")

@tool(SearchContextArgs, cls="atomic")
def silica_search_context(query: str) -> list:
    """Cerca nel contenuto del vault con contesto (snippet + righe). Utile per trovare menzioni di un concetto."""
    hits = DRIVER.search_context(query)
    return [
        {"name": h.ref.name, "path": h.ref.path, "line": h.line, "snippet": h.snippet}
        for h in hits
    ]


class ReadNoteArgs(BaseModel):
    name: str = Field(description="Nome della nota da leggere (stile wikilink, senza estensione)")

@tool(ReadNoteArgs, cls="atomic")
def silica_read_note(name: str) -> str:
    """Legge il contenuto completo di una nota del vault per nome (risoluzione wikilink-style). NON usare path."""
    nc = DRIVER.read_note(name)
    return nc.content


class PropsArgs(BaseModel):
    name: str = Field(description="Nome della nota di cui leggere le proprietà frontmatter")

@tool(PropsArgs, cls="atomic")
def silica_props(name: str) -> dict:
    """Legge le proprietà frontmatter di una nota (~centinaia di token, senza il corpo)."""
    return DRIVER.props_of(name)


class OutlineArgs(BaseModel):
    name: str = Field(description="Nome della nota di cui visualizzare l'albero degli heading")

@tool(OutlineArgs, cls="atomic")
def silica_outline(name: str) -> list:
    """Mostra l'albero degli heading (H1-H6) di una nota."""
    headings = DRIVER.outline(name)
    return [{"level": h.level, "text": h.text} for h in headings]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class LinksArgs(BaseModel):
    name: str = Field(description="Nome della nota di cui elencare i link in uscita")

@tool(LinksArgs, cls="atomic")
def silica_links(name: str) -> list:
    """Elenca i link in uscita da una nota (note collegate)."""
    refs = DRIVER.links(name)
    return [{"name": r.name, "path": r.path} for r in refs]


class BacklinksArgs(BaseModel):
    name: str = Field(description="Nome della nota di cui elencare i backlink")

@tool(BacklinksArgs, cls="atomic")
def silica_backlinks(name: str) -> list:
    """Elenca i backlink (link in entrata) verso una nota."""
    refs = DRIVER.backlinks(name)
    return [{"name": r.name, "path": r.path} for r in refs]


class EmptyArgs(BaseModel):
    pass

@tool(EmptyArgs, cls="atomic")
def silica_orphans() -> list:
    """Elenca le note orfane (senza link in entrata) nel vault."""
    refs = DRIVER.orphans()
    return [{"name": r.name, "path": r.path} for r in refs]


@tool(EmptyArgs, cls="atomic")
def silica_unresolved() -> list:
    """Elenca i wikilink irrisolti nel vault (link che puntano a note inesistenti)."""
    links = DRIVER.unresolved()
    return [{"target": l.target} for l in links]


# ---------------------------------------------------------------------------
# List files
# ---------------------------------------------------------------------------

class ListFilesArgs(BaseModel):
    folder: str = Field(default="", description="Cartella opzionale per filtrare i risultati")

@tool(ListFilesArgs, cls="atomic")
def silica_files(folder: str = "") -> list:
    """Elenca tutti i file markdown nel vault, opzionalmente filtrati per cartella."""
    refs = DRIVER.list_files(folder)
    return [{"name": r.name, "path": r.path} for r in refs]
