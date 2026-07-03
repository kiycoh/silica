"""silica_document — stage a skeleton stub from a source file (ADR-0012).

Thin agent-facing wrapper over the code SourceAdapter (ADR-0014): guards,
sanitization and stub assembly live in silica/sources/code.py. Writes ONLY
to Inbox/ — RBAC inbox-write, never the vault. No LLM call here: the
curation pipeline refines Inbox stubs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from silica.tools import tool


class DocumentArgs(BaseModel):
    path: str = Field(description="Repo-relative path to the source file to document")


@tool(DocumentArgs, cls="composed")
def silica_document(path: str) -> dict:
    """Extract a shallow AST skeleton from a source code file and stage it as a
    documentation stub in Inbox/ (never directly in the vault). Sets
    documents:/code_ref frontmatter for staleness tracking; source-derived text
    is sanitized and fenced. Ingest the stub afterwards with silica_run_injector."""
    from silica.driver import DRIVER
    from silica.sources.code import CODE

    try:
        item = CODE.read(path)
        stub = CODE.to_stub(item)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    DRIVER.create(stub.note_path, stub.body)
    return {
        "status": "ok",
        "note_path": stub.note_path,
        "code_ref": item.meta.get("code_ref", ""),
        "skeleton": item.meta.get("language") is not None,
    }
