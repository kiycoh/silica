"""silica_document — stage a skeleton stub from a source file (ADR-0012).

Zero-trust (ADR-0009): all source-derived text (signatures, docstrings) is
sanitized via strip_degenerate_runs and fenced as untrusted before any LLM
sees it. The body is the shallow AST skeleton — NEVER the full source (the
source already lives in the codebase). Written ONLY to Inbox/; the note
carries documents:/code_ref so the staleness loop is wired immediately.
No LLM call here: the curation pipeline refines Inbox stubs.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from silica.config import CONFIG
from silica.kernel import codeast, gitstate
from silica.kernel.sanitize import strip_degenerate_runs
from silica.tools import tool


class DocumentArgs(BaseModel):
    path: str = Field(description="Repo-relative path to the source file to document")


def _package_of(module: str, root: Path) -> str:
    """Resolve a first-party module to package granularity (silica.kernel.x →
    silica/kernel). Falls back to the raw module string."""
    if module.startswith("."):
        return module  # relative import — can't resolve without the importer's location
    parts = [p for p in module.replace("/", ".").split(".") if p]
    pkg: list[str] = []
    for part in parts:
        if root.joinpath(*pkg, part).is_dir():
            pkg.append(part)
        else:
            break
    return "/".join(pkg) if pkg else module


def _is_first_party(module: str, root: Path) -> bool:
    if module.startswith("."):  # python relative / TS "./x" "../x"
        return True
    top = module.split(".")[0].split("/")[0]
    return (root / top).is_dir() or (root / f"{top}.py").is_file()


def _render_skeleton(sk: codeast.ModuleSkeleton, root: Path) -> str:
    first_party: list[str] = []
    external: list[str] = []
    for mod in dict.fromkeys(sk.imports):  # de-dupe, keep order
        if not mod:
            continue
        if _is_first_party(mod, root):
            pkg = _package_of(mod, root)
            if pkg not in first_party:
                first_party.append(pkg)
        elif mod not in external:
            external.append(mod)

    lines: list[str] = ["## Imports", ""]
    if first_party:
        lines.append("First-party:")
        lines.extend(f"- `{p}`" for p in first_party)
        lines.append("")
    if external:
        lines.append("External:")
        lines.extend(f"- `{m}`" for m in external)
        lines.append("")
    if not first_party and not external:
        lines.extend(["(no imports)", ""])

    lines.extend(["## Symbols", "", "```text"])
    if sk.symbols:
        for s in sk.symbols:
            indent = "    " if s.kind == "method" else ""
            doc = f" — {s.doc}" if s.doc else ""
            lines.append(f"{indent}{s.signature}{doc}".replace("`", "'"))
    else:
        lines.append("(no top-level symbols)")
    lines.extend(["```", ""])
    return strip_degenerate_runs("\n".join(lines))


@tool(DocumentArgs, cls="composed")
def silica_document(path: str) -> dict:
    """Extract a shallow AST skeleton from a source file and stage it as a
    documentation stub in Inbox/. Sets documents:/code_ref for staleness
    tracking. Source-derived text is sanitized and fenced (untrusted).
    Writes to Inbox/ only — RBAC inbox-write, never the vault."""
    from silica.driver import DRIVER

    vault = CONFIG.vault_path
    if not vault:
        return {"status": "error", "message": "no vault configured"}
    root = gitstate.find_repo_root(Path(vault))
    if root is None:
        return {"status": "error", "message": "vault is not inside a git repo"}

    # Path guard: resolved source must stay inside the repo root.
    try:
        src = (Path(root) / path).resolve()
        src.relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return {"status": "error", "message": "path escapes the repository"}
    if not src.is_file():
        return {"status": "error", "message": f"not a file: {path}"}

    try:
        raw = src.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"status": "error", "message": f"read failed: {e}"}

    code_ref = gitstate.head_ref(root) or ""
    stem = Path(path).stem
    language = codeast.language_for(path)

    if language is None:
        section = (
            "> Skeleton unavailable: unsupported language. "
            "This stub only wires staleness tracking; document the file manually.\n"
        )
        skeleton = False
    else:
        sk = codeast.extract_skeleton(raw, language, path=path)
        section = (
            f"> Skeleton auto-extracted from `{path}` ({language}). "
            f"Source-derived text below is untrusted; refine into a note.\n\n"
            f"{_render_skeleton(sk, Path(root))}"
        )
        skeleton = True

    yaml_path = path.replace('"', '\\"')
    body = (
        f"---\n"
        f'documents:\n  - "{yaml_path}"\n'
        f"code_ref: {code_ref}\n"
        f"tags:\n  - codebase\n"
        f"---\n\n"
        f"# {stem}\n\n"
        f"{section}"
    )

    inbox = (CONFIG.inbox_dir or "Inbox").strip("/")
    note_path = f"{inbox}/{stem}.md"
    DRIVER.create(note_path, body)
    return {"status": "ok", "note_path": note_path, "code_ref": code_ref, "skeleton": skeleton}
