# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codegraph — derived structural code index (spec-code-lane, ADR-0018).

Structural ≠ semantic: this index lives BESIDE the semantic legs (embeddings,
co-occurrence), never inside them. Import edges never enter related_notes/RRF
fusion — an import hub (paths.py, imported everywhere) is semantically
peripheral and would flood the ranking (import-linter contract in pyproject).

The store is derived: rebuildable, never repaired, never a source of truth.
Refresh happens only on invocation (no watchers, per charter).

Call edges are OUT of v1 — containment + imports are 100% deterministic,
calls in dynamic languages are not. Future seam: a scope-stack heuristic
emitting edges marked `approximate: true`, excluded from every automatic
decision (autolink, coverage ordering, /impact).
"""
from __future__ import annotations

import posixpath
from dataclasses import dataclass, field as _field
from pathlib import Path

import orjson

from silica.kernel import codeast, gitstate
from silica.kernel import paths as _paths

_TS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_TS_ALIAS_PREFIXES = ("@/", "~/")
# ponytail: no tsconfig.paths parsing in v1; add it if a real TS repo makes unresolved noisy


def package_of(module: str, root: Path) -> str:
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


def is_first_party(module: str, root: Path) -> bool:
    if module.startswith("."):  # python relative / TS "./x" "../x"
        return True
    top = module.split(".")[0].split("/")[0]
    return (root / top).is_dir() or (root / f"{top}.py").is_file()


def _py_candidates(parts: list[str]) -> list[str]:
    """Candidate repo-relative paths for a dotted module, deepest first.
    The last segment may be a `from X import y` name, so after trying the
    full path we back off one segment (module-vs-__init__ rule, spec §1)."""
    out: list[str] = []
    if parts:
        stem = "/".join(parts)
        out += [f"{stem}.py", f"{stem}/__init__.py"]
    if len(parts) > 1:
        stem = "/".join(parts[:-1])
        out += [f"{stem}.py", f"{stem}/__init__.py"]
    return out


def _resolve_python(module: str, importer: str, files: set[str]) -> str | None:
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        rest = [p for p in module[dots:].split(".") if p]
        base = posixpath.dirname(importer)
        for _ in range(dots - 1):
            base = posixpath.dirname(base)
        prefix = [p for p in base.split("/") if p]
        candidates = _py_candidates(prefix + rest)
    else:
        candidates = _py_candidates([p for p in module.split(".") if p])
    for cand in candidates:
        if cand in files:
            return cand
    return None


def _resolve_ts(module: str, importer: str, files: set[str]) -> str | None:
    base = posixpath.normpath(posixpath.join(posixpath.dirname(importer), module))
    candidates = [base] if base.lower().endswith(_TS_EXTS) else []
    candidates += [f"{base}{ext}" for ext in _TS_EXTS]
    candidates += [f"{base}/index{ext}" for ext in _TS_EXTS]
    for cand in candidates:
        if cand in files:
            return cand
    return None


def classify_import(
    module: str, importer: str, files: set[str], language: str, root: Path
) -> tuple[str, str]:
    """Classify one import string → ("resolved", path) | ("external", top)
    | ("unresolved", module). A resolved path is always a member of `files`
    — never an edge to a nonexistent file (spec §1). Unresolvable first-party
    imports land in "unresolved", counted in the report, never dropped."""
    if language == "python":
        resolved = _resolve_python(module, importer, files)
        if resolved:
            return ("resolved", resolved)
        if is_first_party(module, root):
            return ("unresolved", module)
        return ("external", module.split(".")[0])
    # TS/JS
    if module.startswith(("./", "../")) or module in (".", ".."):
        resolved = _resolve_ts(module, importer, files)
        return ("resolved", resolved) if resolved else ("unresolved", module)
    if module.startswith(_TS_ALIAS_PREFIXES):
        return ("unresolved", module)  # alias-like: first-party, not external (spec §1)
    return ("external", module.split("/")[0])


# ---------------------------------------------------------------------------
# store — derived index at paths.index_dir()/codegraph.json
# ---------------------------------------------------------------------------

STORE_VERSION = 1


def store_path() -> Path:
    return _paths.index_dir() / "codegraph.json"


@dataclass
class CodeGraph:
    head_ref: str
    files: dict[str, dict] = _field(default_factory=dict)

    def importers(self, path: str) -> list[str]:
        return sorted(p for p, e in self.files.items() if path in e.get("imports", []))

    def fan_in(self, path: str) -> int:
        return sum(1 for e in self.files.values() if path in e.get("imports", []))


def supported_files(root: Path) -> list[str]:
    """Sorted repo-relative supported files, git-listed (tracked + untracked
    non-ignored), existing on disk. Empty when git is unavailable."""
    listed = gitstate.list_files(root)
    if listed is None:
        return []
    return sorted(
        rel for rel in listed
        if (codeast.language_for(rel) is not None or rel.lower().endswith(".ipynb"))
        and (root / rel).is_file()
    )


def _file_entry(root: Path, rel: str, files: set[str]) -> dict:
    language = codeast.language_for(rel)
    try:
        source = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"language": language, "imports": [], "external": [],
                "unresolved": [], "symbols": [], "parse_error": True}
    if rel.lower().endswith(".ipynb"):
        from silica.kernel import ipynb
        try:
            cells = ipynb.parse_cells(source)
        except ValueError:
            return {"language": None, "imports": [], "external": [],
                    "unresolved": [], "symbols": [], "parse_error": True}
        language = ipynb.CODEAST_LANGUAGE.get(cells.language)
        if language is None:  # e.g. an R kernel: node exists, no structure
            return {"language": cells.language, "imports": [], "external": [],
                    "unresolved": [], "symbols": [], "parse_error": False}
        sk = codeast.extract_skeleton(cells.code, language, path=rel)
    else:
        sk = codeast.extract_skeleton(source, language, path=rel)
    imports: list[str] = []
    external: list[str] = []
    unresolved: list[str] = []
    for mod in dict.fromkeys(sk.imports):
        if not mod:
            continue
        kind, value = classify_import(mod, rel, files, language, root)
        bucket = {"resolved": imports, "external": external, "unresolved": unresolved}[kind]
        if value not in bucket:
            bucket.append(value)
    return {
        "language": language,
        "imports": imports,
        "external": external,
        "unresolved": unresolved,
        "symbols": [
            {"kind": s.kind, "name": s.name, "parent": s.parent,
             "signature": s.signature, "doc": s.doc}
            for s in sk.symbols
        ],
        "parse_error": sk.parse_error,
    }


def build_codegraph(root: Path) -> CodeGraph:
    """Full rebuild — the only write path. The index is never repaired,
    only recomputed (spec: Decisioni.2).
    # ponytail: full rebuild (~ms/file); incremental per-file if a real repo makes it slow
    """
    current = supported_files(root)
    files = set(current)
    entries = {rel: _file_entry(root, rel, files) for rel in current}
    return CodeGraph(head_ref=gitstate.head_ref(root) or "", files=entries)


def _serialize(graph: CodeGraph) -> bytes:
    # OPT_SORT_KEYS → byte-for-byte deterministic for the same repo state
    # (symbols stay lists in document order; sorting only touches map keys).
    return orjson.dumps(
        {"version": STORE_VERSION, "head_ref": graph.head_ref, "files": graph.files},
        option=orjson.OPT_SORT_KEYS,
    )


def _still_valid(data: dict, root: Path, current: list[str], sp: Path) -> bool:
    """Validity key (spec §1): head_ref unchanged AND file set identical AND
    no supported file newer than the store (mtime alone misses adds/deletes;
    the set comparison catches them — same walk, same stat pass)."""
    if data.get("head_ref", "") != (gitstate.head_ref(root) or ""):
        return False
    if set(data.get("files", {}).keys()) != set(current):
        return False
    try:
        store_mtime = sp.stat().st_mtime
        return all((root / rel).stat().st_mtime <= store_mtime for rel in current)
    except OSError:
        return False


def load_codegraph(vault: Path | str) -> CodeGraph | None:
    """Valid store, or transparent full rebuild + save. None when the vault
    is not inside a git repo — the index is disabled and consumers report
    "no repo", degrading soft (never an error in place of a poorer result)."""
    root = gitstate.find_repo_root(Path(vault))
    if root is None:
        return None
    sp = store_path()
    current = supported_files(root)
    if sp.exists():
        try:
            data = orjson.loads(sp.read_bytes())
            if data.get("version") == STORE_VERSION and _still_valid(data, root, current, sp):
                return CodeGraph(head_ref=data.get("head_ref", ""), files=data.get("files", {}))
        except Exception:
            _paths.quarantine(sp)  # corrupt derived store: aside for doctor, then rebuild
    graph = build_codegraph(root)
    _paths.atomic_write_bytes(sp, _serialize(graph))
    return graph


def code_vocabulary(graph: CodeGraph, cap: int = 30) -> list[str]:
    """Canonical code spellings for the 'Vault vocabulary' substrate section:
    module stems + public symbol names from the top-`cap` files by fan-in.
    Names only, never edges — the vocabulary channel is one of the two
    sanctioned structural→semantic contact points (spec §4a). The effect:
    the distiller reuses the canonical grafia (InjectorFSM, not injector-fsm)
    so co-occurrence latches onto it."""
    from collections import Counter

    fan: Counter[str] = Counter()
    for entry in graph.files.values():
        for target in entry.get("imports", []):
            fan[target] += 1
    top = sorted(graph.files.keys(), key=lambda p: (-fan[p], p))[:cap]
    names: list[str] = []
    for p in top:
        stem = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if stem and stem != "__init__":
            names.append(stem)
        for s in graph.files[p].get("symbols", []):
            n = s.get("name", "")
            if n and not n.startswith("_"):
                names.append(n)
    return list(dict.fromkeys(names))
