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
from pathlib import Path

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
