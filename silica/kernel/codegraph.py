# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codegraph — derived structural code index (spec-code-lane, ADR-0018).

Structural ≠ semantic: this index lives BESIDE the semantic legs (embeddings,
co-occurrence), never inside them. Import edges never enter related_notes/RRF
fusion — an import hub (paths.py, imported everywhere) is semantically
peripheral and would flood the ranking (import-linter contract in pyproject).

The store is derived: rebuildable, never repaired, never a source of truth.
Refresh happens only on invocation (no watchers, per charter).

Import-scoped call edges (store v2) are recorded for the code-wiki digest:
a call whose spelled name matches an imported first-party name is a
near-certain usage edge, no scope resolution needed. They stay OUT of
related_notes/RRF/autolink and every automatic decision (coverage ordering,
/impact) — the wiki digest is their only consumer. Broader call resolution
(scope-stack, receivers) remains a future seam.
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
    if language == "java":
        if module.endswith(".*"):
            return ("unresolved", module)  # wildcard: no single target
        # suffix match absorbs src/main/java/ prefixes with no configuration
        suffix = "/" + module.replace(".", "/") + ".java"
        matches = [f for f in files if f.endswith(suffix) or f == suffix[1:]]
        if matches:
            return ("resolved", min(matches, key=lambda p: (len(p), p)))
        top = module.split(".", 1)[0]
        if (root / top).is_dir() or any(f.startswith(top + "/") or f"/{top}/" in f
                                        for f in files):
            return ("unresolved", module)  # first-party package tree, no file
        return ("external", ".".join(module.split(".")[:2]))
    if language in ("c", "cpp"):
        text = module.strip()
        if text.startswith("<"):
            return ("external", text.strip("<>").strip())
        inc = text.strip('"')
        # quoted include: importer-dir-relative, then root-relative, then suffix
        cand = posixpath.normpath(posixpath.join(posixpath.dirname(importer), inc))
        if cand in files:
            return ("resolved", cand)
        if inc in files:
            return ("resolved", inc)
        matches = [f for f in files if f.endswith("/" + inc)]
        if matches:
            return ("resolved", min(matches, key=lambda p: (len(p), p)))
        return ("unresolved", inc)
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

STORE_VERSION = 2


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

    def call_edges(self) -> list[tuple[str, str, str, str]]:
        """Sorted (source_file, target_file, callee, caller) across the graph."""
        out: list[tuple[str, str, str, str]] = []
        for src_path, entry in self.files.items():
            for e in entry.get("calls", []):
                out.append((src_path, e["target"], e["callee"], e["caller"]))
        return sorted(out)


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


def _resolve_calls(sk, rel: str, files: set[str], root: Path,
                   language: str = "python") -> list[dict]:
    """Import-scoped call edges: a call whose spelled name matches an imported
    first-party name is a near-certain usage edge. No scope resolution, no MRO,
    no receivers: only the subset of the call graph that needs no inference.
    Local shadowing can produce rare false positives: accepted, these edges
    ground LLM prose only and never enter automatic decisions (module docstring).
    # ponytail: import-scoped only; scope-stack/receiver if flows read wrong
    """
    imports = [m for m in dict.fromkeys(sk.imports) if m]
    by_len = sorted(imports, key=len, reverse=True)
    edges: dict[tuple[str, str, str], None] = {}
    for call in sk.calls:
        name = call.name
        head = name.split(".", 1)[0]
        alias = sk.import_aliases.get(head)
        if alias:
            name = alias + name[len(head):]
        target = callee = None
        if "." in name:
            # break only on a first-party resolution: an external/unresolved
            # prefix match (e.g. a bare `import yamlmod` shadowing
            # `from app.adapters import yamlmod`) must not eat the edge
            for mod in by_len:
                if name == mod or name.startswith(mod + "."):
                    kind, value = classify_import(mod, rel, files, language, root)
                    if kind == "resolved":
                        target = value
                        rest = name[len(mod):].lstrip(".")
                        callee = rest or mod.rsplit(".", 1)[-1]
                        break
            if target is None:
                # `from pkg import mod; mod.f()` — head equals an import's last segment
                dotted_head, _, dotted_rest = name.partition(".")
                for mod in imports:
                    if "." in mod and mod.rsplit(".", 1)[-1] == dotted_head:
                        kind, value = classify_import(mod, rel, files, language, root)
                        if kind == "resolved":
                            target, callee = value, dotted_rest
                            break
        else:
            for mod in imports:
                if "." in mod and mod.rsplit(".", 1)[-1] == name:
                    kind, value = classify_import(mod, rel, files, language, root)
                    if kind == "resolved":
                        target, callee = value, name
                        break
        if target and target != rel:
            edges[(target, callee or "", call.parent)] = None
    return [{"target": t, "callee": ce, "caller": ca} for (t, ce, ca) in sorted(edges)]


_EMPTY_V2 = {"module_doc": "", "module_comments": [], "dunder_all": None,
             "has_main_guard": False, "calls": []}


def _file_entry(root: Path, rel: str, files: set[str]) -> tuple[dict, list[tuple[str, str]]]:
    """One store entry, plus the raw (name, parent) call sites for C/C++ —
    those resolve later in build_codegraph's include join, once every file's
    symbols exist."""
    language = codeast.language_for(rel)
    try:
        source = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"language": language, "imports": [], "external": [],
                "unresolved": [], "symbols": [], "parse_error": True, **_EMPTY_V2}, []
    if rel.lower().endswith(".ipynb"):
        from silica.kernel import ipynb
        try:
            cells = ipynb.parse_cells(source)
        except ValueError:
            return {"language": None, "imports": [], "external": [],
                    "unresolved": [], "symbols": [], "parse_error": True, **_EMPTY_V2}, []
        language = ipynb.CODEAST_LANGUAGE.get(cells.language)
        if language is None:  # e.g. an R kernel: node exists, no structure
            return {"language": cells.language, "imports": [], "external": [],
                    "unresolved": [], "symbols": [], "parse_error": False, **_EMPTY_V2}, []
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
    entry = {
        "language": language,
        "imports": imports,
        "external": external,
        "unresolved": unresolved,
        "symbols": [
            {"kind": s.kind, "name": s.name, "parent": s.parent,
             "signature": s.signature, "doc": s.doc, "doc_full": s.doc_full,
             "decorators": s.decorators}
            for s in sk.symbols
        ],
        "module_doc": sk.module_doc,
        "module_comments": sk.module_comments,
        "dunder_all": sk.dunder_all,
        "has_main_guard": sk.has_main_guard,
        # ponytail: TS call edges empty in v1, matches codeast TS deferral;
        # C/C++ edges come from the graph-level include join in build_codegraph
        "calls": (_resolve_calls(sk, rel, files, root, language)
                  if language in ("python", "java") else []),
        "parse_error": sk.parse_error,
    }
    raw_calls = ([(c.name, c.parent) for c in sk.calls]
                 if language in ("c", "cpp") else [])
    return entry, raw_calls


def _join_c_calls(rel: str, raw: list[tuple[str, str]], entries: dict[str, dict]) -> list[dict]:
    """C/C++ call edges: includes carry no names, so the import-scoped matcher
    cannot attach calls. Instead, an edge exists when a spelled callee is
    among the symbols of a directly included, resolved file.
    # ponytail: direct includes only, no transitivity; deepen if real repos read thin
    """
    edges: dict[tuple[str, str, str], None] = {}
    for target in entries[rel].get("imports", []):
        names = {s["name"] for s in entries.get(target, {}).get("symbols", [])}
        for name, parent in raw:
            callee = name.rsplit(".", 1)[-1]
            if callee in names and target != rel:
                edges[(target, callee, parent)] = None
    return [{"target": t, "callee": c, "caller": p} for (t, c, p) in sorted(edges)]


def build_codegraph(root: Path) -> CodeGraph:
    """Full rebuild — the only write path. The index is never repaired,
    only recomputed (spec: Decisioni.2).
    # ponytail: full rebuild (~ms/file); incremental per-file if a real repo makes it slow
    """
    current = supported_files(root)
    files = set(current)
    entries: dict[str, dict] = {}
    c_raw: dict[str, list[tuple[str, str]]] = {}
    for rel in current:
        entry, raw_calls = _file_entry(root, rel, files)
        entries[rel] = entry
        if raw_calls:
            c_raw[rel] = raw_calls
    for rel, raw in c_raw.items():
        entries[rel]["calls"] = _join_c_calls(rel, raw, entries)
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
    root = _paths.repo_root_for(vault)
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


@dataclass(frozen=True)
class ImpactEntry:
    path: str
    change_level: str
    details: list[str]
    fan_in: int
    notes: list[str]            # notes documenting `path`
    neighbor_notes: list[str]   # notes documenting 1-hop import neighbors


def compute_impact(vault: Path | str, range_spec: str | None = None) -> list[ImpactEntry] | None:
    """Changed supported files → change_level + documenting notes + 1-hop
    import-neighbor notes. None when the vault is not in a git repo (the
    consumer reports "no repo"). Zero LLM; sorted (structural, fan-in desc)."""
    from silica.kernel import codedocs

    root = _paths.repo_root_for(vault)
    if root is None:
        return None
    changed = gitstate.changed_paths(root, range_spec) or []
    graph = load_codegraph(vault)
    docmap: dict[str, list[str]] = {}
    for note_path, data, _ in codedocs.iter_documenting_notes(vault):
        for p in codedocs.documents_of(data):
            docmap.setdefault(p, []).append(note_path)

    if range_spec and ".." in range_spec:
        base_ref, _, new_ref = range_spec.partition("..")
        new_ref = new_ref.lstrip(".") or None   # tolerate A...B
    else:
        base_ref, new_ref = (range_spec or gitstate.head_ref(root) or ""), None

    entries: list[ImpactEntry] = []
    for path in changed:
        if codeast.language_for(path) is None and not path.lower().endswith(".ipynb"):
            continue  # non-code files are outside the code lane
        level, details = codedocs.classify_change(root, base_ref, path, new_ref=new_ref)
        neighbors: set[str] = set()
        fan = 0
        if graph is not None:
            entry = graph.files.get(path, {})
            neighbors = set(entry.get("imports", [])) | set(graph.importers(path))
            fan = graph.fan_in(path)
        neighbor_notes = sorted({n for nb in neighbors for n in docmap.get(nb, [])})
        entries.append(ImpactEntry(
            path=path, change_level=level, details=details, fan_in=fan,
            notes=sorted(docmap.get(path, [])), neighbor_notes=neighbor_notes,
        ))
    entries.sort(key=lambda e: (0 if e.change_level == "structural" else 1, -e.fan_in, e.path))
    return entries
