# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codeast.base — dataclasses, extension map, dispatch, shared helpers.

Deterministic, in-process, LLM-free. Extracts ONLY the skeleton: imports,
classes, function/method signatures, first docstring line. The parsimony
line is hard: no call-graph, no scope resolution, no MRO — that is deep
structural machinery (see ADR-0011's dormant external seam).

Grammars come from tree-sitter-language-pack (one package, no postinstall
compilation). Language detection is extension-based, but
limited to languages this extractor actually supports.

NOTE: tree-sitter >= 0.23 uses a method-call API — node.kind(), node.start_byte(),
etc. are methods, not properties. All internal helpers use that calling convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_CALL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".java": "java",
    ".c": "c",
    ".h": "cpp",   # pragmatic superset for skeleton purposes
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
}

# Presence-only languages: files enter the graph (co-change edges, folder
# structure) with zero extraction. "No structure" is true, not a failure.
BARE_LANGUAGES: frozenset[str] = frozenset({"toml", "html", "css"})


@dataclass(frozen=True)
class Symbol:
    kind: str        # "class" | "function" | "method"
    name: str
    signature: str   # declaration line, whitespace-collapsed
    doc: str = ""    # first docstring line ("" when absent)
    parent: str = "" # enclosing class name for methods
    doc_full: str = ""  # whole docstring, per-line stripped ("" when absent)
    decorators: list[str] = field(default_factory=list)  # names, '@' and call args stripped


@dataclass(frozen=True)
class Call:
    name: str    # called name as written, dotted allowed ("x.f")
    parent: str  # enclosing top-level symbol ("" at module level)


@dataclass(frozen=True)
class ModuleSkeleton:
    path: str                  # repo-relative source path
    language: str              # EXTENSION_MAP value
    imports: list[str] = field(default_factory=list)   # module strings, duplicates possible
    symbols: list[Symbol] = field(default_factory=list)  # document order
    parse_error: bool = False  # tree-sitter setup failed — consumers must not read "empty" as "no structure"
    module_doc: str = ""                                   # module-level docstring, whole
    module_comments: list[str] = field(default_factory=list)  # top-level comment blocks
    dunder_all: list[str] | None = None  # literal __all__, or None (absent / dynamic)
    calls: list[Call] = field(default_factory=list)  # call sites, deduped by (name, parent)
    import_aliases: dict[str, str] = field(default_factory=dict)  # alias -> real dotted name
    has_main_guard: bool = False  # `if __name__ == "__main__"` present


def language_for(path: str | Path) -> str | None:
    """Map a file path to a supported language, or None."""
    return EXTENSION_MAP.get(Path(path).suffix.lower())


def extract_skeleton(source: str, language: str, path: str = "") -> ModuleSkeleton:
    """Parse `source` and return its shallow skeleton. Never raises: any
    parser failure degrades to an empty skeleton (tree-sitter itself is
    error-tolerant, so partial sources still yield partial skeletons)."""
    if language in BARE_LANGUAGES:
        return ModuleSkeleton(path=path, language=language)
    try:
        from tree_sitter_language_pack import get_parser
        tree = get_parser(language).parse_bytes(source.encode("utf-8"))
    except Exception:
        return ModuleSkeleton(path=path, language=language, parse_error=True)

    src = source.encode("utf-8")
    imports: list[str] = []
    symbols: list[Symbol] = []
    root = tree.root_node()
    module_doc, module_comments, dunder_all = ("", [], None)
    calls: list[Call] = []
    aliases: dict[str, str] = {}
    has_main_guard = False
    if language == "java":
        from silica.kernel.codeast import java as _java
        return _java.extract(root, src, path=path, language=language)
    if language in ("c", "cpp"):
        from silica.kernel.codeast import c as _c
        return _c.extract(root, src, path=path, language=language)
    if language == "python":
        from silica.kernel.codeast import python as _py
        for i in range(root.named_child_count()):
            _py._py_extract(root.named_child(i), src, imports, symbols, aliases=aliases)
        module_doc, module_comments = _py._py_module_docs(root, src)
        dunder_all = _py._py_dunder_all(root, src)
        calls = _py._py_calls(root, src)
        has_main_guard = _py._py_has_main_guard(root, src)
    else:
        # ponytail: TS doc/comment/call capture deferred with the rest of the TS lane
        from silica.kernel.codeast import ts as _ts
        for i in range(root.named_child_count()):
            _ts._ts_extract(root.named_child(i), src, imports, symbols)
    return ModuleSkeleton(path=path, language=language, imports=imports,
                          symbols=symbols, module_doc=module_doc,
                          module_comments=module_comments, dunder_all=dunder_all,
                          calls=calls, import_aliases=aliases,
                          has_main_guard=has_main_guard)


# ---------------------------------------------------------------------------
# helpers shared by the per-language walkers
# ---------------------------------------------------------------------------

def _text(node, src: bytes) -> str:
    return src[node.start_byte():node.end_byte()].decode("utf-8", errors="replace")


def _signature(node, src: bytes) -> str:
    """Declaration text up to (excluding) the body, whitespace-collapsed."""
    body = node.child_by_field_name("body")
    end = body.start_byte() if body is not None else node.end_byte()
    sig = src[node.start_byte():end].decode("utf-8", errors="replace")
    return " ".join(sig.split()).rstrip(":")


def _block_comment_text(node, src: bytes) -> str:
    """Block comment body: /** */ (or /*! */) delimiters and per-line '*'
    gutters stripped — shared by the Java and C/C++ doc-comment paths."""
    text = _text(node, src).strip()
    if text.startswith("/*"):
        text = text[2:]
        if text.endswith("*/"):
            text = text[:-2]
        text = text.lstrip("*!")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("*"):
            line = line[1:].strip()
        lines.append(line)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# structural diff (COSMETIC vs STRUCTURAL, spec-code-lane §2)
# ---------------------------------------------------------------------------

def diff_skeletons(old: ModuleSkeleton, new: ModuleSkeleton) -> list[str]:
    """Structural differences old→new, one human-readable line each; empty
    list = same shape. Compares import sets, symbol sets (kind, name, parent)
    and whitespace-collapsed signatures — the COSMETIC/STRUCTURAL verdict
    for git-native staleness classification."""
    out: list[str] = []
    old_imp, new_imp = set(old.imports), set(new.imports)
    out.extend(f"+ import {m}" for m in sorted(new_imp - old_imp))
    out.extend(f"- import {m}" for m in sorted(old_imp - new_imp))

    def _key(s: Symbol) -> tuple[str, str, str]:
        return (s.kind, s.name, s.parent)

    def _label(k: tuple[str, str, str]) -> str:
        kind, name, parent = k
        return f"{kind} {parent + '.' if parent else ''}{name}"

    def _qual(k: tuple[str, str, str]) -> str:
        _, name, parent = k
        return f"{parent + '.' if parent else ''}{name}"

    old_syms = {_key(s): s for s in old.symbols}
    new_syms = {_key(s): s for s in new.symbols}
    out.extend(f"+ {_label(k)}" for k in sorted(new_syms.keys() - old_syms.keys()))
    out.extend(f"- {_label(k)}" for k in sorted(old_syms.keys() - new_syms.keys()))
    for k in sorted(old_syms.keys() & new_syms.keys()):
        if old_syms[k].signature != new_syms[k].signature:
            out.append(f"signature changed: {_qual(k)}")
    return out
