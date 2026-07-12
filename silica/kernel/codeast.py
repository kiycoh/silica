# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codeast — native shallow AST skeleton extraction (ADR-0012).

Deterministic, in-process, LLM-free. Extracts ONLY the skeleton: imports,
classes, function/method signatures, first docstring line. The parsimony
line is hard: no call-graph, no scope resolution, no MRO — that is deep
structural machinery (see ADR-0011's dormant external seam).

Grammars come from tree-sitter-language-pack (one package, no postinstall
compilation). Language detection is extension-based, GitNexus-style, but
limited to languages this extractor actually supports.

NOTE: tree-sitter >= 0.23 uses a method-call API — node.kind(), node.start_byte(),
etc. are methods, not properties. All internal helpers use that calling convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}


@dataclass(frozen=True)
class Symbol:
    kind: str        # "class" | "function" | "method"
    name: str
    signature: str   # declaration line, whitespace-collapsed
    doc: str = ""    # first docstring line ("" when absent)
    parent: str = "" # enclosing class name for methods
    doc_full: str = ""  # whole docstring, per-line stripped ("" when absent)


@dataclass(frozen=True)
class ModuleSkeleton:
    path: str                  # repo-relative source path
    language: str              # EXTENSION_MAP value
    imports: list[str] = field(default_factory=list)   # module strings, duplicates possible
    symbols: list[Symbol] = field(default_factory=list)  # document order
    parse_error: bool = False  # tree-sitter setup failed — consumers must not read "empty" as "no structure"
    module_doc: str = ""                                   # module-level docstring, whole
    module_comments: list[str] = field(default_factory=list)  # top-level comment blocks


def language_for(path: str | Path) -> str | None:
    """Map a file path to a supported language, or None."""
    return EXTENSION_MAP.get(Path(path).suffix.lower())


def extract_skeleton(source: str, language: str, path: str = "") -> ModuleSkeleton:
    """Parse `source` and return its shallow skeleton. Never raises: any
    parser failure degrades to an empty skeleton (tree-sitter itself is
    error-tolerant, so partial sources still yield partial skeletons)."""
    try:
        from tree_sitter_language_pack import get_parser
        tree = get_parser(language).parse_bytes(source.encode("utf-8"))
    except Exception:
        return ModuleSkeleton(path=path, language=language, parse_error=True)

    src = source.encode("utf-8")
    imports: list[str] = []
    symbols: list[Symbol] = []
    root = tree.root_node()
    extract = _py_extract if language == "python" else _ts_extract
    for i in range(root.named_child_count()):
        extract(root.named_child(i), src, imports, symbols)
    module_doc, module_comments = ("", [])
    if language == "python":
        module_doc, module_comments = _py_module_docs(root, src)
    # ponytail: TS doc/comment capture deferred; Python-first wiki
    return ModuleSkeleton(path=path, language=language, imports=imports,
                          symbols=symbols, module_doc=module_doc,
                          module_comments=module_comments)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _text(node, src: bytes) -> str:
    return src[node.start_byte():node.end_byte()].decode("utf-8", errors="replace")


def _signature(node, src: bytes) -> str:
    """Declaration text up to (excluding) the body, whitespace-collapsed."""
    body = node.child_by_field_name("body")
    end = body.start_byte() if body is not None else node.end_byte()
    sig = src[node.start_byte():end].decode("utf-8", errors="replace")
    return " ".join(sig.split()).rstrip(":")


def _py_doc_node(node):
    """Bare string node of a body's leading docstring, or None."""
    body = node.child_by_field_name("body")
    if body is None or body.named_child_count() == 0:
        return None
    first = body.named_child(0)
    # In tree-sitter >= 0.23 the docstring is a bare 'string' node as first
    # child of the block; older grammars wrap it in expression_statement.
    if first.kind() == "expression_statement" and first.named_child_count() > 0:
        first = first.named_child(0)
    return first if first.kind() == "string" else None


def _strip_quotes(text: str) -> str:
    for q in ('"""', "'''", '"', "'"):
        if text.startswith(q) and text.endswith(q) and len(text) >= 2 * len(q):
            return text[len(q):-len(q)]
    return text


def _doc_text(string_node, src: bytes) -> str:
    """Whole docstring, quotes stripped, each line stripped, blank edges gone."""
    text = _strip_quotes(_text(string_node, src).strip())
    return "\n".join(line.strip() for line in text.strip().splitlines()).strip()


def _py_docstring_full(node, src: bytes) -> str:
    doc = _py_doc_node(node)
    return _doc_text(doc, src) if doc is not None else ""


def _py_docstring(node, src: bytes) -> str:
    """First line of the body's leading docstring, quotes stripped."""
    full = _py_docstring_full(node, src)
    return full.splitlines()[0].strip() if full else ""


_COMMENT_CAP_LINES = 40  # per file: keeps the digest bounded


def _py_module_docs(root, src: bytes) -> tuple[str, list[str]]:
    """Module-level docstring (whole) and top-level comment blocks. Comments
    group by consecutive source rows; capped at _COMMENT_CAP_LINES per file."""
    module_doc = ""
    if root.named_child_count() > 0:
        first = root.named_child(0)
        if first.kind() == "expression_statement" and first.named_child_count() > 0:
            first = first.named_child(0)
        if first.kind() == "string":
            module_doc = _doc_text(first, src)
    blocks: list[str] = []
    current: list[str] = []
    last_row = None
    total = 0
    for i in range(root.child_count()):
        child = root.child(i)
        if child.kind() != "comment":
            continue
        row = child.start_position().row
        if last_row is not None and row != last_row + 1 and current:
            blocks.append("\n".join(current))
            current = []
        if total < _COMMENT_CAP_LINES:
            current.append(_text(child, src).lstrip("#").strip())
            total += 1
        last_row = row
    if current:
        blocks.append("\n".join(current))
    return module_doc, blocks


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

def _py_extract(node, src: bytes, imports: list[str], symbols: list[Symbol]) -> None:
    if node.kind() == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner is not None:
            _py_extract(inner, src, imports, symbols)
        return
    if node.kind() == "import_statement":
        for i in range(node.named_child_count()):
            child = node.named_child(i)
            if child.kind() == "dotted_name":
                imports.append(_text(child, src))
            elif child.kind() == "aliased_import":
                name = child.child_by_field_name("name")
                if name is not None:
                    imports.append(_text(name, src))
        return
    if node.kind() == "import_from_statement":
        module = node.child_by_field_name("module_name")
        if module is None:
            return
        base = _text(module, src)
        names: list[str] = []
        mstart = module.start_byte()
        for i in range(node.named_child_count()):
            child = node.named_child(i)
            if child.start_byte() == mstart:
                continue  # the module_name node itself
            if child.kind() == "dotted_name":
                names.append(_text(child, src))
            elif child.kind() == "aliased_import":
                name = child.child_by_field_name("name")
                if name is not None:
                    names.append(_text(name, src))
        if names:
            sep = "" if base.endswith(".") else "."
            imports.extend(f"{base}{sep}{n}" for n in names)
        else:
            imports.append(base)  # `from X import *` — bare module
        return
    if node.kind() == "function_definition":
        name = node.child_by_field_name("name")
        symbols.append(Symbol(
            kind="function",
            name=_text(name, src) if name is not None else "?",
            signature=_signature(node, src),
            doc=_py_docstring(node, src),
            doc_full=_py_docstring_full(node, src),
        ))
        return
    if node.kind() == "class_definition":
        name_node = node.child_by_field_name("name")
        cls_name = _text(name_node, src) if name_node is not None else "?"
        symbols.append(Symbol(
            kind="class",
            name=cls_name,
            signature=_signature(node, src),
            doc=_py_docstring(node, src),
            doc_full=_py_docstring_full(node, src),
        ))
        body = node.child_by_field_name("body")
        for i in range(body.named_child_count() if body is not None else 0):
            child = body.named_child(i)
            target = child
            if child.kind() == "decorated_definition":
                target = child.child_by_field_name("definition") or child
            if target.kind() == "function_definition":
                mname = target.child_by_field_name("name")
                symbols.append(Symbol(
                    kind="method",
                    name=_text(mname, src) if mname is not None else "?",
                    signature=_signature(target, src),
                    doc=_py_docstring(target, src),
                    doc_full=_py_docstring_full(target, src),
                    parent=cls_name,
                ))


# ---------------------------------------------------------------------------
# TypeScript / JavaScript
# ---------------------------------------------------------------------------

def _ts_extract(node, src: bytes, imports: list[str], symbols: list[Symbol]) -> None:
    if node.kind() == "export_statement":
        decl = node.child_by_field_name("declaration")
        if decl is not None:
            _ts_extract(decl, src, imports, symbols)
        return
    if node.kind() == "import_statement":
        source = node.child_by_field_name("source")
        if source is not None:
            imports.append(_text(source, src).strip("\"'"))
        return
    if node.kind() == "function_declaration":
        name = node.child_by_field_name("name")
        symbols.append(Symbol(
            kind="function",
            name=_text(name, src) if name is not None else "?",
            signature=_signature(node, src),
        ))
        return
    if node.kind() in ("class_declaration", "abstract_class_declaration"):
        name_node = node.child_by_field_name("name")
        cls_name = _text(name_node, src) if name_node is not None else "?"
        symbols.append(Symbol(kind="class", name=cls_name, signature=_signature(node, src)))
        body = node.child_by_field_name("body")
        for i in range(body.named_child_count() if body is not None else 0):
            child = body.named_child(i)
            if child.kind() == "method_definition":
                mname = child.child_by_field_name("name")
                symbols.append(Symbol(
                    kind="method",
                    name=_text(mname, src) if mname is not None else "?",
                    signature=_signature(child, src),
                    parent=cls_name,
                ))


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
