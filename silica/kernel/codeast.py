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


@dataclass(frozen=True)
class ModuleSkeleton:
    path: str                  # repo-relative source path
    language: str              # EXTENSION_MAP value
    imports: list[str] = field(default_factory=list)   # module strings, duplicates possible
    symbols: list[Symbol] = field(default_factory=list)  # document order


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
        return ModuleSkeleton(path=path, language=language)

    src = source.encode("utf-8")
    imports: list[str] = []
    symbols: list[Symbol] = []
    root = tree.root_node()
    extract = _py_extract if language == "python" else _ts_extract
    for i in range(root.named_child_count()):
        extract(root.named_child(i), src, imports, symbols)
    return ModuleSkeleton(path=path, language=language, imports=imports, symbols=symbols)


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


def _py_docstring(node, src: bytes) -> str:
    """First line of the body's leading string node, quotes stripped."""
    body = node.child_by_field_name("body")
    if body is None or body.named_child_count() == 0:
        return ""
    first = body.named_child(0)
    # In tree-sitter >= 0.23 the docstring is a bare 'string' node as first
    # child of the block (no wrapping expression_statement).
    if first.kind() != "string":
        # Fallback: may be wrapped in expression_statement in some grammar versions
        if first.kind() == "expression_statement" and first.named_child_count() > 0:
            first = first.named_child(0)
            if first.kind() != "string":
                return ""
        else:
            return ""
    text = _text(first, src).strip()
    for q in ('"""', "'''", '"', "'"):
        if text.startswith(q) and text.endswith(q) and len(text) >= 2 * len(q):
            text = text[len(q):-len(q)]
            break
    return text.strip().splitlines()[0].strip() if text.strip() else ""


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
        if module is not None:
            imports.append(_text(module, src))
        return
    if node.kind() == "function_definition":
        name = node.child_by_field_name("name")
        symbols.append(Symbol(
            kind="function",
            name=_text(name, src) if name is not None else "?",
            signature=_signature(node, src),
            doc=_py_docstring(node, src),
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
