# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codeast.c — C/C++ skeleton walker, one walker on both grammars.

Function definitions and header prototypes map to kind "function";
struct/class/enum/union/typedef to kind "class"; methods inside a class body
to kind "method" with `parent`. namespace_definition and template_declaration
are transparent (the walker recurses through them). Within one file, symbols
dedupe by (kind, name, parent); the first occurrence carrying a doc comment
wins. Includes are stored with their delimiters so resolution can distinguish
quoted from angled.
"""
from __future__ import annotations

from silica.kernel.codeast.base import (
    _CALL_NAME, Call, ModuleSkeleton, Symbol,
    _block_comment_text, _signature, _text,
)

_TYPE_SPECIFIERS = {
    "struct_specifier", "class_specifier", "union_specifier", "enum_specifier",
}
_TRANSPARENT = {
    "namespace_definition", "template_declaration", "linkage_specification",
    "preproc_ifdef", "preproc_if", "preproc_else", "declaration_list",
}
_DOC_TAKERS = {"function_definition", "declaration", "type_definition"} | _TYPE_SPECIFIERS


def extract(root, src: bytes, path: str, language: str) -> ModuleSkeleton:
    imports: list[str] = []
    symbols: dict[tuple[str, str, str], Symbol] = {}
    calls: dict[tuple[str, str], None] = {}
    state = {"main": False, "module_doc": None}
    _walk(root, src, imports, symbols, calls, state)
    return ModuleSkeleton(
        path=path, language=language, imports=imports,
        symbols=list(symbols.values()),
        module_doc=state["module_doc"] or "",
        calls=[Call(name=k[0], parent=k[1]) for k in calls],
        has_main_guard=state["main"],
    )


def _walk(container, src: bytes, imports, symbols, calls, state) -> None:
    pending: list = []
    for i in range(container.named_child_count()):
        node = container.named_child(i)
        kind = node.kind()
        if kind == "comment":
            pending.append(node)
            continue
        if state["module_doc"] is None:
            # file-header comment block: whatever the first real node does not
            # claim as its own doc comment becomes module_doc
            header = _split_doc(pending, src)[0] if kind in _DOC_TAKERS else pending
            state["module_doc"] = _comments_text(header, src)
        if kind == "preproc_include":
            p = node.child_by_field_name("path")
            if p is not None:
                imports.append(_text(p, src).strip())
        elif kind in _TRANSPARENT:
            _walk(node, src, imports, symbols, calls, state)
        elif kind == "function_definition":
            _function(node, src, symbols, calls, state,
                      _split_doc(pending, src)[1], parent_class="", is_def=True)
        elif kind == "declaration":
            _function(node, src, symbols, calls, state,
                      _split_doc(pending, src)[1], parent_class="", is_def=False)
        elif kind in _TYPE_SPECIFIERS:
            _type_spec(node, src, symbols, calls, state, _split_doc(pending, src)[1])
        elif kind == "type_definition":
            decl = node.child_by_field_name("declarator")
            if decl is not None:
                full = _comments_text(_split_doc(pending, src)[1], src)
                _add(symbols, Symbol(kind="class", name=_text(decl, src),
                                     signature=_sig(node, src),
                                     doc=_first_line(full), doc_full=full))
        pending = []


# ---------------------------------------------------------------------------
# doc comments: /** */, /*! */, or a run of ///
# ---------------------------------------------------------------------------

def _split_doc(pending: list, src: bytes) -> tuple[list, list]:
    """(header_part, doc_part): the trailing doc-style run is claimed by the
    declaration that follows; anything before it is not a doc comment."""
    if not pending:
        return [], []
    last = _text(pending[-1], src).lstrip()
    if last.startswith(("/**", "/*!")):
        return pending[:-1], pending[-1:]
    if last.startswith("///"):
        i = len(pending)
        while i > 0 and _text(pending[i - 1], src).lstrip().startswith("///"):
            i -= 1
        return pending[:i], pending[i:]
    return pending, []


def _comments_text(comments: list, src: bytes) -> str:
    parts: list[str] = []
    for c in comments:
        text = _text(c, src).strip()
        if text.startswith("/*"):
            parts.append(_block_comment_text(c, src))
        elif text.startswith("//"):
            parts.append(text.lstrip("/!").strip())
    return "\n".join(p for p in parts if p).strip()


def _first_line(full: str) -> str:
    return full.splitlines()[0].strip() if full else ""


# ---------------------------------------------------------------------------
# symbols
# ---------------------------------------------------------------------------

def _add(symbols: dict, sym: Symbol) -> None:
    key = (sym.kind, sym.name, sym.parent)
    old = symbols.get(key)
    if old is None or (not old.doc and sym.doc):
        symbols[key] = sym


def _sig(node, src: bytes) -> str:
    return _signature(node, src).rstrip(";").strip()


def _function_declarator(node):
    d = node.child_by_field_name("declarator")
    depth = 0
    while d is not None and d.kind() != "function_declarator" and depth < 8:
        d = d.child_by_field_name("declarator")
        depth += 1
    return d if d is not None and d.kind() == "function_declarator" else None


def _function(node, src: bytes, symbols, calls, state, doc_comments,
              parent_class: str, is_def: bool) -> None:
    fd = _function_declarator(node)
    if fd is None:
        return  # not a function declaration (plain variable, field, ...)
    decl = fd.child_by_field_name("declarator")
    if decl is None:
        return
    text = _text(decl, src)
    if "::" in text:
        # out-of-class definition Foo::bar — same (method, bar, Foo) key as
        # the in-class prototype, so the two dedupe to one symbol
        scope, _, name = text.rpartition("::")
        parent = scope.rsplit("::", 1)[-1]
        kind = "method"
    elif parent_class:
        name, parent, kind = text, parent_class, "method"
    else:
        name, parent, kind = text, "", "function"
    if not _CALL_NAME.match(name):
        return  # operator overloads, destructors: outside the skeleton
    full = _comments_text(doc_comments, src)
    _add(symbols, Symbol(kind=kind, name=name, signature=_sig(node, src),
                         doc=_first_line(full), doc_full=full, parent=parent))
    if is_def:
        body = node.child_by_field_name("body")
        if body is not None:
            _collect_calls(body, src, calls, parent or name)
        if kind == "function" and name == "main":
            state["main"] = True


def _type_spec(node, src: bytes, symbols, calls, state, doc_comments,
               parent: str = "") -> None:
    name_node = node.child_by_field_name("name")
    body = node.child_by_field_name("body")
    if name_node is None or body is None:
        return  # anonymous, forward declaration, or bare usage: no symbol
    name = _text(name_node, src)
    full = _comments_text(doc_comments, src)
    _add(symbols, Symbol(kind="class", name=name, signature=_sig(node, src),
                         doc=_first_line(full), doc_full=full, parent=parent))
    pending: list = []
    for i in range(body.named_child_count()):
        child = body.named_child(i)
        kind = child.kind()
        if kind == "comment":
            pending.append(child)
            continue
        if kind in ("field_declaration", "declaration"):
            _function(child, src, symbols, calls, state,
                      _split_doc(pending, src)[1], parent_class=name, is_def=False)
        elif kind == "function_definition":
            _function(child, src, symbols, calls, state,
                      _split_doc(pending, src)[1], parent_class=name, is_def=True)
        elif kind in _TYPE_SPECIFIERS:
            _type_spec(child, src, symbols, calls, state,
                       _split_doc(pending, src)[1], parent=name)
        pending = []


def _collect_calls(node, src: bytes, out: dict, parent: str) -> None:
    if node.kind() == "call_expression":
        fn = node.child_by_field_name("function")
        if fn is not None:
            text = _text(fn, src).replace("::", ".")
            if _CALL_NAME.match(text):
                out[(text, parent)] = None
    for i in range(node.named_child_count()):
        _collect_calls(node.named_child(i), src, out, parent)
