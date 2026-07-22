# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codeast.java — Java skeleton walker (Python parity).

Type declarations (class/interface/enum/record/annotation) map to kind
"class"; methods and constructors to kind "method" with `parent`; fields are
skipped. Javadoc immediately preceding a declaration (annotations live inside
the declaration node, so "above annotations" holds by construction) fills
doc/doc_full. Annotations fill `decorators` with '@' and call args stripped.
"""
from __future__ import annotations

from silica.kernel.codeast.base import (
    _CALL_NAME, Call, ModuleSkeleton, Symbol,
    _block_comment_text as _comment_text, _text,
)

_TYPE_DECLS = {
    "class_declaration", "interface_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
}
_METHOD_DECLS = {
    "method_declaration", "constructor_declaration",
    "compact_constructor_declaration",
}


def extract(root, src: bytes, path: str, language: str) -> ModuleSkeleton:
    imports: list[str] = []
    aliases: dict[str, str] = {}
    symbols: list[Symbol] = []
    calls: dict[tuple[str, str], None] = {}
    has_main = False

    module_doc = ""
    if root.named_child_count() > 0:
        first = root.named_child(0)
        nxt = root.named_child(1) if root.named_child_count() > 1 else None
        # a leading comment is the file header unless it is javadoc-positioned
        # (directly above the first type declaration — then the walk claims it)
        if first.kind() == "block_comment" and (nxt is None or nxt.kind() not in _TYPE_DECLS):
            module_doc = _comment_text(first, src)

    last_comment = None
    for i in range(root.named_child_count()):
        node = root.named_child(i)
        kind = node.kind()
        if kind == "block_comment":
            last_comment = node
            continue
        if kind == "import_declaration":
            spec = _import_text(node, src)
            if spec:
                imports.append(spec)
                simple = spec.rsplit(".", 1)[-1]
                if simple != "*" and "." in spec:
                    # SimpleName -> com.example.SimpleName: the import-scoped
                    # call matcher in codegraph catches Foo.bar() unchanged
                    aliases[simple] = spec
        elif kind in _TYPE_DECLS:
            has_main |= _type_decl(node, src, symbols, last_comment, parent="")
            _collect_calls(node, src, calls, parent=_name_of(node, src))
        last_comment = None

    # ponytail: same-package Java calls need no import and are invisible here;
    # resolve against a per-package symbol table if Java call graphs read thin
    # on a real repo (single-package projects lose every internal edge)
    return ModuleSkeleton(
        path=path, language=language, imports=imports, symbols=symbols,
        module_doc=module_doc,
        calls=[Call(name=k[0], parent=k[1]) for k in calls],
        import_aliases=aliases, has_main_guard=has_main,
    )


def _name_of(node, src: bytes) -> str:
    name = node.child_by_field_name("name")
    return _text(name, src) if name is not None else "?"


def _import_text(node, src: bytes) -> str:
    """Dotted import string verbatim, wildcard kept as written."""
    text = _text(node, src).strip().rstrip(";").strip()
    for kw in ("import", "static"):
        if text.startswith(kw):
            text = text[len(kw):].strip()
    return "".join(text.split())


def _annotations(node, src: bytes) -> list[str]:
    """Annotation names of a declaration, '@' and call args stripped."""
    out: list[str] = []
    for i in range(node.named_child_count()):
        child = node.named_child(i)
        if child.kind() != "modifiers":
            continue
        for j in range(child.named_child_count()):
            ann = child.named_child(j)
            if ann.kind() in ("marker_annotation", "annotation"):
                out.append(_text(ann, src).lstrip("@").split("(", 1)[0].strip())
    return out


def _signature(node, src: bytes) -> str:
    """Declaration text after annotations, up to the body, collapsed —
    annotations live in `decorators`, mirroring Python's decorator handling."""
    start = node.start_byte()
    for i in range(node.named_child_count()):
        child = node.named_child(i)
        if child.kind() == "modifiers":
            for j in range(child.named_child_count()):
                ann = child.named_child(j)
                if ann.kind() in ("marker_annotation", "annotation"):
                    start = max(start, ann.end_byte())
    body = node.child_by_field_name("body")
    end = body.start_byte() if body is not None else node.end_byte()
    return " ".join(src[start:end].decode("utf-8", errors="replace").split())


def _is_static_void(node, src: bytes) -> bool:
    static = False
    void = False
    for i in range(node.named_child_count()):
        child = node.named_child(i)
        if child.kind() == "modifiers":
            static = "static" in _text(child, src).split()
        elif child.kind() == "void_type":
            void = True
    return static and void


def _doc_pair(comment, src: bytes) -> tuple[str, str]:
    full = _comment_text(comment, src) if comment is not None else ""
    return (full.splitlines()[0].strip() if full else "", full)


def _type_decl(node, src: bytes, symbols: list[Symbol], comment, parent: str) -> bool:
    """Append the type + its members; True when a `static void main` exists."""
    name = _name_of(node, src)
    doc, doc_full = _doc_pair(comment, src)
    symbols.append(Symbol(
        kind="class", name=name, signature=_signature(node, src),
        doc=doc, doc_full=doc_full, parent=parent,
        decorators=_annotations(node, src),
    ))
    has_main = False
    body = node.child_by_field_name("body")
    if body is None:
        return False
    last_comment = None
    for i in range(body.named_child_count()):
        child = body.named_child(i)
        kind = child.kind()
        if kind == "block_comment":
            last_comment = child
            continue
        if kind in _METHOD_DECLS:
            mdoc, mdoc_full = _doc_pair(last_comment, src)
            mname = _name_of(child, src)
            symbols.append(Symbol(
                kind="method", name=mname, signature=_signature(child, src),
                doc=mdoc, doc_full=mdoc_full, parent=name,
                decorators=_annotations(child, src),
            ))
            if mname == "main" and _is_static_void(child, src):
                has_main = True
        elif kind in _TYPE_DECLS:
            has_main |= _type_decl(child, src, symbols, last_comment, parent=name)
        last_comment = None
    return has_main


def _collect_calls(node, src: bytes, out: dict[tuple[str, str], None], parent: str) -> None:
    kind = node.kind()
    if kind == "method_invocation":
        obj = node.child_by_field_name("object")
        name = node.child_by_field_name("name")
        text = ((_text(obj, src) + ".") if obj is not None else "")
        text += _text(name, src) if name is not None else ""
        if text and _CALL_NAME.match(text):
            out[(text, parent)] = None
    elif kind == "object_creation_expression":
        typ = node.child_by_field_name("type")
        if typ is not None:
            text = _text(typ, src)
            if _CALL_NAME.match(text):
                out[(text, parent)] = None
    for i in range(node.named_child_count()):
        _collect_calls(node.named_child(i), src, out, parent)
