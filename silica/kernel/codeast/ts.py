# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codeast.ts — TypeScript / JavaScript skeleton walker."""
from __future__ import annotations

from silica.kernel.codeast.base import Symbol, _signature, _text


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
