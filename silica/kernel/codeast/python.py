# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codeast.python — Python skeleton walker."""
from __future__ import annotations

from silica.kernel.codeast.base import _CALL_NAME, Call, Symbol, _signature, _text


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


def _py_decorators(node, src: bytes) -> list[str]:
    """Decorator names of a decorated_definition, '@' and call args stripped."""
    out: list[str] = []
    for i in range(node.named_child_count()):
        child = node.named_child(i)
        if child.kind() == "decorator":
            out.append(_text(child, src).lstrip("@").split("(", 1)[0].strip())
    return out


def _py_dunder_all(root, src: bytes) -> list[str] | None:
    """Literal `__all__` list, or None (absent / dynamic: no authority). This
    grammar emits a bare `assignment` at module level; older ones wrap it in
    `expression_statement`, so both shapes are unwrapped."""
    for i in range(root.named_child_count()):
        node = root.named_child(i)
        assign = node
        if node.kind() == "expression_statement" and node.named_child_count() > 0:
            assign = node.named_child(0)
        if assign.kind() != "assignment":
            continue
        left = assign.child_by_field_name("left")
        right = assign.child_by_field_name("right")
        if left is None or right is None or _text(left, src) != "__all__":
            continue
        if right.kind() != "list":
            return None
        names: list[str] = []
        for j in range(right.named_child_count()):
            el = right.named_child(j)
            if el.kind() != "string":
                return None
            names.append(_strip_quotes(_text(el, src).strip()))
        return names
    return None


def _py_calls(root, src: bytes) -> list[Call]:
    """Every call site's spelled name, tagged with its top-level container.
    Only grammar-clean names (identifier / dotted attribute) are kept:
    `f().g(...)` and subscripted receivers are skipped by the regex."""
    out: dict[tuple[str, str], None] = {}

    def walk(node, parent: str) -> None:
        if node.kind() == "call":
            fn = node.child_by_field_name("function")
            if fn is not None:
                text = _text(fn, src)
                if _CALL_NAME.match(text):
                    out[(text, parent)] = None
        for i in range(node.named_child_count()):
            walk(node.named_child(i), parent)

    for i in range(root.named_child_count()):
        node = root.named_child(i)
        target = node
        if node.kind() == "decorated_definition":
            target = node.child_by_field_name("definition") or node
        name = ""
        if target.kind() in ("function_definition", "class_definition"):
            n = target.child_by_field_name("name")
            name = _text(n, src) if n is not None else ""
        walk(node, name)
    return [Call(name=k[0], parent=k[1]) for k in out]


def _py_has_main_guard(root, src: bytes) -> bool:
    for i in range(root.named_child_count()):
        node = root.named_child(i)
        if node.kind() == "if_statement":
            cond = node.child_by_field_name("condition")
            if cond is not None and "__name__" in _text(cond, src):
                return True
    return False


def _py_extract(node, src: bytes, imports: list[str], symbols: list[Symbol],
                decorators: list[str] | None = None,
                aliases: dict[str, str] | None = None) -> None:
    if node.kind() == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner is not None:
            _py_extract(inner, src, imports, symbols, _py_decorators(node, src), aliases)
        return
    if node.kind() == "import_statement":
        for i in range(node.named_child_count()):
            child = node.named_child(i)
            if child.kind() == "dotted_name":
                imports.append(_text(child, src))
            elif child.kind() == "aliased_import":
                name = child.child_by_field_name("name")
                alias = child.child_by_field_name("alias")
                if name is not None:
                    imports.append(_text(name, src))
                    if alias is not None and aliases is not None:
                        aliases[_text(alias, src)] = _text(name, src)
        return
    if node.kind() == "import_from_statement":
        module = node.child_by_field_name("module_name")
        if module is None:
            return
        base = _text(module, src)
        sep = "" if base.endswith(".") else "."
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
                alias = child.child_by_field_name("alias")
                if name is not None:
                    names.append(_text(name, src))
                    if alias is not None and aliases is not None:
                        aliases[_text(alias, src)] = f"{base}{sep}{_text(name, src)}"
        if names:
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
            decorators=decorators or [],
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
            decorators=decorators or [],
        ))
        body = node.child_by_field_name("body")
        for i in range(body.named_child_count() if body is not None else 0):
            child = body.named_child(i)
            target = child
            method_decos: list[str] = []
            if child.kind() == "decorated_definition":
                target = child.child_by_field_name("definition") or child
                method_decos = _py_decorators(child, src)
            if target.kind() == "function_definition":
                mname = target.child_by_field_name("name")
                symbols.append(Symbol(
                    kind="method",
                    name=_text(mname, src) if mname is not None else "?",
                    signature=_signature(target, src),
                    doc=_py_docstring(target, src),
                    doc_full=_py_docstring_full(target, src),
                    parent=cls_name,
                    decorators=method_decos,
                ))
