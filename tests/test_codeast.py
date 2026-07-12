"""kernel/codeast — shallow tree-sitter skeleton extraction (ADR-0012 slice)."""
from silica.kernel.codeast import EXTENSION_MAP, ModuleSkeleton, extract_skeleton, language_for

PY_SRC = '''\
"""Module docstring."""
import os
import silica.kernel.gitstate
from pathlib import Path
from silica.kernel import frontmatter


def hi(name: str) -> str:
    """Say hi to name.

    Second line ignored.
    """
    return f"hi {name}"


class FSM:
    """Injector state machine."""

    def run(self, files: list[str]) -> None:
        """Run the loop."""
        return None

    def _private(self):
        return 1
'''

TS_SRC = '''\
import { foo } from "./local/helper";
import * as fs from "fs";

export function greet(name: string): string {
  return `hi ${name}`;
}

class Machine {
  run(files: string[]): void {
    return;
  }
}
'''


def test_language_for_known_and_unknown():
    assert language_for("silica/cli.py") == "python"
    assert language_for("src/app.ts") == "typescript"
    assert language_for("src/app.jsx") == "javascript"
    assert language_for("notes/readme.md") is None
    assert language_for("Makefile") is None


def test_extension_map_only_supported_languages():
    assert set(EXTENSION_MAP.values()) <= {"python", "typescript", "javascript"}


def test_python_imports():
    sk = extract_skeleton(PY_SRC, "python", path="src/m.py")
    assert isinstance(sk, ModuleSkeleton)
    assert "os" in sk.imports
    assert "silica.kernel.gitstate" in sk.imports
    assert "pathlib.Path" in sk.imports               # was: "pathlib"
    assert "silica.kernel.frontmatter" in sk.imports  # was: "silica.kernel"


def test_python_symbols_signatures_and_docstrings():
    sk = extract_skeleton(PY_SRC, "python", path="src/m.py")
    by_name = {s.name: s for s in sk.symbols}
    fn = by_name["hi"]
    assert fn.kind == "function"
    assert "def hi(name: str) -> str" in fn.signature
    assert fn.doc == "Say hi to name."
    cls = by_name["FSM"]
    assert cls.kind == "class"
    assert cls.doc == "Injector state machine."
    run = by_name["run"]
    assert run.kind == "method"
    assert run.parent == "FSM"
    assert "def run(self, files: list[str]) -> None" in run.signature
    assert run.doc == "Run the loop."
    # private methods are still skeleton (shallow = mechanical, no judgement)
    assert "_private" in by_name


def test_typescript_imports_and_symbols():
    sk = extract_skeleton(TS_SRC, "typescript", path="src/app.ts")
    assert "./local/helper" in sk.imports
    assert "fs" in sk.imports
    by_name = {s.name: s for s in sk.symbols}
    assert "greet" in by_name
    assert "function greet(name: string): string" in by_name["greet"].signature
    assert by_name["run"].parent == "Machine"


def test_unparseable_source_returns_empty_skeleton():
    sk = extract_skeleton("\x00\x01garbage((", "python", path="x.py")
    assert isinstance(sk, ModuleSkeleton)  # never raises


def test_parser_failure_degrades_to_empty_skeleton():
    sk = extract_skeleton("def hi(): pass", "not-a-language", path="x.py")
    assert isinstance(sk, ModuleSkeleton)
    assert sk.imports == [] and sk.symbols == []


PY_DECORATED = '''\
from dataclasses import dataclass


@dataclass
class Config:
    """Holds settings."""

    @staticmethod
    def load(path: str) -> "Config":
        """Load from disk."""
        return Config()
'''


def test_python_decorated_class_and_method():
    sk = extract_skeleton(PY_DECORATED, "python", path="src/c.py")
    by_name = {s.name: s for s in sk.symbols}
    assert by_name["Config"].kind == "class"
    assert by_name["Config"].doc == "Holds settings."
    assert by_name["load"].kind == "method"
    assert by_name["load"].parent == "Config"


def test_javascript_smoke():
    sk = extract_skeleton('import x from "./x";\nfunction go(a) {\n  return a;\n}\n', "javascript", path="a.js")
    assert "./x" in sk.imports
    assert any(s.name == "go" and s.kind == "function" for s in sk.symbols)


FROM_IMPORTS = '''\
from silica.kernel import frontmatter, gitstate
from pathlib import Path
from .paths import atomic_write_bytes
from . import helpers
from os import *
'''


def test_from_import_records_module_dot_name():
    sk = extract_skeleton(FROM_IMPORTS, "python", path="src/m.py")
    assert "silica.kernel.frontmatter" in sk.imports
    assert "silica.kernel.gitstate" in sk.imports
    assert "pathlib.Path" in sk.imports
    assert ".paths.atomic_write_bytes" in sk.imports
    assert ".helpers" in sk.imports        # `from . import helpers`
    assert "os" in sk.imports              # wildcard falls back to bare module


def test_parse_error_flag():
    ok = extract_skeleton("def hi(): pass", "python", path="x.py")
    assert ok.parse_error is False
    bad = extract_skeleton("def hi(): pass", "not-a-language", path="x.py")
    assert bad.parse_error is True


def test_diff_skeletons_empty_for_body_only_change():
    old = extract_skeleton("def hi(name: str) -> str:\n    return name\n", "python")
    new = extract_skeleton("def hi(name: str) -> str:\n    x = name.upper()\n    return x\n", "python")
    from silica.kernel.codeast import diff_skeletons
    assert diff_skeletons(old, new) == []


def test_diff_skeletons_reports_structure():
    from silica.kernel.codeast import diff_skeletons
    old = extract_skeleton(
        "import os\n\nclass A:\n    def run(self) -> None: ...\n\ndef gone(): ...\n", "python")
    new = extract_skeleton(
        "import sys\n\nclass A:\n    def run(self, fast: bool) -> None: ...\n\ndef added(): ...\n", "python")
    diff = diff_skeletons(old, new)
    assert "+ import sys" in diff
    assert "- import os" in diff
    assert "+ function added" in diff
    assert "- function gone" in diff
    assert "signature changed: A.run" in diff


# ---------------------------------------------------------------------------
# Task 1: full docstrings + module doc + module comments
# ---------------------------------------------------------------------------

PY_DOCFULL = '''"""Module doc line one.

Second paragraph."""
# top comment A
# top comment B

import os


def f():
    """First line.

    More detail here.
    """
    return 1
'''


def test_doc_full_module_doc_and_comments():
    sk = extract_skeleton(PY_DOCFULL, "python", path="m.py")
    assert sk.module_doc.startswith("Module doc line one.")
    assert "Second paragraph." in sk.module_doc
    assert sk.module_comments == ["top comment A\ntop comment B"]
    f = next(s for s in sk.symbols if s.name == "f")
    assert f.doc == "First line."
    assert "More detail here." in f.doc_full


def test_no_doc_no_comments_yield_empty_fields():
    sk = extract_skeleton("x = 1\n", "python", path="m.py")
    assert sk.module_doc == ""
    assert sk.module_comments == []


# ---------------------------------------------------------------------------
# Task 2: decorators + __all__
# ---------------------------------------------------------------------------

PY_DECOS = '''__all__ = ["Cli", "run"]

import functools


class Cli:
    @property
    def name(self):
        return "x"


@functools.lru_cache(maxsize=8)
def run():
    pass


def _hidden():
    pass
'''


def test_decorators_captured_function_and_method():
    sk = extract_skeleton(PY_DECOS, "python", path="m.py")
    run = next(s for s in sk.symbols if s.name == "run")
    assert run.decorators == ["functools.lru_cache"]
    name = next(s for s in sk.symbols if s.name == "name" and s.parent == "Cli")
    assert name.decorators == ["property"]


def test_dunder_all_literal_captured():
    sk = extract_skeleton(PY_DECOS, "python", path="m.py")
    assert sk.dunder_all == ["Cli", "run"]


def test_dunder_all_dynamic_is_none():
    sk = extract_skeleton("__all__ = [x for x in names]\n", "python", path="m.py")
    assert sk.dunder_all is None
