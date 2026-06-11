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
    assert "pathlib" in sk.imports
    assert "silica.kernel" in sk.imports


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
