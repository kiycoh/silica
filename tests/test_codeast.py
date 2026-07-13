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
    assert set(EXTENSION_MAP.values()) <= {
        "python", "typescript", "javascript", "java", "c", "cpp",
        "toml", "html", "css"}


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


# ---------------------------------------------------------------------------
# Task 3: call sites + import aliases + main guard
# ---------------------------------------------------------------------------

PY_CALLS = '''from pkg.util import helper
from pkg import util
import pkg.alias_target as at


def main():
    helper()
    util.helper()
    at.go()
    _local()


def _local():
    pass


if __name__ == "__main__":
    main()
'''


def test_calls_collected_with_parent():
    sk = extract_skeleton(PY_CALLS, "python", path="pkg/app.py")
    pairs = {(c.name, c.parent) for c in sk.calls}
    assert ("helper", "main") in pairs
    assert ("util.helper", "main") in pairs
    assert ("at.go", "main") in pairs
    assert ("_local", "main") in pairs
    assert ("main", "") in pairs  # module-level call under the guard


def test_import_aliases_and_main_guard():
    sk = extract_skeleton(PY_CALLS, "python", path="pkg/app.py")
    assert sk.import_aliases == {"at": "pkg.alias_target"}
    assert sk.has_main_guard is True
    assert extract_skeleton("x = 1\n", "python", path="m.py").has_main_guard is False


def test_from_import_alias_recorded():
    sk = extract_skeleton("from pkg.util import helper as h\n", "python", path="m.py")
    assert sk.import_aliases == {"h": "pkg.util.helper"}


# ---------------------------------------------------------------------------
# Languages spec §1: bare languages (toml/html/css) — presence only
# ---------------------------------------------------------------------------

def test_bare_language_extensions_mapped():
    from silica.kernel.codeast import BARE_LANGUAGES
    assert BARE_LANGUAGES == {"toml", "html", "css"}
    assert language_for("pyproject.toml") == "toml"
    assert language_for("site/index.html") == "html"
    assert language_for("site/style.css") == "css"


def test_bare_language_empty_skeleton_without_parsing(monkeypatch):
    # "no structure" is true, not a failure: parse_error stays False, and the
    # parser is never consulted (a broken parser must not matter for bare files)
    import tree_sitter_language_pack

    def boom(_lang):
        raise AssertionError("bare language reached the parser")

    monkeypatch.setattr(tree_sitter_language_pack, "get_parser", boom)
    for lang in ("toml", "html", "css"):
        sk = extract_skeleton("<<< anything {{{", lang, path=f"x.{lang}")
        assert sk.parse_error is False
        assert sk.imports == [] and sk.symbols == [] and sk.calls == []


# ---------------------------------------------------------------------------
# Languages spec §2: Java extractor (Python parity)
# ---------------------------------------------------------------------------

JAVA_SRC = '''\
/** File header. */
package com.example.app;

import com.example.util.Helper;
import com.example.io.*;
import java.util.List;

/**
 * Greeter service.
 * Second line.
 */
@Service
@RequestMapping("/greet")
public class Greeter {
    private int count;

    /** Say hi. */
    @Override
    public String hi(String name) {
        Helper.assist(name);
        List.of(name);
        local();
        Helper h = new Helper();
        return "hi";
    }

    public Greeter(int c) { this.count = c; }

    /** Inner. */
    static class Inner {
        void run() {}
    }

    public static void main(String[] args) {
        new Greeter(1).hi("x");
    }
}

record Point(int x, int y) {}

interface Shape { double area(); }

enum Color { RED, GREEN }

@interface Marker {}
'''


def test_java_language_for():
    assert language_for("src/main/java/com/example/App.java") == "java"


def test_java_imports_verbatim_including_wildcard():
    sk = extract_skeleton(JAVA_SRC, "java", path="Greeter.java")
    assert sk.parse_error is False
    assert "com.example.util.Helper" in sk.imports
    assert "com.example.io.*" in sk.imports   # wildcard kept as written
    assert "java.util.List" in sk.imports


def test_java_symbols_kinds_parents_fields_skipped():
    sk = extract_skeleton(JAVA_SRC, "java", path="Greeter.java")
    by_key = {(s.kind, s.name): s for s in sk.symbols}
    assert ("class", "Greeter") in by_key
    assert by_key[("method", "hi")].parent == "Greeter"
    assert by_key[("method", "Greeter")].parent == "Greeter"   # constructor
    assert by_key[("class", "Inner")].parent == "Greeter"      # inner stays class
    assert by_key[("method", "run")].parent == "Inner"
    # class-kinded declarations: record / interface / enum / annotation
    for name in ("Point", "Shape", "Color", "Marker"):
        assert ("class", name) in by_key
    # fields are skipped (Python parity: only def/class captured)
    assert not any(s.name == "count" for s in sk.symbols)


def test_java_javadoc_and_module_doc():
    sk = extract_skeleton(JAVA_SRC, "java", path="Greeter.java")
    assert sk.module_doc == "File header."
    greeter = next(s for s in sk.symbols if s.name == "Greeter" and s.kind == "class")
    assert greeter.doc == "Greeter service."
    assert "Second line." in greeter.doc_full
    hi = next(s for s in sk.symbols if s.name == "hi")
    assert hi.doc == "Say hi."


def test_java_annotations_as_decorators():
    sk = extract_skeleton(JAVA_SRC, "java", path="Greeter.java")
    greeter = next(s for s in sk.symbols if s.name == "Greeter" and s.kind == "class")
    assert greeter.decorators == ["Service", "RequestMapping"]  # args stripped
    hi = next(s for s in sk.symbols if s.name == "hi")
    assert hi.decorators == ["Override"]
    assert "@" not in greeter.signature   # annotations live in decorators, not the signature


def test_java_calls_aliases_and_main():
    sk = extract_skeleton(JAVA_SRC, "java", path="Greeter.java")
    assert sk.import_aliases["Helper"] == "com.example.util.Helper"
    assert sk.import_aliases["List"] == "java.util.List"
    pairs = {(c.name, c.parent) for c in sk.calls}
    assert ("Helper.assist", "Greeter") in pairs
    assert ("local", "Greeter") in pairs
    assert ("Helper", "Greeter") in pairs   # `new Helper()` — constructor call
    assert sk.has_main_guard is True
    assert sk.dunder_all is None
    no_main = extract_skeleton("class A { void go() {} }", "java", path="A.java")
    assert no_main.has_main_guard is False


# ---------------------------------------------------------------------------
# Languages spec §3: C/C++ extractor (single walker on both grammars)
# ---------------------------------------------------------------------------

CPP_SRC = '''\
/** File header. */
#include "util/helper.h"
#include <stdio.h>

/// Doxygen line one.
/// Line two.
int helper(int x);

/*! Adds numbers. */
int add(int a, int b) {
    helper(a);
    printf("x");
    return a + b;
}

/** Point struct. */
struct Point { int x; int y; };

typedef struct Point PointT;

namespace geo {
    /** Shape class. */
    class Shape {
    public:
        /** Area. */
        double area() const;
        void scale(double f) { helper(1); }
    };

    double Shape::area() const { return 0.0; }
}

template <typename T>
T identity(T v) { return v; }

int main(int argc, char** argv) {
    add(1, 2);
    return 0;
}
'''


def test_c_cpp_extensions_mapped():
    assert language_for("src/main.c") == "c"
    for ext in (".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"):
        assert language_for(f"src/x{ext}") == "cpp"   # .h parsed as cpp superset


def test_cpp_includes_keep_delimiters():
    sk = extract_skeleton(CPP_SRC, "cpp", path="src/m.cpp")
    assert sk.parse_error is False
    assert '"util/helper.h"' in sk.imports   # quoted vs angled must survive
    assert "<stdio.h>" in sk.imports


def test_cpp_symbols_functions_structs_methods():
    sk = extract_skeleton(CPP_SRC, "cpp", path="src/m.cpp")
    by_key = {(s.kind, s.name): s for s in sk.symbols}
    assert ("function", "helper") in by_key          # header prototype
    assert ("function", "add") in by_key
    assert ("class", "Point") in by_key              # struct → class
    assert ("class", "PointT") in by_key             # typedef → class
    assert ("class", "Shape") in by_key              # through namespace (transparent)
    assert by_key[("method", "area")].parent == "Shape"
    assert by_key[("method", "scale")].parent == "Shape"
    assert ("function", "identity") in by_key        # through template (transparent)
    # prototype + out-of-class definition dedupe to one symbol
    assert len([s for s in sk.symbols if s.name == "area"]) == 1
    assert sk.has_main_guard is True
    assert extract_skeleton("int helper(int);\n", "cpp", path="h.h").has_main_guard is False


def test_cpp_doc_comments_doxygen_styles():
    sk = extract_skeleton(CPP_SRC, "cpp", path="src/m.cpp")
    assert sk.module_doc == "File header."
    helper = next(s for s in sk.symbols if s.name == "helper")
    assert helper.doc == "Doxygen line one."         # /// run
    assert "Line two." in helper.doc_full
    add = next(s for s in sk.symbols if s.name == "add")
    assert add.doc == "Adds numbers."                # /*! */
    shape = next(s for s in sk.symbols if s.name == "Shape")
    assert shape.doc == "Shape class."               # /** */
    area = next(s for s in sk.symbols if s.name == "area")
    assert area.doc == "Area."                       # first occurrence carrying a doc wins


def test_cpp_calls_collected_with_parent():
    sk = extract_skeleton(CPP_SRC, "cpp", path="src/m.cpp")
    pairs = {(c.name, c.parent) for c in sk.calls}
    assert ("helper", "add") in pairs
    assert ("printf", "add") in pairs
    assert ("add", "main") in pairs
    assert ("helper", "Shape") in pairs              # inline method body


def test_c_grammar_same_walker():
    src = "#include \"u.h\"\n\nstruct P { int x; };\n\nint go(void) { return 0; }\n"
    sk = extract_skeleton(src, "c", path="m.c")
    assert '"u.h"' in sk.imports
    by_key = {(s.kind, s.name) for s in sk.symbols}
    assert ("class", "P") in by_key and ("function", "go") in by_key


def test_cpp_header_guard_is_transparent():
    src = ("#ifndef X_H\n#define X_H\n\n"
           "int helper(int x);\n\n#endif\n")
    sk = extract_skeleton(src, "cpp", path="x.h")
    assert any(s.name == "helper" and s.kind == "function" for s in sk.symbols)
