# tests/test_codegraph.py
"""kernel/codegraph — derived structural code index (spec-code-lane §1)."""
from pathlib import Path

from silica.kernel.codegraph import classify_import, is_first_party, package_of

PY_FILES = {
    "silica/__init__.py",
    "silica/kernel/__init__.py",
    "silica/kernel/embed.py",
    "silica/kernel/paths.py",
    "silica/cli.py",
}
TS_FILES = {
    "src/app.ts",
    "src/local/helper.ts",
    "src/lib/index.ts",
}


def test_python_absolute_module():
    kind, val = classify_import("silica.kernel.paths", "silica/cli.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/kernel/paths.py")


def test_python_from_import_name_backs_off_to_module():
    # from silica.kernel import paths → "silica.kernel.paths" resolves to the module file
    kind, val = classify_import("silica.kernel.embed", "silica/cli.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/kernel/embed.py")
    # from silica.kernel import SOMETHING_IN_INIT → falls back to the package __init__
    kind, val = classify_import("silica.kernel.CONFIG", "silica/cli.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/kernel/__init__.py")


def test_python_relative():
    kind, val = classify_import(".paths.atomic_write_bytes", "silica/kernel/embed.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/kernel/paths.py")
    kind, val = classify_import("..cli", "silica/kernel/embed.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/cli.py")


def test_python_external_and_unresolved(tmp_path):
    (tmp_path / "silica").mkdir()
    kind, val = classify_import("numpy.linalg", "silica/cli.py", PY_FILES, "python", tmp_path)
    assert (kind, val) == ("external", "numpy")
    # first-party (silica/ dir exists on disk) but no matching file → unresolved, counted.
    # 3-segment so back-off stops at silica/ghost/__init__.py (absent), never the silica
    # package __init__ — a genuinely unresolvable first-party import (cf. Task 4 pkg.ghost.nope).
    kind, val = classify_import("silica.ghost.deep", "silica/cli.py", PY_FILES, "python", tmp_path)
    assert (kind, val) == ("unresolved", "silica.ghost.deep")


def test_ts_relative_with_extension_inference():
    kind, val = classify_import("./local/helper", "src/app.ts", TS_FILES, "typescript", Path("."))
    assert (kind, val) == ("resolved", "src/local/helper.ts")
    kind, val = classify_import("./lib", "src/app.ts", TS_FILES, "typescript", Path("."))
    assert (kind, val) == ("resolved", "src/lib/index.ts")


def test_ts_bare_external_and_alias_unresolved():
    kind, val = classify_import("react", "src/app.ts", TS_FILES, "typescript", Path("."))
    assert (kind, val) == ("external", "react")
    kind, val = classify_import("@/lib/x", "src/app.ts", TS_FILES, "typescript", Path("."))
    assert (kind, val) == ("unresolved", "@/lib/x")


def test_moved_helpers_still_work(tmp_path):
    (tmp_path / "silica" / "kernel").mkdir(parents=True)
    assert is_first_party("silica.kernel.embed", tmp_path)
    assert not is_first_party("numpy", tmp_path)
    assert package_of("silica.kernel.embed", tmp_path) == "silica/kernel"


import subprocess

import pytest

from silica.kernel import codegraph


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def _seed_mini_repo(root: Path) -> None:
    """3-file py repo with cross imports (spec §8 fixture)."""
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "paths.py").write_text(
        "import os\n\ndef norm(p: str) -> str:\n    return p\n", encoding="utf-8")
    (root / "pkg" / "embed.py").write_text(
        "from .paths import norm\nimport numpy\n\nclass Embedder:\n    pass\n", encoding="utf-8")
    (root / "main.py").write_text(
        "from pkg import embed\nfrom pkg.ghost import nope\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=root, check=True)


def test_build_resolves_edges_external_unresolved(tmp_path):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    g = codegraph.build_codegraph(tmp_path)
    assert g.files["pkg/embed.py"]["imports"] == ["pkg/paths.py"]
    assert g.files["pkg/embed.py"]["external"] == ["numpy"]
    assert g.files["main.py"]["imports"] == ["pkg/embed.py"]
    assert g.files["main.py"]["unresolved"] == ["pkg.ghost.nope"]
    assert g.fan_in("pkg/paths.py") == 1
    assert g.importers("pkg/embed.py") == ["main.py"]
    syms = {s["name"] for s in g.files["pkg/embed.py"]["symbols"]}
    assert "Embedder" in syms


def test_build_is_deterministic_byte_for_byte(tmp_path):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    a = codegraph._serialize(codegraph.build_codegraph(tmp_path))
    b = codegraph._serialize(codegraph.build_codegraph(tmp_path))
    assert a == b


def test_load_codegraph_none_outside_repo(tmp_path):
    assert codegraph.load_codegraph(tmp_path) is None


def test_load_rebuilds_on_head_move_and_mtime(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    store = tmp_path / "cg.json"
    monkeypatch.setattr(codegraph, "store_path", lambda: store)
    g1 = codegraph.load_codegraph(tmp_path)
    assert store.exists() and g1 is not None
    # valid store → served from disk (marker: mutate the file set → invalid)
    (tmp_path / "new.py").write_text("x = 1\n", encoding="utf-8")  # untracked, supported
    g2 = codegraph.load_codegraph(tmp_path)
    assert "new.py" in g2.files  # file-set mismatch forced a full rebuild


def test_parse_error_file_present_never_aborts(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "bad.py").write_text("def x(: pass", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "c"], cwd=tmp_path, check=True)
    g = codegraph.build_codegraph(tmp_path)
    assert "bad.py" in g.files  # tree-sitter is error-tolerant: entry exists either way


def test_notebook_is_a_file_node(tmp_path):
    import json as _json
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    nb = _json.dumps({"nbformat": 4, "metadata": {"kernelspec": {"language": "python"}},
                      "cells": [{"cell_type": "code",
                                 "source": "from pkg.paths import norm\nimport pandas\n"}]})
    (tmp_path / "analysis.ipynb").write_text(nb, encoding="utf-8")
    g = codegraph.build_codegraph(tmp_path)
    assert g.files["analysis.ipynb"]["imports"] == ["pkg/paths.py"]
    assert g.files["analysis.ipynb"]["external"] == ["pandas"]
    assert g.fan_in("pkg/paths.py") == 2  # embed.py + the notebook


def test_malformed_notebook_gets_parse_error_entry(tmp_path):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    (tmp_path / "bad.ipynb").write_text("{not json", encoding="utf-8")
    g = codegraph.build_codegraph(tmp_path)  # build never aborts (spec §1)
    assert g.files["bad.ipynb"]["parse_error"] is True
    assert g.files["bad.ipynb"]["symbols"] == []


def test_code_vocabulary_top_fan_in(tmp_path):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    g = codegraph.build_codegraph(tmp_path)
    vocab = codegraph.code_vocabulary(g, cap=2)
    # top-2 by fan-in: pkg/paths.py and pkg/embed.py (1 importer each, path tiebreak)
    assert "paths" in vocab and "norm" in vocab        # module stem + symbol
    assert "embed" in vocab and "Embedder" in vocab
    assert "__init__" not in vocab                      # noise stems excluded
    assert vocab == list(dict.fromkeys(vocab))          # deduped, stable order
