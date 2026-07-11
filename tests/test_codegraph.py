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
