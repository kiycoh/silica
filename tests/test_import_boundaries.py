"""The consolidated capabilities boundary, enforced by import-linter.

The contracts live in pyproject.toml ([tool.importlinter]); this test runs
``lint-imports`` so that ``uv run pytest`` is the enforcement point (there is
no CI). Prior art: the relatedness facade boundary test.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_linter_contracts_hold():
    exe = shutil.which("lint-imports")
    assert exe, "import-linter is not installed — run: uv sync --extra dev"
    proc = subprocess.run(
        [exe],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "import-linter contracts violated:\n" + proc.stdout + proc.stderr
    )
