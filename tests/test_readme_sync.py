"""README must track the real command surface and never link into gitignored docs/."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from silica.ui.commands import COMMANDS

ROOT = Path(__file__).resolve().parent.parent


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_every_registry_command_is_documented():
    readme = _readme()
    missing = [c.name for c in COMMANDS if c.name not in readme]
    assert not missing, f"README.md is missing commands: {missing}"


def test_no_markdown_links_into_gitignored_docs():
    # docs/ is gitignored — any README link into it is broken for every clone.
    offending = re.findall(r"\]\(docs/[^)]*\)", _readme())
    assert not offending, f"README.md links into gitignored docs/: {offending}"


def test_pyproject_readme_file_exists():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme_rel = data["project"]["readme"]
    assert (ROOT / readme_rel).is_file(), f"pyproject readme points to missing {readme_rel}"
