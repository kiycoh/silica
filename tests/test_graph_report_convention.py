"""Convention test for silica/kernel/graph_report/.

kernel/graph_report/ is covered by the import-linter contract "graph
structure is deterministic — structural modules never import agent
(P2/DKB)" (pyproject.toml): structural modules there must never import
silica.agent. That contract's source_modules is the authoritative list of
which graph_report modules are structural.

This test enumerates the package on disk and checks every module is
accounted for: either listed in the contract's source_modules, or
explicitly exempted here with a rationale. A new .py file dropped into
graph_report/ without either fails this test — forcing a decision instead
of silently slipping past P2/DKB.

Style precedent: tests/test_relatedness_boundary.py (allowlist-vs-disk
boundary test).
"""
from __future__ import annotations

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_REPORT_DIR = REPO_ROOT / "silica" / "kernel" / "graph_report"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

CONTRACT_NAME = (
    "graph structure is deterministic — structural modules never import agent (P2/DKB)"
)

# Modules present in kernel/graph_report/ but deliberately NOT in the
# contract's source_modules, each with a written rationale (mirrors the
# pyproject.toml comment for this same contract):
EXEMPT = {
    "embed_signals",  # embedding overlay via get_embedder — legitimate, per pyproject comment
    "__init__",        # facade re-export, not a structural module itself
}


def _covered_stems() -> set[str]:
    """Read source_modules for CONTRACT_NAME straight from pyproject.toml
    (no mirrored list to drift; also catches someone shrinking it there)."""
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    contracts = data["tool"]["importlinter"]["contracts"]
    for contract in contracts:
        if contract.get("name") == CONTRACT_NAME:
            prefix = "silica.kernel.graph_report."
            return {
                mod[len(prefix):]
                for mod in contract["source_modules"]
                if mod.startswith(prefix)
            }
    raise AssertionError(f"import-linter contract not found in pyproject.toml: {CONTRACT_NAME!r}")


def test_graph_report_modules_are_covered_or_exempt():
    covered = _covered_stems()
    actual_stems = {p.stem for p in GRAPH_REPORT_DIR.glob("*.py")}
    assert actual_stems == covered | EXEMPT, (
        "silica/kernel/graph_report/*.py drifted from the import-linter contract "
        f"{CONTRACT_NAME!r} + this test's EXEMPT set.\n"
        f"On disk but unaccounted for: {sorted(actual_stems - covered - EXEMPT)}. "
        "Add to source_modules in pyproject.toml (and covered here), or add to "
        "EXEMPT with a one-line rationale.\n"
        f"Accounted for but missing on disk: {sorted((covered | EXEMPT) - actual_stems)}. "
        "Remove the stale entry."
    )
