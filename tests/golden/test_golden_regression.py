# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The golden regression gate.

Env-gated (``SILICA_EVAL=1``, existing convention) — never runs in a bare
``pytest``. Refuses to compare across a drifted vault or a changed cooccur mode;
otherwise re-runs the cheap-tier probes and fails on any gated metric drop.

  SILICA_EVAL=1 SILICA_VAULT=~/Documents/Obsidian/test uv run pytest \
      tests/golden/test_golden_regression.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("SILICA_EVAL"),
    reason="golden regression gate — set SILICA_EVAL=1 (+ SILICA_VAULT) to run",
)


def test_golden_regression():
    import silica.driver
    from tests.eval.golden import runner

    baseline_path = Path(runner.__file__).parent / "baseline.json"
    if not baseline_path.exists():
        pytest.fail(
            "no baseline.json — freeze one with: "
            "uv run python -m tests.eval.golden --vault <v> --freeze-baseline"
        )
    baseline = json.loads(baseline_path.read_text())

    vault = runner.resolve_vault(None)  # honors SILICA_VAULT

    # Refuse BEFORE running — a drifted corpus is a deliberate re-baseline, not a regression.
    digest, _notes = runner.vault_digest(vault)
    if digest != baseline["vault"]["digest"]:
        pytest.fail("vault drifted — re-baseline deliberately with --freeze-baseline")

    try:
        doc = runner.collect(vault, tier="cheap")
    finally:
        silica.driver._driver = None

    if doc["config"]["cooccur_store"] != baseline["config"]["cooccur_store"]:
        pytest.fail("cooccur mode changed — re-baseline deliberately with --freeze-baseline")

    fails = runner.compare(baseline, doc)
    assert not fails, "gated metric regression:\n" + "\n".join(fails)
