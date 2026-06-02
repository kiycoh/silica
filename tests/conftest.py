"""Shared pytest fixtures for the silica-agent test suite."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _fresh_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the global BUS singleton for every test to prevent cross-test contamination."""
    import silica.agent.bus as bus_mod
    monkeypatch.setattr(bus_mod, "BUS", bus_mod.EventBus())


@pytest.fixture(scope="session")
def synthetic_vault() -> Path:
    """Return the path to the synthetic test vault, building it if needed.

    Session-scoped: built exactly once per pytest run.
    Location: tests/fixtures/synthetic_vault/ (or SILICA_TEST_VAULT env var).
    """
    from tests.fixtures.vault_factory import build_synthetic_vault, _resolve_root
    return build_synthetic_vault(_resolve_root())
