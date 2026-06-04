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


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    """Provide a temporary filesystem-backed vault for unit tests.

    Returns a helper with:
      .note(rel, content="") -> str   — create a note, return absolute path
      .read(path) -> str              — read note at absolute path
      .write(path, content)           — overwrite note at absolute path
    """
    import silica.config
    import silica.driver

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    monkeypatch.setattr(silica.config.CONFIG, "backend", "fs")
    monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault_dir))
    silica.driver._driver = None  # reset lazy singleton

    class _VaultHelper:
        def note(self, rel: str, content: str = "") -> str:
            p = vault_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return str(p)

        def read(self, path: str) -> str:
            from pathlib import Path as _Path
            return _Path(path).read_text(encoding="utf-8")

        def write(self, path: str, content: str) -> None:
            from pathlib import Path as _Path
            _Path(path).write_text(content, encoding="utf-8")

    yield _VaultHelper()
    silica.driver._driver = None  # reset after test
