"""Shared pytest fixtures for the silica-agent test suite."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _fresh_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the global BUS singleton for every test to prevent cross-test contamination."""
    import silica.agent.bus as bus_mod
    monkeypatch.setattr(bus_mod, "BUS", bus_mod.EventBus())


@pytest.fixture(autouse=True)
def _no_recon_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable silica_recon's network embedder by default: recon falls back to the
    deterministic YAKE rank. Keeps the suite fast and offline; the rerank path is
    covered by test_keyphrase (FakeEmbedder) and the SILICA_EVAL golden eval."""
    import silica.tools.pipeline as pipe_mod
    monkeypatch.setattr(pipe_mod, "_recon_embedder", lambda: None)


@pytest.fixture(autouse=True)
def _isolate_embed_legacy_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against the real ~/.silica/index/embeddings.json leaking into tests
    via the legacy-migration fallback. Any test that redirects _index_path to a
    non-existent tmp file would otherwise fall back to the developer's real index."""
    import silica.kernel.embed as embed_mod
    monkeypatch.setattr(embed_mod, "_LEGACY_INDEX_PATH", tmp_path / "legacy_embed.json")


@pytest.fixture(autouse=True)
def _isolate_cooccurrence_index(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the default co-occurrence index to a per-test tmp path.

    The post-write freshness hook refreshes the co-occurrence index with no
    embedder gate (it is the embedder-free stable leg), so any test that drives
    the write handler would otherwise write the user's real
    ~/.silica/index/cooccurrence.json. Tests that need a store pass an explicit
    path; this only redirects the default.
    """
    import silica.kernel.cooccurrence as cooc_mod
    monkeypatch.setattr(cooc_mod, "_index_path", lambda: tmp_path / "cooccurrence_index.json")
    monkeypatch.setattr(cooc_mod, "_LEGACY_INDEX_PATH", tmp_path / "legacy_cooc.json")


@pytest.fixture(autouse=True)
def _reset_overlay_cache() -> None:
    """Reset the module-level overlay cache before every test.

    Prevents a test that calls get_active_overlay() (or monkeypatches the vault
    path) from polluting the cached result seen by subsequent tests.
    """
    import silica.kernel.overlay as overlay_mod
    overlay_mod.reset_overlay_cache()


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
