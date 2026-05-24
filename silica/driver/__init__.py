"""Obsidian Driver package — exposes the global DRIVER instance.

The backend is selected at import time based on SILICA_BACKEND env var:
  - "cli" (default): ObsidianCLIBackend — wraps the official Obsidian CLI
  - "fs": (future) filesystem backend for headless/oracle mode

Usage:
    from silica.driver import DRIVER
    content = DRIVER.read_note("Computer Vision")
"""
from __future__ import annotations

import logging

from silica.driver.base import (  # noqa: F401 — re-export domain types
    GraphSnapshot,
    Heading,
    Hit,
    Link,
    NoteContent,
    NoteRef,
    ObsidianDriver,
    Txn,
)

logger = logging.getLogger(__name__)


def _create_driver() -> ObsidianDriver:
    """Create the appropriate driver backend based on config."""
    from silica.config import CONFIG

    if CONFIG.backend == "cli":
        from silica.driver.cli_backend import ObsidianCLIBackend

        return ObsidianCLIBackend(vault_name=CONFIG.vault_name)
    elif CONFIG.backend == "fs":
        from silica.driver.fs_backend import ObsidianFSBackend

        return ObsidianFSBackend(vault_path=CONFIG.vault_path)
    else:
        raise ValueError(f"Unknown backend: {CONFIG.backend!r}")


# Lazy initialization — created on first access
_driver: ObsidianDriver | None = None


def get_driver() -> ObsidianDriver:
    """Get the global driver instance (lazy-initialized)."""
    global _driver
    if _driver is None:
        _driver = _create_driver()
    return _driver


# For convenience: DRIVER can be imported directly
# But since it's lazy, access via get_driver() in hot paths
class _DriverProxy:
    """Proxy that lazy-initializes the driver on first attribute access."""

    def __getattr__(self, name: str):
        return getattr(get_driver(), name)


DRIVER = _DriverProxy()
