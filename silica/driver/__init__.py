# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Obsidian Driver package — exposes the global DRIVER instance.

The backend is selected at import time based on CONFIG.backend (from config.py),
which reads the SILICA_BACKEND environment variable:
  - "fs" (default): ObsidianFSBackend — direct filesystem access, headless, no Obsidian required
  - "cli": ObsidianCLIBackend — wraps the official Obsidian CLI (adds version-history
    rollback for patch ops, live metadata-cache reads, and user link-format preference
    in autolink; requires the Obsidian desktop app to be running)

Usage:
    from silica.driver import DRIVER
    content = DRIVER.read_note("Computer Vision")
"""
from __future__ import annotations

import logging
import threading

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
    elif CONFIG.backend == "ws":
        raise ValueError(
            "backend='ws' is installed by `silica connect` when the plugin dials in; "
            "run `silica connect`, or set SILICA_BACKEND to fs/cli"
        )
    else:
        raise ValueError(f"Unknown backend: {CONFIG.backend!r}")


# Lazy initialization — created on first access, protected by lock for thread safety
_driver: ObsidianDriver | None = None
_driver_lock = threading.Lock()


def get_driver() -> ObsidianDriver:
    """Get the global driver instance (lazy-initialized, thread-safe)."""
    global _driver
    if _driver is None:
        with _driver_lock:
            # Double-checked locking: recheck after acquiring lock
            if _driver is None:
                _driver = _create_driver()
    return _driver


def reset_driver() -> None:
    """Drop the cached driver so the next get_driver() rebuilds against the
    current CONFIG.vault_path / CONFIG.backend. Used by the runtime /vault
    switch."""
    global _driver
    with _driver_lock:
        _driver = None


def set_driver(driver: ObsidianDriver | None) -> None:
    """Install a live driver instance — `silica connect`'s attached ws backend.
    None falls back to building from CONFIG.backend on next access."""
    global _driver
    with _driver_lock:
        _driver = driver


# For convenience: DRIVER can be imported directly
# But since it's lazy, access via get_driver() in hot paths
class _DriverProxy:
    """Proxy that lazy-initializes the driver on first attribute access."""

    def __getattr__(self, name: str):
        return getattr(get_driver(), name)


DRIVER = _DriverProxy()
