# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Silica — Obsidian-native agentic CLI."""
from __future__ import annotations

try:
    # Written by setuptools-scm at build/install time (gitignored).
    from ._version import version as __version__
except ImportError:  # pragma: no cover - source tree without a build
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("silica")
    except (ImportError, PackageNotFoundError):  # pragma: no cover
        __version__ = "0.0.0+unknown"
