# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Shared fixtures for the benchmark tests moved out of tests/eval/.

These files live outside the `tests/` tree (they are slow, not in the default
`testpaths`), so pytest's upward conftest discovery no longer reaches
tests/conftest.py — re-export the fixtures they still use.
"""
from tests.conftest import tmp_vault  # noqa: F401
