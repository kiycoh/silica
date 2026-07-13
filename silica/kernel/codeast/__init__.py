# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codeast — native shallow AST skeleton extraction (ADR-0012).

Package layout: base (dataclasses, extension map, dispatch, shared helpers,
structural diff), python / ts (per-language walkers). This façade re-exports
the full historical module surface; consumers import from
silica.kernel.codeast only (precedent: graph_report/).
"""
from silica.kernel.codeast.base import (
    BARE_LANGUAGES,
    EXTENSION_MAP,
    Call,
    ModuleSkeleton,
    Symbol,
    diff_skeletons,
    extract_skeleton,
    language_for,
)

__all__ = [
    "BARE_LANGUAGES",
    "EXTENSION_MAP",
    "Call",
    "ModuleSkeleton",
    "Symbol",
    "diff_skeletons",
    "extract_skeleton",
    "language_for",
]
