# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Per-phase state handlers for the Injector FSM.

Each module holds the handler bodies for one pipeline phase group;
InjectorFSM in orchestrator.py is the wiring that dispatches to them.
"""
from silica.router.states import collision, distill, finalize, linking, setup, write  # noqa: F401
