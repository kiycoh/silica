# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Optional localhost GUI (`silica --gui`). Requires the `[gui]` extra."""


# Lazy so importing sibling modules (e.g. graph_view) does not drag in the
# server and its FastAPI dependency — /graph must work on a base install
# without the [gui] extra. `from silica.ui.web import serve` still works (PEP 562).
def __getattr__(name):
    if name == "serve":
        from .server import serve
        return serve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["serve"]
