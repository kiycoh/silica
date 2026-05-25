"""L2 Workers — semantic sub-agents, stateless, CoT-intensive."""
from __future__ import annotations

from silica.tools import TOOLS, Tool

WORKER_BLOCKED_CLASSES = frozenset({"composed", "wrapped"})
BLOCKED_TOOL_NAMES = frozenset({
    "silica_run_injector",
    "silica_bulk_write",
    "silica_move",
    "silica_delete",
    "silica_snapshot",
    "silica_restore",
    "silica_cleanup",
})


def build_worker_toolset() -> dict[str, Tool]:
    """Filters the global tool registry to return only read-only atomic tools.

    Excludes composed and wrapped classes, and explicitly blocks mutation tools.
    """
    allowed_tools = {}
    for name, tool in TOOLS.items():
        if tool.cls in WORKER_BLOCKED_CLASSES:
            continue
        if name in BLOCKED_TOOL_NAMES:
            continue
        allowed_tools[name] = tool
    return allowed_tools
