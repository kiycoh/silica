# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica mcp` — stdio MCP server exposing the tool registry to external agents.

Any MCP client (Claude Code first) gets Silica as vault memory: context
search through the relatedness facade, note reading, and gated writing.

Default surface is CORE_TOOLS — the subset (search + read +
write) — because every exposed schema is context the client pays for on each
session. `--all` exposes the full default toolset (same sensitive/internal
filter as the chat agent's loop).

stdout is the protocol channel: nothing here may print to it. Logging goes
to stderr (wired by the `silica mcp` dispatch in cli.py).

Only stdlib at import time — the `mcp` SDK is imported inside run_mcp so the
module loads without the [mcp] extra.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Search + read + single-note write: the surface a coding agent needs to use
# the vault as memory. Everything else (pipelines, batches, taxonomy, graph
# exports) stays behind --all.
CORE_TOOLS = (
    "silica_semantic_search",
    "silica_similar",
    "silica_related",
    "silica_search",
    "silica_search_context",
    "silica_read_note",
    "silica_outline",
    "silica_links",
    "silica_props",
    "silica_files",
    "silica_exists",
    "silica_write_note",
    "silica_patch_note",
)


def exposed_tools(all_tools: bool = False) -> dict[str, Any]:
    """The registry slice served over MCP: Tool objects keyed by name."""
    # Registration side effect — same module set as cli.py.
    import silica.tools.atomic  # noqa: F401
    import silica.tools.composed  # noqa: F401
    import silica.tools.wrapped  # noqa: F401
    import silica.tools.codedocs_tool  # noqa: F401
    import silica.tools.delegate_tool  # noqa: F401
    from silica.tools import TOOLS

    allowed = {n: t for n, t in TOOLS.items() if not t.sensitive and not t.internal}
    if all_tools:
        return allowed
    return {n: allowed[n] for n in CORE_TOOLS}  # KeyError = registry drift, fail loud


def run_mcp(all_tools: bool = False) -> int:
    """Serve the tool registry over MCP stdio. Blocks until the client hangs up."""
    try:
        import anyio
        import mcp.types as types
        from mcp.server.lowlevel import Server
        from mcp.server.stdio import stdio_server
    except ImportError:
        print(
            "silica mcp needs the [mcp] extra: uv pip install 'silica[mcp]'",
            file=sys.stderr,
        )
        return 1

    tools = exposed_tools(all_tools)
    server = Server("silica")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.json_schema()["function"]["parameters"],
            )
            for t in tools.values()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        t = tools.get(name)
        if t is None:
            raise ValueError(f"Unknown tool: {name}")
        # Tool.run validates args via pydantic and always returns a JSON string
        # (errors included) — exactly what a text content block wants.
        out = await anyio.to_thread.run_sync(lambda: t.run(**(arguments or {})))
        return [types.TextContent(type="text", text=out)]

    async def _serve() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    logger.info("silica mcp: serving %d tools on stdio", len(tools))
    anyio.run(_serve)
    return 0
