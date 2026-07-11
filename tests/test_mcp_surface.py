"""The MCP surface must stay a valid slice of the tool registry."""
from __future__ import annotations

import json
import re
from pathlib import Path

from silica.ui.mcp import CORE_TOOLS, exposed_tools

ROOT = Path(__file__).resolve().parent.parent


def test_core_tools_resolve_and_are_agent_visible():
    core = exposed_tools()
    assert set(core) == set(CORE_TOOLS)
    for t in core.values():
        assert not t.internal and not t.sensitive
        # every exposed tool must yield a servable JSON schema
        params = t.json_schema()["function"]["parameters"]
        assert params.get("type") == "object"


def test_all_surface_matches_agent_loop_filter():
    from silica.tools import TOOLS

    exposed = exposed_tools(all_tools=True)
    expected = {n for n, t in TOOLS.items() if not t.sensitive and not t.internal}
    assert set(exposed) == expected


def test_skill_references_only_core_tools():
    # The Claude skill teaches the default MCP surface — a tool name in the
    # skill that isn't in CORE_TOOLS is drift (renamed, or never exposed).
    skill = (ROOT / "skills" / "silica" / "SKILL.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"silica_\w+", skill))
    unknown = referenced - set(CORE_TOOLS)
    assert not unknown, f"SKILL.md references tools outside the MCP core surface: {unknown}"


def test_plugin_manifest_launches_silica_mcp():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    args = plugin["mcpServers"]["silica"]["args"]
    assert args[-2:] == ["silica", "mcp"]
    marketplace = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert marketplace["plugins"][0]["name"] == plugin["name"]
