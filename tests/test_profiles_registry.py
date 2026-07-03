from __future__ import annotations

from silica.capabilities.profile import WorkerResult
from silica.capabilities.profiles_builtin import READER, ROUTER


READONLY_TOOLS = {
    "silica_read_note",
    "silica_search",
    "silica_search_context",
    "silica_outline",
    "silica_links",
}


def test_reader_and_router_registered():
    from silica.capabilities import CAPABILITIES

    assert "reader" in CAPABILITIES
    assert "router" in CAPABILITIES


def test_profiles_are_read_only():
    for p in (READER, ROUTER):
        assert p.bounds_factory is None
        assert set(p.tools).issubset(READONLY_TOOLS), f"{p.name} exposes non-readonly tools"


def test_reader_parser_returns_digest():
    r = READER.result_parser("here is the gathered context", [])
    assert isinstance(r, WorkerResult)
    assert r.status == "ok"
    assert "gathered context" in str(r.output)


def test_router_parser_returns_decision():
    r = ROUTER.result_parser('{"decision": "patch", "target": "ROS"}', [])
    assert isinstance(r, WorkerResult)
    assert r.status == "ok"
    assert r.output["decision"] == "patch"
    assert r.output["target"] == "ROS"


def test_router_parser_tolerates_nonjson():
    r = ROUTER.result_parser("I think this should create a new note", [])
    # Non-JSON final text must not crash; it degrades to a no_op decision.
    assert r.status in ("no_op", "ok")
