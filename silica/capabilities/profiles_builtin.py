"""Built-in worker profiles registered into PROFILES on import.

Phase A ships two read-only profiles:
  * reader — gathers context, returns a digest (orchestrator's retrieval relief).
  * router — adjudicates a high-sim patch-vs-new-note decision (ADR-0003). Its Op
    consumer lands in Phase C; here it parses a {decision,target} dict.
"""
from __future__ import annotations

import orjson

from silica.capabilities.profile import WorkerProfile, WorkerResult, PROFILES


def _reader_parser(final_text: str, trace: list[dict]) -> WorkerResult:
    return WorkerResult(status="ok", output=final_text, detail=f"{len(trace)} reads")


def _router_parser(final_text: str, trace: list[dict]) -> WorkerResult:
    try:
        data = orjson.loads(final_text)
        if isinstance(data, dict) and "decision" in data:
            return WorkerResult(status="ok", output=data)
    except Exception:
        pass
    return WorkerResult(status="no_op", output={"decision": "unknown", "raw": final_text})


READER = WorkerProfile(
    name="reader",
    tools=("silica_read_note", "silica_search", "silica_search_context", "silica_outline"),
    bounds_factory=None,
    max_iterations=4,
    system_prompt=(
        "You are a read-only retrieval worker. Gather the context relevant to the "
        "goal from the vault using the provided tools, then reply with a concise "
        "digest. Do not speculate; cite note names you read."
    ),
    result_parser=_reader_parser,
)

ROUTER = WorkerProfile(
    name="router",
    tools=("silica_read_note", "silica_search", "silica_outline", "silica_links"),
    bounds_factory=None,
    max_iterations=4,
    system_prompt=(
        "You are a routing adjudicator. A chunk has a high embedding similarity to "
        "candidate note(s). Decide whether the chunk truly belongs in a candidate "
        "(decision='patch', target=<note name>) or is a false positive that needs a "
        "new note (decision='write'). Decide WHERE, never WHETHER — never skip a "
        "chunk. Reply with a single JSON object: {\"decision\": \"patch\"|\"write\", "
        "\"target\": <note name or null>}."
    ),
    result_parser=_router_parser,
)

PROFILES["reader"] = READER
PROFILES["router"] = ROUTER
