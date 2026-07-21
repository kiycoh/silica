# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""LoCoMo adapter — multi-session two-speaker conversational QA, LLM-judged.

Reuses the LongMemEval pipeline (tests/eval/longmemeval/runner.py) wholesale:
vault binding, distiller ingest + episodic capture, index build, the
perception product path (fused facade + rerank + query-densest windows +
facts-first episodic block), the judge rubric, and aggregate(). What differs
is the dataset shape (snap-research/locomo, locomo10.json):

  * ONE conversation (up to ~35 sessions between two NAMED speakers) is
    shared by all its questions -> one vault per conversation, built and
    indexed once; every question perceives against it. (LongMemEval builds a
    per-question haystack instead.)
  * turns carry speaker names and optional photos; rendering keeps the name
    and the ``blip_caption`` — questions reference speakers by name and
    photo content.
  * gold evidence is per-dialog ("D3:12"); dia-ids map to their session, so
    session_recall is the same metric as LongMemEval's.
  * categories: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop,
    5 adversarial. Category 5 is unanswerable — scored with the LongMemEval
    abstention rubric against the ``adversarial_answer``. Category 2
    deliberately does NOT get the LME off-by-one temporal leniency: the base
    rubric keeps the number conservative and mem0/Zep-comparable.
  * LoCoMo has no per-question date; "today" is the LAST session's date
    (every question is asked after the conversation ended).

Per-conversation pipeline (retrieval-then-answer, the config comparable 1:1
with Mem0 / Zep):

  load (verbatim or --distill through the Silica distiller + episodic
  capture) -> index (cooccur offline; embeddings when an embedder is served)
  -> per question: perceive -> answer -> judge.

  uv run python -m tests.eval.locomo \
      --data locomo10.json --run-root bench/locomo --distill --verbose

Requires an LLM for answer + judge (litellm, via CONFIG.model). The
personal-memory lane (ADR-0019) is never passed — only the conversation's own
sessions can be recalled.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
from pathlib import Path

# Shared pipeline — the LoCoMo adapter owns dataset shape only; every Silica
# seam (vault lifecycle, distiller, indexes, judge, metrics) is the LME one.
from tests.eval.longmemeval.runner import (
    _ephemeral_hit,
    _gold_in_context,
    _session_rel,
    _write_note,
    aggregate,
    bind_vault,
    build_indexes,
    distill_session,
    judge,
    question_vault,
)

logger = logging.getLogger(__name__)

METRICS_PATH = Path(__file__).parent / "metrics.json"

_CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain",
             4: "single-hop", 5: "adversarial"}
_ADVERSARIAL = 5
_DIA_RE = re.compile(r"\bD(\d+):")
_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
# "1:56 pm on 8 May, 2023" (occasionally date-only).
_DT_FMTS = ("%I:%M %p on %d %B, %Y", "%d %B, %Y")


# --- Dataset shape -----------------------------------------------------------

def parse_date_time(raw: str) -> str:
    """LoCoMo session date_time -> ISO date; '' when unparseable (the note
    still loads — only temporal reasoning degrades for that session)."""
    for fmt in _DT_FMTS:
        try:
            return datetime.datetime.strptime((raw or "").strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def render_turn(t: dict) -> str:
    """One dialog turn -> 'Speaker: text [shares a photo: caption]'. Photo-only
    turns keep the caption line — evidence dia-ids can point at photos."""
    parts = [(t.get("text") or "").strip()]
    cap = (t.get("blip_caption") or "").strip()
    if cap:
        parts.append(f"[shares a photo: {cap}]")
    return f"{t.get('speaker', '?')}: {' '.join(p for p in parts if p)}"


def render_session(turns: list[dict]) -> str:
    return "\n\n".join(render_turn(t) for t in turns)


def conversation_sessions(conv: dict) -> list[tuple[int, str, list[dict]]]:
    """[(session_number, raw date_time, turns)] in chronological order."""
    out = []
    for key, turns in conv.items():
        m = _SESSION_KEY_RE.match(key)
        if m and isinstance(turns, list):
            n = int(m.group(1))
            out.append((n, conv.get(f"session_{n}_date_time", ""), turns))
    return sorted(out)


def evidence_sessions(evidence: list) -> set[str]:
    """Gold dia-ids ('D3:12') -> their sessions ({'session_3'})."""
    return {f"session_{n}" for e in evidence or []
            for n in _DIA_RE.findall(str(e))}


def _note(session_id: str, date: str, date_time: str, body: str) -> str:
    return (
        "---\n"
        f"session_id: {json.dumps(session_id, ensure_ascii=False)}\n"
        f"date: {json.dumps(date, ensure_ascii=False)}\n"
        f"date_time: {json.dumps(date_time, ensure_ascii=False)}\n"
        "source: locomo\n"
        "tags:\n  - benchmark\n"
        "AI: true\n"
        "---\n\n"
        f"{body}\n"
    )


# --- Conversation vault ------------------------------------------------------

def load_conversation_vault(vault: Path, inst: dict, distill: bool = False,
                            reuse: bool = False) -> dict[str, dict]:
    """Write one note per session; return {rel: {session_id, date}}.

    Same contract as the LME loader: ``distill`` routes each session through
    the Silica distiller (episodic capture keyed run_id=session, seen=date);
    ``reuse`` adopts an already-populated vault as-is (frozen corpus — the
    distiller re-rolls every note per run, which confounds cross-run A/Bs)."""
    sess_dir = vault / "sessions"
    if reuse and sess_dir.is_dir():
        existing = sorted(sess_dir.glob("s*.md"))
        if existing:
            from silica.kernel import frontmatter

            index = {}
            for f in existing:
                data, _raw, _body = frontmatter.split(f.read_text(encoding="utf-8"))
                index[f"sessions/{f.stem}"] = {"session_id": data.get("session_id", ""),
                                               "date": data.get("date", "")}
            return index
    sess_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for i, (n, date_time, turns) in enumerate(conversation_sessions(inst["conversation"])):
        rel = _session_rel(i)
        sid = f"session_{n}"
        date = parse_date_time(date_time)
        excerpt = render_session(turns)
        body = distill_session(sid, date, excerpt) if distill else excerpt
        _write_note(rel, _note(sid, date, date_time, body))
        index[rel.removesuffix(".md")] = {"session_id": sid, "date": date}
    return index


def _conv_now(index: dict[str, dict]) -> str:
    """'Today' for a conversation = its last session's date (ISO sorts)."""
    return max((e["date"] for e in index.values() if e.get("date")), default="")


def build_timeline_seed(vault: Path) -> str:
    """Chronological index over a vault's dated notes -> agent system seed.

    Pure, LLM-free overlay for the distill+timeline experiment (spec
    2026-07-20): read each note's ``date``/``session_id`` frontmatter, sort by
    date ascending (NOT file order — filename order need not match dates), and
    emit one line per note pointing at it by the identifier silica_read_note
    resolves (its filename stem). Undated notes are EXCLUDED: a note with no
    date has no place on a chronology, and 'end of list' would read as
    most-recent — wrong. Returns "" when nothing is dated (seed suppressed)."""
    from silica.kernel import frontmatter

    rows = []
    for f in sorted(vault.rglob("*.md")):
        data, _raw, _body = frontmatter.split(f.read_text(encoding="utf-8"))
        date = (data or {}).get("date")
        if not date:
            continue
        label = (data or {}).get("session_id") or f.stem
        rows.append((str(date), str(label), f.stem))
    if not rows:
        return ""
    rows.sort(key=lambda r: (r[0], r[2]))   # date asc; stem tie-break for determinism
    lines = ["## Timeline (session chronology)",
             "Consult once to order events; read the linked note for detail.", ""]
    for i, (date, label, stem) in enumerate(rows, 1):
        lines.append(f"{i:2d}. {date}  -> {label:<11} ({stem}.md)")
    return "\n".join(lines)


# --- FSM ingest (e2e write path) --------------------------------------------
# --ingest fsm: each session goes through the product Coordinator (collision,
# dedup, deferred, anneal all live), sequentially in chronological order,
# with seen_override carrying the session's historical date. Vault freeze:
# ingest once, persist a marker, reuse only complete ingests.

_INBOX_RE = re.compile(r"^session_(\d+)\.md$")


def _clear_fsm_state() -> None:
    """Vault-keyed singletons beyond bind_vault's: deferred store cache and
    overlay/manifest caches (the FSM write path touches all of them)."""
    import silica.kernel.deferred as deferred_mod
    from silica.kernel.overlay import reset_overlay_cache
    from silica.kernel.vault_manifest import reset_manifest_cache

    deferred_mod._stores.clear()
    reset_overlay_cache()
    reset_manifest_cache()


def _wipe_index_namespace() -> None:
    """Drop this vault's ~/.silica/index/<digest>/ (embeddings, cooccur,
    deferred bundles) so a from-scratch re-ingest starts clean."""
    import shutil

    from silica.kernel import paths as kpaths

    shutil.rmtree(kpaths.index_dir(), ignore_errors=True)


def _ingest_failed(result: dict) -> bool:
    # Hard failure only: ERROR-state runs (context["error"]) or exceptions.
    # "partial" is NOT failure: a contained chunk's ops land in the deferred
    # store (write.py defers on lint/write rejection) and the anneal step
    # recovers them — legitimate product state, recorded as partial_sessions
    # in the marker and diagnosable via the report's anneal/still_deferred.
    return bool(result.get("error"))


def _ingest_partial(result: dict) -> bool:
    return bool(result.get("has_partial_failure")) \
        or result.get("final_status") == "partial"


def fsm_ingest_conversation(vault: Path, inst: dict, *, reuse: bool,
                            key_schema: bool) -> dict | None:
    """Ingest every session through Coordinator, then anneal. Returns the
    freeze marker (with ``reused`` flag), or None when the conversation failed
    ingest twice and must be EXCLUDED from metrics (a declared hole beats a
    false number). Assumes bind_vault(vault) was already called."""
    from silica.router import coordinator as coord_mod

    marker_path = vault / "fsm_ingest.json"
    runs_path = vault / "fsm_runs.json"
    sessions = conversation_sessions(inst["conversation"])
    sids = [f"session_{n}" for n, _, _ in sessions]

    if reuse and marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            marker = {}
        if marker.get("complete") and marker.get("sessions") == sids:
            marker["reused"] = True
            return marker
        logger.warning("fsm marker stale or partial for %s — re-ingesting from scratch",
                       vault.name)

    # Never reuse a partial ingest: wipe the vault and its index namespace.
    import shutil
    shutil.rmtree(vault, ignore_errors=True)
    vault.mkdir(parents=True, exist_ok=True)
    if key_schema:
        (vault / "vault.yaml").write_text(
            "conventions:\n  episodic_keys: {}\n", encoding="utf-8")
    _wipe_index_namespace()
    _clear_fsm_state()

    (vault / "inbox").mkdir(exist_ok=True)
    run_map: dict[str, str] = {}
    partial_sids: list[str] = []
    for n, date_time, turns in sessions:
        sid = f"session_{n}"
        date = parse_date_time(date_time)
        (vault / "inbox" / f"{sid}.md").write_text(
            _note(sid, date, date_time, render_session(turns)), encoding="utf-8")
        result: dict | None = None
        run_id = ""
        for attempt in (1, 2):
            try:
                coord = coord_mod.Coordinator(
                    inbox_files=[f"inbox/{sid}.md"], target_dir="memory",
                    seen_override=date or None)
                run_id = coord.fsm.progress.run_id
                result = coord.run()
            except Exception as e:
                logger.warning("Coordinator crashed on %s/%s (attempt %d): %s",
                               vault.name, sid, attempt, e)
                result = None
            if result is not None and not _ingest_failed(result):
                break
            if result is not None:
                logger.warning("Coordinator run failed on %s/%s (attempt %d): "
                               "final_status=%s error=%s", vault.name, sid, attempt,
                               result.get("final_status"), result.get("error"))
        else:
            logger.error("conversation %s: session %s failed ingest twice — "
                         "conversation EXCLUDED from metrics", vault.name, sid)
            return None
        if _ingest_partial(result):
            partial_sids.append(sid)
        run_map[run_id] = sid
        runs_path.write_text(json.dumps(run_map, indent=2), encoding="utf-8")

    # Grain-boundary recovery is part of the mechanisms under measurement.
    from silica.tools import pipeline as pipeline_mod
    anneal = pipeline_mod.silica_anneal(steer=True)

    marker = {
        "complete": True,
        "reused": False,
        "sessions": sids,
        "partial_sessions": partial_sids,
        "date": datetime.date.today().isoformat(),
        "anneal": {k: anneal.get(k) for k in ("bundles", "written", "still_deferred")},
    }
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return marker


def _provenance_session_map(vault: Path) -> dict[str, set[str]]:
    """note rel (no .md) -> contributing session ids, via provenance records
    keyed by inbox basename session_<n>.md. A note merged from 3 sessions
    counts for all 3; notes with no record count for no session."""
    from silica.kernel.provenance import read_records

    out: dict[str, set[str]] = {}
    for rec in read_records(vault_path=str(vault)):
        m = _INBOX_RE.match(rec.get("source") or "")
        if not m:
            continue
        sid = f"session_{m.group(1)}"
        for note in rec.get("notes") or []:
            out.setdefault(note, set()).add(sid)
    return out


def _sessions_for(session_map: dict[str, set[str]], ref: str) -> set[str]:
    """Sessions a retrieved ref counts for. Exact rel match first; wikilink
    names (silica_read_note takes names, not paths) fall back to basename."""
    hit = session_map.get(ref)
    if hit is not None:
        return hit
    stem = ref.rsplit("/", 1)[-1].removesuffix(".md").casefold()
    out: set[str] = set()
    for rel, sids in session_map.items():
        if rel.rsplit("/", 1)[-1].casefold() == stem:
            out |= sids
    return out


# --- Answer ------------------------------------------------------------------
# Shared sentences between the one-shot and agent system prompts (e2e leg
# comparability rule: the judge must see the same contract; only the memory
# delivery sentence differs). Byte-stability asserted by the harness test.

_CONTRACT_OPEN = (
    "You are a helpful assistant answering questions from your memory of "
    "past conversations between {a} and {b}. Today's "
    "date is {now}. "
)
_ONESHOT_DELIVERY = (
    "Use ONLY the memory provided. A 'Personal memory' "
    "section, when present, lists dated facts distilled from those "
    "conversations — treat them as reliable memory on par with the session "
    "transcripts. "
)
_AGENT_DELIVERY = (
    "Use your memory tools to recall those conversations before answering; "
    "nothing is provided inline. "
)
_CONTRACT_CLOSE = (
    "Answer concisely with only the information asked for. If "
    "the memory does not contain the answer, reply that you do not have "
    "that information — never guess."
)


def answer_question(model: str, question: str, now: str, context: str,
                    speakers: tuple[str, str]) -> str:
    from silica.agent.llm import call_llm

    system = (_CONTRACT_OPEN.format(a=speakers[0], b=speakers[1], now=now)
              + _ONESHOT_DELIVERY + _CONTRACT_CLOSE)
    user = f"Memory:\n{context}\n\nQuestion: {question}"
    # temperature=0: same rationale as the LME harness — single-run A/Bs need
    # greedy decoding (a byte-identical prompt flipped verdicts otherwise).
    resp = call_llm(model, [{"role": "system", "content": system},
                            {"role": "user", "content": user}], max_tokens=512,
                    temperature=0.0)
    return (resp.text or "").strip()


# --- Agent answer (e2e read path) --------------------------------------------
# --answer agent: the real product loop over the frozen vault. Tools only,
# plus the product's session-start vault map; WHAT and WHEN to retrieve is
# entirely the agent's. Read-only lane: a write would contaminate the frozen
# vault and break reuse.

_READONLY_TOOLS = (
    "silica_recall", "silica_search", "silica_semantic_search",
    "silica_search_context", "silica_related", "silica_read_note",
    "silica_outline", "silica_links", "silica_concepts",
    "silica_graph_explain", "silica_props", "silica_exists", "silica_files",
)
_AGENT_MAX_ITERATIONS = 10   # below the product default 20: declared cost control
_ABSTAIN = "I do not have that information."


def answer_question_agent(model: str, question: str, now: str,
                          speakers: tuple[str, str],
                          timeline_seed: str | None = None) -> dict:
    """One question through run_agent. Returns response + instrumentation:
    iterations, tools_used (sequence), notes_read (recall/read deliveries,
    not search hits), budget_exhausted, error.

    ``timeline_seed`` (distill+timeline experiment): an extra system message
    injected after the product vmap. None = bit-identical to the R baseline."""
    from silica.agent import loop as loop_mod
    from silica.agent.constraints import AgentConstraints
    from silica.agent.events import ToolCompleteEvent
    from silica.kernel.vault_map import build_vault_map

    system = (_CONTRACT_OPEN.format(a=speakers[0], b=speakers[1], now=now)
              + _AGENT_DELIVERY + _CONTRACT_CLOSE)
    messages = [{"role": "system", "content": system}]
    vmap = build_vault_map()   # the product's CoALA session-start seed
    if vmap:
        messages.append({"role": "system", "content": vmap})
    if timeline_seed:
        messages.append({"role": "system", "content": timeline_seed})
    messages.append({"role": "user", "content": question})

    events: list[ToolCompleteEvent] = []

    def _collect(evt) -> None:
        if isinstance(evt, ToolCompleteEvent):
            events.append(evt)

    err = None
    try:
        response = loop_mod.run_agent(
            messages, model, tool_progress_callback=_collect,
            constraints=AgentConstraints(tools=_READONLY_TOOLS,
                                         max_iterations=_AGENT_MAX_ITERATIONS))
    except Exception as e:   # includes the loop's 3-strike RuntimeError
        response, err = "", f"{type(e).__name__}: {e}"

    notes_read: set[str] = set()
    for e in events:
        if e.name == "silica_recall":
            try:
                notes_read.update(json.loads(e.result).get("notes") or [])
            except Exception:
                pass
        elif e.name == "silica_read_note":
            name = (e.args or {}).get("name")
            if name:
                notes_read.add(str(name))
    exhausted = response == "(silica: maximum iterations reached)"
    if exhausted:
        # The product's real behavior under budget: no answer = abstention.
        response = _ABSTAIN
    return {
        "response": (response or "").strip(),
        "iterations": (_AGENT_MAX_ITERATIONS if exhausted
                       else len({e.iteration for e in events}) + 1),
        "tools_used": [e.name for e in events],
        "notes_read": sorted(notes_read),
        "budget_exhausted": exhausted,
        "error": err,
    }


def _vault_digest(vault: Path) -> str:
    """Belt beyond the toolset braces: content digest of every vault .md, taken
    before the first and after the last question of a conversation."""
    import hashlib

    h = hashlib.sha256()
    for f in sorted(vault.rglob("*.md")):
        h.update(str(f.relative_to(vault)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _agent_aggregate(rows: list[dict]) -> dict | None:
    from collections import Counter

    agent_rows = [r for r in rows if r.get("iterations") is not None]
    if not agent_rows:
        return None
    dist = Counter(t for r in agent_rows for t in (r.get("tools_used") or []))
    return {
        "iterations_mean": round(sum(r["iterations"] for r in agent_rows)
                                 / len(agent_rows), 2),
        "tool_calls": dict(dist.most_common()),
        "budget_exhausted_n": sum(bool(r.get("budget_exhausted")) for r in agent_rows),
        "error_n": sum(bool(r.get("error")) for r in agent_rows),
    }


def _compute_metrics(rows: list[dict]) -> dict:
    """Shared metrics for both answer modes. Adds top-level error_n (errored
    questions score correct=None, so aggregate() excludes them — error_n keeps
    provider flakiness visible instead of silently shrinking the denominator)."""
    m = aggregate(rows)
    errs = sum(1 for r in rows if r.get("error"))
    if errs:
        m["error_n"] = errs
    ag = _agent_aggregate(rows)
    if ag:
        m["agent"] = ag
    return m


# --- Run ---------------------------------------------------------------------

def run_question(qa: dict, qid: str, index: dict[str, dict], *, model: str,
                 judge_model: str, k: int, stuff: bool, use_embedder: bool,
                 use_rerank: bool, retrieval_only: bool, distill: bool,
                 episodic_ttl: int, flat_context: bool, facts_last: bool,
                 windows: int | None, window_chars: int | None,
                 now: str, speakers: tuple[str, str],
                 answer_mode: str = "oneshot",
                 session_map: dict[str, set[str]] | None = None,
                 run_sessions: dict[str, str] | None = None,
                 n_sessions: int | None = None,
                 timeline_seed: str | None = None,
                 improve: bool = False, assemble: bool = False,
                 lexical: bool = False) -> dict:
    cat = qa.get("category")
    qtype = _CATEGORY.get(cat, f"cat-{cat}")
    is_abs = cat == _ADVERSARIAL
    gold = qa.get("adversarial_answer") if is_abs else qa.get("answer")

    gold_sessions = evidence_sessions(qa.get("evidence") or []) if not is_abs else set()
    agent: dict | None = None
    ephemeral_hit: bool | None = None
    gold_in_ctx: bool | None = None
    err: str | None = None

    if answer_mode == "agent":
        agent = answer_question_agent(model, qa["question"], now, speakers,
                                      timeline_seed=timeline_seed)
        response = agent["response"]
        err = agent["error"]
        rels = agent["notes_read"]
    else:
        from silica.kernel import perception

        win_kw = {}
        if windows is not None:
            win_kw["windows"] = windows
        if window_chars is not None:
            win_kw["window_chars"] = window_chars
        p = perception.perceive(qa["question"], now=now, k=k,
                                use_embedder=use_embedder, use_rerank=use_rerank,
                                episodic_ttl_days=episodic_ttl, with_facts=distill,
                                paths=list(index.keys()) if stuff else None,
                                use_recall_weights=improve,
                                assemble=assemble, use_lexical=lexical, **win_kw)
        rels = [b.path for b in p.blocks]

        if distill and gold_sessions:
            if run_sessions:
                # FSM mode: fact runs are Coordinator run_ids, mapped back to
                # the session each run ingested (fsm_runs.json).
                ephemeral_hit = any(run_sessions.get(r) in gold_sessions
                                    for chain in p.fact_chains
                                    for f in chain for r in f.runs)
            else:
                ephemeral_hit = _ephemeral_hit(p.fact_chains, gold_sessions)

        if retrieval_only:
            response = ""
        else:
            context = p.render(facts_first=not facts_last, windowed=not flat_context)
            if not is_abs:
                gold_in_ctx = _gold_in_context(gold, context)
            # Per-question guard: one flaky provider response must become an
            # error row, not kill the whole run (post-mortem: baseline died at
            # 9/585 on a transient OpenRouter APIError). Same isolation the
            # agent path already gives run_agent.
            try:
                response = answer_question(model, qa["question"], now, context, speakers)
            except Exception as e:
                response, err = "", f"{type(e).__name__}: {e}"

    # Judge (both paths, remote LLM): guarded too, and a failed answer is never
    # judged. An errored question scores None — excluded from accuracy, surfaced
    # via metrics error_n — rather than counted wrong for provider flakiness.
    if retrieval_only or err:
        correct = None
    else:
        try:
            correct = judge(judge_model, qtype, qa["question"], gold, response,
                            is_abs=is_abs)
        except Exception as e:
            correct, err = None, f"{type(e).__name__}: {e}"

    # Bump only when the weight can actually feed back into a later retrieval:
    # oneshot (agent mode doesn't read recall_weights yet) and non-stuff (stuff
    # bypasses retrieval, so a bumped weight would never be re-read). The CLI
    # rejects both combos with --improve; this guard also protects a direct
    # run()/run_question() caller that bypasses main().
    if improve and correct and answer_mode == "oneshot" and not stuff:
        from silica.kernel.recall_weights import bump

        bump(rels)

    retrieved_sessions: set[str] = set()
    if session_map is not None:
        for r in rels:
            retrieved_sessions |= _sessions_for(session_map, r)
    else:
        retrieved_sessions = {index.get(r, {}).get("session_id") for r in rels}
    return {
        "question_id": qid,
        "question_type": qtype,
        "abstention": is_abs,
        "correct": correct,
        "sessions": n_sessions if n_sessions is not None else len(index),
        "retrieved": len(rels),
        "session_recall": (len(gold_sessions & retrieved_sessions) / len(gold_sessions))
                          if gold_sessions and not stuff and not is_abs else None,
        "ephemeral_hit": ephemeral_hit,
        "gold_in_context": gold_in_ctx,
        "response": response[:500],
        "iterations": agent["iterations"] if agent else None,
        "tools_used": agent["tools_used"] if agent else None,
        "notes_read": agent["notes_read"] if agent else None,
        "budget_exhausted": agent["budget_exhausted"] if agent else None,
        "error": err,
    }


def _filtered_qa(inst: dict, categories: set[int] | None) -> list[tuple[int, dict]]:
    return [(i, qa) for i, qa in enumerate(inst.get("qa") or [])
            if not categories or qa.get("category") in categories]


def run(data: list[dict], run_root: Path, *, model: str, judge_model: str, k: int,
        stuff: bool, use_embedder: bool, use_rerank: bool = True,
        retrieval_only: bool = False, distill: bool = False,
        episodic_ttl: int = 0, reuse: bool = False, flat_context: bool = False,
        facts_last: bool = False, windows: int | None = None,
        window_chars: int | None = None, key_schema: bool = False,
        categories: set[int] | None = None, limit: int | None = None,
        ingest_mode: str = "distill", answer_mode: str = "oneshot",
        timeline: bool = False, improve: bool = False, assemble: bool = False,
        lexical: bool = False,
        verbose: bool = False, out: Path | None = None) -> dict:
    from silica.config import CONFIG
    from silica.kernel import perception

    planned = sum(len(_filtered_qa(inst, categories)) for inst in data)
    if limit is not None:
        planned = min(planned, limit)
    rows: list[dict] = []
    doc = {
        "generated_at": datetime.date.today().isoformat(),
        "benchmark": "locomo",
        "config": {"answer_model": None if retrieval_only else model,
                   "judge_model": None if retrieval_only else judge_model,
                   "retrieval": "stuff-all" if stuff else f"facade-top{k}",
                   "retrieval_only": retrieval_only,
                   "distill": distill,
                   "reuse": reuse,
                   "key_schema": key_schema,
                   "categories": sorted(categories) if categories else "all",
                   "ingest_mode": ingest_mode,
                   "answer_mode": answer_mode,
                   "timeline": timeline,
                   "improve": improve,
                   "assemble": assemble,
                   "lexical": lexical,
                   "seen_override": "session-date" if ingest_mode == "fsm" else None,
                   "fsm": {},
                   "failed_conversations": [],
                   "max_iterations": (_AGENT_MAX_ITERATIONS
                                      if answer_mode == "agent" else None),
                   "agent_tools": (list(_READONLY_TOOLS)
                                   if answer_mode == "agent" else None),
                   "agent_temperature": ("provider-default"
                                         if answer_mode == "agent" else None),
                   "tainted": [],
                   "context": "flat" if flat_context else "windowed",
                   "windows": windows if windows is not None else perception.DEFAULT_WINDOWS,
                   "window_chars": (window_chars if window_chars is not None
                                    else perception.WINDOW_CHARS),
                   "facts_position": "last" if facts_last else "first",
                   "episodic_ttl": episodic_ttl,
                   "provider_pin": CONFIG.openrouter_provider or None,
                   "embedder": use_embedder and not stuff,
                   "reranker": (getattr(CONFIG, "rerank_model", None) or None)
                               if use_rerank and not stuff else None},
        "metrics": {},
        "questions": rows,
    }

    _metrics = _compute_metrics

    old_ttl = CONFIG.episodic_ttl_days
    if answer_mode == "agent":
        # silica_recall reads CONFIG (no per-call TTL): mirror --episodic-ttl,
        # the same seam the slice parameterizes (0 = never expire, LoCoMo span).
        CONFIG.episodic_ttl_days = episodic_ttl
    try:
        _run_conversations(data, rows, doc, run_root=run_root, model=model,
                           judge_model=judge_model, k=k, stuff=stuff,
                           use_embedder=use_embedder, use_rerank=use_rerank,
                           retrieval_only=retrieval_only, distill=distill,
                           episodic_ttl=episodic_ttl, reuse=reuse,
                           flat_context=flat_context, facts_last=facts_last,
                           windows=windows, window_chars=window_chars,
                           key_schema=key_schema, categories=categories,
                           limit=limit, ingest_mode=ingest_mode,
                           answer_mode=answer_mode, timeline=timeline,
                           verbose=verbose, out=out,
                           planned=planned, metrics=_metrics, improve=improve,
                           assemble=assemble, lexical=lexical)
    finally:
        CONFIG.episodic_ttl_days = old_ttl
    doc.pop("partial", None)
    doc["metrics"] = _metrics(rows)
    return doc


def _run_conversations(data, rows, doc, *, run_root, model, judge_model, k,
                       stuff, use_embedder, use_rerank, retrieval_only,
                       distill, episodic_ttl, reuse, flat_context, facts_last,
                       windows, window_chars, key_schema, categories, limit,
                       ingest_mode, answer_mode, timeline, verbose, out, planned,
                       metrics, improve, assemble, lexical) -> None:
    for inst in data:
        if limit is not None and len(rows) >= limit:
            break
        qa_list = _filtered_qa(inst, categories)
        if limit is not None:
            qa_list = qa_list[:limit - len(rows)]
        if not qa_list:
            continue
        sample_id = inst.get("sample_id") or f"conv{data.index(inst)}"
        conv = inst["conversation"]
        speakers = (conv.get("speaker_a", "speaker A"), conv.get("speaker_b", "speaker B"))
        vault = question_vault(run_root, sample_id)
        vault.mkdir(parents=True, exist_ok=True)
        if key_schema:
            # ADR-0021 lever, same seam as LME: the manifest makes
            # capture_from_distill enforce the default key schema.
            (vault / "vault.yaml").write_text(
                "conventions:\n  episodic_keys: {}\n", encoding="utf-8")
        bind_vault(vault)
        session_map = run_sessions = None
        if ingest_mode == "fsm":
            _clear_fsm_state()
            marker = fsm_ingest_conversation(vault, inst, reuse=reuse,
                                             key_schema=key_schema)
            if marker is None:
                doc["config"]["failed_conversations"].append(sample_id)
                continue
            doc["config"]["fsm"][sample_id] = {
                "anneal": marker.get("anneal"),
                "partial_sessions": marker.get("partial_sessions"),
                "runs": "fsm_runs.json"}
            session_map = _provenance_session_map(vault)
            run_sessions = json.loads((vault / "fsm_runs.json")
                                      .read_text(encoding="utf-8"))
            sess = conversation_sessions(inst["conversation"])
            index = {}
            n_sessions = len(sess)
            now = max((d for d in (parse_date_time(dt) for _, dt, _ in sess) if d),
                      default="")
            fsm_reused = bool(marker.get("reused"))
        else:
            index = load_conversation_vault(vault, inst, distill=distill, reuse=reuse)
            n_sessions = len(index)
            now = _conv_now(index)
            fsm_reused = reuse
        if not stuff:
            build_indexes(embed=use_embedder, force=not fsm_reused)
        digest_before = _vault_digest(vault) if answer_mode == "agent" else None
        # Timeline overlay: one deterministic seed per conversation from the
        # already-bound vault, injected as an extra agent system message. The
        # only variable that differs from the R baseline (--timeline off).
        timeline_seed = (build_timeline_seed(vault)
                         if timeline and answer_mode == "agent" else None)
        for qi, qa in qa_list:
            row = run_question(qa, f"{sample_id}_q{qi}", index, model=model,
                               judge_model=judge_model, k=k, stuff=stuff,
                               use_embedder=use_embedder, use_rerank=use_rerank,
                               retrieval_only=retrieval_only,
                               distill=distill or ingest_mode == "fsm",
                               episodic_ttl=episodic_ttl, flat_context=flat_context,
                               facts_last=facts_last, windows=windows,
                               window_chars=window_chars, now=now, speakers=speakers,
                               answer_mode=answer_mode, session_map=session_map,
                               run_sessions=run_sessions, n_sessions=n_sessions,
                               timeline_seed=timeline_seed, improve=improve,
                               assemble=assemble, lexical=lexical)
            rows.append(row)
            if verbose:
                mark = (f"sr={row['session_recall']}" if retrieval_only
                        else ("OK " if row["correct"] else "XX "))
                print(f"  [{len(rows)}/{planned}] {row['question_id']:<24} "
                      f"{row['question_type']:<14} {mark} "
                      f"{'(abs)' if row['abstention'] else ''}", flush=True)
            if out:
                # Checkpoint after every question: a killed/hung run keeps
                # everything scored so far, marked partial until the last row.
                doc["partial"] = f"{len(rows)}/{planned}"
                doc["metrics"] = metrics(rows)
                out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        if digest_before is not None and _vault_digest(vault) != digest_before:
            # Read-only invariant: belt beyond the toolset braces.
            logger.error("vault %s mutated during agent answering — RUN TAINTED",
                         sample_id)
            doc["config"]["tainted"].append(sample_id)


def _print_summary(doc: dict) -> None:
    m, cfg = doc["metrics"], doc["config"]
    print(f"\nlocomo — answer={cfg['answer_model']} judge={cfg['judge_model']} "
          f"retrieval={cfg['retrieval']}")
    print(f"  overall accuracy   {m['overall_accuracy']}  (n={m['answerable_n']})")
    print(f"  adversarial        {m['abstention_accuracy']}  (n={m['abstention_n']})")
    if m["session_recall_mean"] is not None:
        print(f"  session recall     {m['session_recall_mean']}")
    if m.get("ephemeral_hit_mean") is not None:
        print(f"  ephemeral hit      {m['ephemeral_hit_mean']}")
    for qt, s in m["by_type"].items():
        sr = f"  sr={s['session_recall']}" if s.get("session_recall") is not None else ""
        print(f"  {qt:<14} acc={s['accuracy']}  (n={s['n']}){sr}")


def main(argv=None) -> int:
    from silica.config import CONFIG

    ap = argparse.ArgumentParser(prog="python -m tests.eval.locomo")
    ap.add_argument("--data", required=True, help="locomo10.json")
    ap.add_argument("--run-root", required=True, help="dir for the per-conversation vaults")
    ap.add_argument("--model", default=CONFIG.model, help="answer model (litellm string)")
    ap.add_argument("--judge-model", default=CONFIG.model, help="judge model (litellm string)")
    ap.add_argument("--stuff", action="store_true",
                    help="feed all sessions, skip retrieval (reasoning ceiling)")
    ap.add_argument("--distill", action="store_true",
                    help="distill each session via the Silica distiller before "
                         "indexing (mem0-comparable LLM ingest; default is verbatim)")
    ap.add_argument("--fsm-max-concepts", type=int, default=0,
                    help="coarsen FSM concept extraction: cap salient keyphrases per "
                         "session (0 = product default 40). YAKE-ranked, so the trivial "
                         "tail drops first -> fewer, denser notes. --ingest fsm only.")
    ap.add_argument("--ingest", choices=("distill", "fsm"), default="distill",
                    help="write path: 'distill' = slice ingest (per --distill), "
                         "'fsm' = full product Coordinator per session "
                         "(collision/dedup/deferred/anneal live)")
    ap.add_argument("--answer", choices=("oneshot", "agent"), default="oneshot",
                    help="read path: 'oneshot' = stuffed-context single call, "
                         "'agent' = product run_agent loop with read-only tools")
    ap.add_argument("--timeline", action="store_true",
                    help="inject a chronological index of the vault's dated "
                         "notes as an extra agent system seed (--answer agent "
                         "only; benchmark overlay, not a product feature)")
    ap.add_argument("--improve", action="store_true",
                    help="LoCoMo eval-only recall-outcome reweighting (phase 1 of "
                         "the Cognee-parity `improve` probe): bump a note's weight "
                         "when it contributed to a correctly judged answer, feed "
                         "weights back into RRF as an extra leg on the next "
                         "question in the same conversation. Oneshot only in "
                         "phase 1 — agent mode's silica_recall tool call doesn't "
                         "consume the weights yet — and incompatible with "
                         "--stuff, which bypasses retrieval.")
    ap.add_argument("--assemble", action="store_true",
                    help="read-time assembly A/B arm: expand each seed 1 hop over "
                         "parent/children/related/edges and squash co-hub seeds "
                         "into one breadcrumbed block (default off).")
    ap.add_argument("--lexical", action="store_true",
                    help="read-time lexical (BM25) A/B arm: fuse the lexical leg "
                         "into facade retrieval (default off). Requires the "
                         "lexical index to have been built first with "
                         "silica_lexical_refresh (the /lexical CLI) — otherwise "
                         "this is a no-op with an empty index.")
    ap.add_argument("--conversations", default="",
                    help="comma-separated sample_ids to run "
                         "(pilot: conv-26,conv-47,conv-49)")
    ap.add_argument("--episodic-ttl", type=int, default=0,
                    help="episodic fact TTL in days (default 0 = off, the "
                         "Mem0/Zep-comparable headline)")
    ap.add_argument("--reuse-vaults", action="store_true",
                    help="adopt existing conversation vaults as-is (frozen corpus: "
                         "skip re-distillation so A/Bs across runs are causal)")
    ap.add_argument("--key-schema", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="drop a default episodic_keys manifest into each fresh "
                         "conversation vault (ADR-0021; default ON — "
                         "--no-key-schema for the legacy free-key arm)")
    ap.add_argument("--categories", default="",
                    help="comma-separated LoCoMo categories to run, e.g. '1,2,4' "
                         "(default: all; 5 = adversarial/abstention)")
    ap.add_argument("--flat-context", action="store_true",
                    help="legacy perception: full note bodies, no windowing")
    ap.add_argument("--facts-last", action="store_true",
                    help="legacy layout: Personal memory block after the sessions")
    ap.add_argument("--windows", type=int,
                    help="query-dense windows per note (default: the perceive() default)")
    ap.add_argument("--window-chars", type=int,
                    help="chars per window (default: the perceive() default)")
    ap.add_argument("--no-embed", action="store_true", help="cooccur retrieval only")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rerank pass")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip answer+judge; report session_recall only (LLM-free)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, help="max questions across all conversations")
    ap.add_argument("--out")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.ingest == "fsm" and (args.stuff or args.distill):
        print("--ingest fsm distills inside the FSM; drop --stuff/--distill")
        return 2
    if args.fsm_max_concepts:
        if args.ingest != "fsm":
            print("--fsm-max-concepts only applies to --ingest fsm")
            return 2
        # Read at call time inside keyphrase._cutoff; set for the whole run so
        # every session's nucleation coarsens identically. Freeze markers key on
        # vault content, not this env, so mixing arms needs distinct --run-root.
        os.environ["SILICA_MAX_CONCEPTS"] = str(args.fsm_max_concepts)
    if args.answer == "agent" and (args.stuff or args.retrieval_only):
        print("--answer agent retrieves via tools; drop --stuff/--retrieval-only")
        return 2
    if args.timeline and args.answer != "agent":
        print("--timeline seeds the agent; pass --answer agent")
        return 2
    if args.improve and args.answer != "oneshot":
        print("--improve is oneshot-only in phase 1 (agent-mode retrieval "
              "doesn't read recall_weights yet)")
        return 2
    if args.improve and args.stuff:
        print("--improve is incompatible with --stuff: --stuff bypasses "
              "retrieval, so a bumped weight would never feed back")
        return 2
    if args.retrieval_only:
        args.stuff = False  # nothing to retrieve when every session is stuffed in
    elif not args.model:
        print("no answer model: set SILICA_MODEL or pass --model")
        return 2
    categories = ({int(c) for c in args.categories.split(",") if c.strip()}
                  if args.categories else None)
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.conversations:
        wanted = {c.strip() for c in args.conversations.split(",") if c.strip()}
        data = [inst for inst in data if inst.get("sample_id") in wanted]
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    out = Path(args.out) if args.out else METRICS_PATH
    try:
        doc = run(data, run_root, model=args.model, judge_model=args.judge_model,
                  k=args.k, stuff=args.stuff, use_embedder=not args.no_embed,
                  use_rerank=not args.no_rerank, retrieval_only=args.retrieval_only,
                  distill=args.distill, episodic_ttl=args.episodic_ttl,
                  reuse=args.reuse_vaults, flat_context=args.flat_context,
                  facts_last=args.facts_last, windows=args.windows,
                  window_chars=args.window_chars, key_schema=args.key_schema,
                  categories=categories, limit=args.limit,
                  ingest_mode=args.ingest, answer_mode=args.answer,
                  timeline=args.timeline, improve=args.improve,
                  assemble=args.assemble, lexical=args.lexical,
                  verbose=args.verbose, out=out)
    finally:
        import silica.driver
        silica.driver._driver = None

    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(doc)
    print(f"\nreport → {out}")
    return 0
