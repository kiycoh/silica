# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""LongMemEval adapter — chat-memory QA scored by an LLM judge.

Silica's home turf: long-term conversational memory. Unlike MuSiQue (one
shared corpus, recall@k), LongMemEval gives every question its OWN timestamped
haystack of chat sessions and scores QA correctness with an LLM judge across
six ability types (single-session-user/assistant/preference, temporal-reasoning,
knowledge-update, multi-session) plus abstention (``question_id`` ending _abs).

Per-question pipeline (retrieval-then-answer — the config comparable 1:1 with
Mem0 / Zep, not the agentic loop):

  1. load    — one .md note per session into a fresh isolated vault; the session
               date + id go in frontmatter (temporal-reasoning needs the date).
               Default is verbatim, so this bypasses the OFM lint (the code-wiki
               rationale: external ground truth, the structural validators do not
               apply — and chat text with literal '[[' must not be rejected).
               ``--distill`` instead routes each session through the Silica
               distiller (the mem0-comparable LLM ingest); it runs per-session so
               one note stays one session and session_recall holds, and falls
               back to verbatim if the distiller errors on a session.
  2. index   — cooccur (offline) and, when an embedder is served, embeddings.
  3. perceive— retrieval AND context assembly are silica.kernel.perception
               .perceive(), the product path: fused facade + rerank, per-note
               query-densest window under rank/evidence/date headers, episodic
               facts first. The harness owns no context assembler, so eval and
               product cannot diverge on this seam. (``--stuff`` skips
               retrieval and feeds every session, isolating the reasoning
               ceiling — the right mode for the oracle split.)
  4. answer  — the configured chat model answers from ONLY the retrieved memory;
               told to decline when the memory lacks the answer (abstention).
  5. judge   — the judge model applies the LongMemEval rubric (per-type prompt);
               'yes' in the reply => correct. Prompts mirror
               github.com/xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py.

  uv run python -m tests.eval.longmemeval \
      --data longmemeval_oracle.json --run-root bench/lme --stuff --limit 20

Requires an LLM for answer + judge (litellm, via CONFIG.model). The embedder is
optional (cooccur leg / --stuff work offline). The personal-memory lane
(ADR-0019) is never passed — only the question's own sessions can be recalled.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

METRICS_PATH = Path(__file__).parent / "metrics.json"
_ABS = "_abs"
# Distilled-body meta-description markers (matched on a snippet's opening chars).
_META_BODY_RE = re.compile(
    r"(?i)\b(user and assistant|user-generated prompt|assistant response"
    r"|placeholder entry|dialog(?:ue)? between|the user inquir"
    r"|this (?:conversation|session|transcript)"
    r"|(?:conversation|session|transcript) (?:about|between|covers|discusses))\b")

# --- Judge rubric (LongMemEval evaluate_qa.py, reconstructed faithfully) -----
_JUDGE_BASE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, answer yes. If the "
    "response only contains a subset of the information required by the answer, "
    "answer no."
)
_JUDGE_TEMPORAL = (
    " In addition, do not penalize off-by-one errors for the number of days. If "
    "the question asks for a number of days/weeks/months and the model makes an "
    "off-by-one error, the response is still correct."
)
_JUDGE_KNOWLEDGE = (
    " If the response contains some previous information along with an updated "
    "answer, it is correct as long as the updated answer is the required answer."
)
_JUDGE_PREFERENCE = (
    "I will give you a question, a rubric for the desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all points in the rubric. The response is correct as long as it recalls and "
    "utilizes the user's personal information correctly."
)
_JUDGE_ABSTENTION = (
    "I will give you an unanswerable question, an incorrect answer, and a response "
    "from a model. Please answer yes if the model correctly identifies the question "
    "as unanswerable. The model could say that the information is incomplete, or "
    "some other information is given but the asked-for information is not. Answer no "
    "if the model attempts to answer the question with the incorrect answer or any "
    "other made-up information."
)


def _judge_instruction(qtype: str, is_abs: bool = False) -> str:
    if is_abs:
        return _JUDGE_ABSTENTION
    if qtype == "single-session-preference":
        return _JUDGE_PREFERENCE
    if qtype == "temporal-reasoning":
        return _JUDGE_BASE + _JUDGE_TEMPORAL
    if qtype == "knowledge-update":
        return _JUDGE_BASE + _JUDGE_KNOWLEDGE
    return _JUDGE_BASE


# --- Session -> note ---------------------------------------------------------

def _session_rel(i: int) -> str:
    return f"sessions/s{i:04d}.md"


def render_session(turns: list[dict]) -> str:
    """Turns [{role, content}] -> a readable transcript body."""
    lines = []
    for t in turns:
        role = "User" if t.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {(t.get('content') or '').strip()}")
    return "\n\n".join(lines)


def _frontmatter_note(session_id: str, date: str, body: str) -> str:
    return (
        "---\n"
        f"session_id: {json.dumps(session_id, ensure_ascii=False)}\n"
        f"date: {json.dumps(date, ensure_ascii=False)}\n"
        "source: longmemeval\n"
        "tags:\n  - benchmark\n"
        "AI: true\n"
        "---\n\n"
        f"{body}\n"
    )


def _note(session_id: str, date: str, turns: list[dict]) -> str:
    return _frontmatter_note(session_id, date, render_session(turns))


def _episodic_keys_substrate() -> str | None:
    """Live episodic keys as the only substrate section for the eval distill:
    capture snaps to the established key vocabulary instead of coining
    synonyms. Failure never blocks the distill (the section is advisory)."""
    try:
        from silica.kernel.episodic import EpisodicStore, key_vocabulary_section

        return key_vocabulary_section(EpisodicStore())
    except Exception as e:
        logger.warning("episodic keys substrate failed (distill proceeds): %s", e)
        return None


def distill_session(session_id: str, date: str, turns: list[dict]) -> str:
    """Distill one session's transcript into a knowledge-note body via the Silica
    distiller — the mem0-comparable ingest (LLM-driven memory formation).

    Question-blind: the payload carries only this session's text; the eval
    question and gold answer are never passed. Runs per-session so one note stays
    one session (session_recall's rel->session_id map holds). Falls back to the
    verbatim transcript when the distiller errors or yields nothing, so one bad
    call never drops a session from the haystack."""
    from silica.kernel import prep_delegation

    excerpt = render_session(turns)
    payload = {
        "schema_version": 1,
        "batches": [{"inbox_file": session_id,
                     "concepts": [{"name": session_id, "inbox_excerpt": excerpt}]}],
    }
    try:
        result = prep_delegation.run_distiller(
            payload, target="sessions", substrate=_episodic_keys_substrate())
    except Exception as e:  # ponytail: distiller hiccup -> keep the session verbatim
        logger.warning("distiller failed for session %s: %s — keeping verbatim", session_id, e)
        return excerpt
    # Episodic lane: session id is the run_id, the SESSION date is `seen`
    # (simulated time — what makes benchmark temporal reasoning possible).
    from silica.kernel.episodic import capture_from_distill

    capture_from_distill(result, run_id=session_id, seen=date)
    bodies = [(u.get("snippet") or "").strip() for u in (result.get("updates") or [])]
    # Meta-description guard (post-mortem 2026-07-14): a snippet that DESCRIBES
    # the conversation instead of carrying its facts is distill-loss — drop it;
    # with nothing left the verbatim fallback keeps every detail.
    # ponytail: opening-line denylist; false positives just fall back to verbatim.
    bodies = [b for b in bodies if b and not _META_BODY_RE.search(b[:160])]
    body = "\n\n".join(bodies)
    if not body:
        logger.warning("distiller yielded no body for session %s — keeping verbatim", session_id)
    return body or excerpt


def _write_note(rel: str, content: str) -> None:
    """Lint-free create: benchmark ground truth, the OFM structural lint does
    not apply (verbatim chat may carry '[[', code fences, unbalanced markup)."""
    from silica.driver import DRIVER

    DRIVER.create(rel, content)


# --- Per-question vault lifecycle -------------------------------------------

def bind_vault(vault: Path) -> None:
    """Point CONFIG/DRIVER at ``vault`` and drop every store singleton, so no
    prior question's sessions or indexes can leak into this one."""
    import silica.driver
    import silica.kernel.cooccurrence as cooc_mod
    import silica.kernel.embed as embed_mod
    from silica.config import CONFIG

    CONFIG.vault_path = str(vault)
    # Episodic isolation: episodic_home() resolves to the question's own vault,
    # while memory_lane keeps abstaining bit-identically (coincident-vault rule)
    # so the "personal-memory lane is never passed" guarantee holds.
    CONFIG.memory_vault = str(vault)
    CONFIG.backend = "fs"
    silica.driver._driver = None
    embed_mod.clear()
    cooc_mod.clear()


def load_question_vault(vault: Path, inst: dict, distill: bool = False,
                        reuse: bool = False) -> dict[str, dict]:
    """Write one note per haystack session; return {rel: {session_id, date}}.

    ``distill`` routes each session through the Silica distiller (mem0-comparable
    ingest) instead of storing the verbatim transcript. ``reuse`` adopts an
    already-populated vault as-is (frozen corpus: the distiller is LLM-driven and
    re-rolls every note per run, which confounds any A/B across runs) — no
    distiller call, notes untouched, index rebuilt from note frontmatter."""
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
    sids = inst.get("haystack_session_ids") or []
    dates = inst.get("haystack_dates") or []
    sessions = inst["haystack_sessions"]
    index = {}
    for i, turns in enumerate(sessions):
        rel = _session_rel(i)
        sid = sids[i] if i < len(sids) else f"s{i}"
        date = dates[i] if i < len(dates) else ""
        body = distill_session(sid, date, turns) if distill else render_session(turns)
        _write_note(rel, _frontmatter_note(sid, date, body))
        index[rel.removesuffix(".md")] = {"session_id": sid, "date": date}
    return index


def build_indexes(embed: bool, force: bool = True) -> None:
    from silica.tools.graph import silica_cooccurrence_refresh, silica_embed_refresh

    silica_cooccurrence_refresh(force=force)
    if embed:
        silica_embed_refresh(force=force)


def _gold_in_context(gold: str, context: str) -> bool | None:
    """Whether the gold answer is extractably present in the assembled context —
    splits memory-fail (False: the evidence never reached the model) from
    answer-fail (True: it did, the model missed it). Direct substring hit, or
    every content token (len>3) present. None when the gold has no content
    tokens (numeric/derived golds — '3', '7 days' — are computed, not quoted)."""
    g, c = str(gold).lower().strip(), context.lower()  # dataset golds can be JSON numbers
    if len(g) >= 4 and g in c:
        return True
    toks = [t for t in re.findall(r"[a-z0-9']+", g) if len(t) > 3]
    if not toks:
        return None
    return all(t in c for t in toks)


# --- Answer + judge ----------------------------------------------------------

def answer_question(model: str, question: str, question_date: str, context: str) -> str:
    from silica.agent.llm import call_llm

    system = (
        "You are a helpful assistant answering from your memory of past "
        f"conversations with the user. Today's date is {question_date}. Use ONLY "
        "the memory provided. A 'Personal memory' section, when present, lists "
        "dated facts distilled from those conversations — treat them as reliable "
        "memory on par with the session transcripts. If the memory does not "
        "contain the answer, reply that you do not have that information — "
        "never guess."
    )
    user = f"Memory:\n{context}\n\nQuestion: {question}"
    resp = call_llm(model, [{"role": "system", "content": system},
                            {"role": "user", "content": user}], max_tokens=512)
    return (resp.text or "").strip()


def judge(model: str, qtype: str, question: str, gold: str, response: str,
          is_abs: bool = False) -> bool:
    from silica.agent.llm import call_llm

    if is_abs:
        label, closing = "Incorrect Answer", "Did the model correctly abstain?"
    elif qtype == "single-session-preference":
        label, closing = "Rubric", "Is the model response correct?"
    else:
        label, closing = "Correct Answer", "Is the model response correct?"
    prompt = (
        f"{_judge_instruction(qtype, is_abs)}\n\n"
        f"Question: {question}\n"
        f"{label}: {gold}\n"
        f"Model Response: {response}\n\n"
        f"{closing} Answer yes or no only."
    )
    resp = call_llm(model, [{"role": "user", "content": prompt}], max_tokens=8)
    return "yes" in (resp.text or "").lower()


# --- Run ---------------------------------------------------------------------

def _ephemeral_hit(fact_chains, gold_sessions: set[str]) -> bool:
    """True when at least one recalled fact's chain was seen in a gold session."""
    return any(set(f.runs) & gold_sessions for chain in fact_chains for f in chain)


def question_vault(run_root: Path, qid: str) -> Path:
    """Per-question vault dir; qid sanitized exactly as run_instance writes it.

    Single source of truth for the run-root layout — the key-drift probes
    resolve frozen stores through this same helper."""
    return run_root / re.sub(r"[^A-Za-z0-9_.-]", "_", qid)


def run_instance(inst: dict, run_root: Path, *, model: str, judge_model: str,
                 k: int, stuff: bool, use_embedder: bool, use_rerank: bool = True,
                 retrieval_only: bool = False, distill: bool = False,
                 episodic_ttl: int = 0, reuse: bool = False,
                 flat_context: bool = False, facts_last: bool = False,
                 windows: int | None = None,
                 window_chars: int | None = None) -> dict:
    qid = inst["question_id"]
    qtype = inst["question_type"]
    is_abs = qid.endswith(_ABS)
    vault = question_vault(run_root, qid)
    vault.mkdir(parents=True, exist_ok=True)
    bind_vault(vault)

    index = load_question_vault(vault, inst, distill=distill, reuse=reuse)
    if not stuff:
        # Reused corpus -> incremental refresh (no-op on unchanged notes).
        build_indexes(embed=use_embedder, force=not reuse)

    # Retrieval + context assembly are the PRODUCT path (perception promotion,
    # spec 2026-07-15): the harness owns no assembler of its own. Facts recall
    # is gated on --distill so verbatim arms stay episodic-free even on reused
    # vaults that carry a grafted store.
    from silica.kernel.perception import perceive

    # None = mirror the perceive() defaults, so existing invocations track the
    # product surface even when the grid moves it (multi-window spec 2026-07-15).
    win_kw = {}
    if windows is not None:
        win_kw["windows"] = windows
    if window_chars is not None:
        win_kw["window_chars"] = window_chars
    p = perceive(inst["question"], now=inst.get("question_date", ""),
                 k=k, use_embedder=use_embedder, use_rerank=use_rerank,
                 episodic_ttl_days=episodic_ttl, with_facts=distill,
                 paths=list(index.keys()) if stuff else None, **win_kw)
    rels = [b.path for b in p.blocks]

    gold_sessions = set(inst.get("answer_session_ids") or [])
    ephemeral_hit: bool | None = None
    if distill and gold_sessions and not is_abs:
        ephemeral_hit = _ephemeral_hit(p.fact_chains, gold_sessions)

    gold_in_ctx: bool | None = None
    if retrieval_only:
        # LLM-free loop: measure session_recall only, no answer/judge cost.
        response, correct = "", None
    else:
        # Facts first (post-mortem 2026-07-14): the block is the densest evidence
        # and the answer model misses it when it trails long session notes.
        # ``facts_last`` / ``flat_context`` are the legacy layout A/B arms.
        context = p.render(facts_first=not facts_last, windowed=not flat_context)
        if not is_abs:
            gold_in_ctx = _gold_in_context(inst["answer"], context)
        response = answer_question(model, inst["question"], inst.get("question_date", ""), context)
        correct = judge(judge_model, qtype, inst["question"], inst["answer"], response,
                        is_abs=is_abs)
    retrieved_sessions = {index.get(r, {}).get("session_id") for r in rels}
    return {
        "question_id": qid,
        "question_type": qtype,
        "abstention": is_abs,
        "correct": correct,
        "sessions": len(index),
        "retrieved": len(rels),
        # Abstention gold ids are synthetic markers absent from the haystack;
        # recall is undefined for them, not a 0.0 miss.
        "session_recall": (len(gold_sessions & retrieved_sessions) / len(gold_sessions))
                          if gold_sessions and not stuff and not is_abs else None,
        "ephemeral_hit": ephemeral_hit,
        "gold_in_context": gold_in_ctx,
        "response": response[:500],
    }


def aggregate(rows: list[dict]) -> dict:
    def acc(subset: list[dict]) -> float | None:
        graded = [r for r in subset if r["correct"] is not None]  # None in --retrieval-only
        return round(sum(r["correct"] for r in graded) / len(graded), 4) if graded else None

    def sr(subset: list[dict]) -> float | None:
        vals = [r["session_recall"] for r in subset if r["session_recall"] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    answerable = [r for r in rows if not r["abstention"]]
    by_type: dict[str, dict] = {}
    for qt in sorted({r["question_type"] for r in answerable}):
        sub = [r for r in answerable if r["question_type"] == qt]
        by_type[qt] = {"n": len(sub), "accuracy": acc(sub), "session_recall": sr(sub)}
    recalls = [r["session_recall"] for r in rows if r["session_recall"] is not None]
    eph = [r["ephemeral_hit"] for r in rows if r.get("ephemeral_hit") is not None]
    gic = [r["gold_in_context"] for r in rows if r.get("gold_in_context") is not None]
    return {
        "overall_accuracy": acc(answerable),
        "answerable_n": len(answerable),
        "abstention_accuracy": acc([r for r in rows if r["abstention"]]),
        "abstention_n": sum(r["abstention"] for r in rows),
        "by_type": by_type,
        "session_recall_mean": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "ephemeral_hit_mean": round(sum(eph) / len(eph), 4) if eph else None,
        "gold_in_context_mean": round(sum(gic) / len(gic), 4) if gic else None,
    }


def run(data: list[dict], run_root: Path, *, model: str, judge_model: str, k: int,
        stuff: bool, use_embedder: bool, use_rerank: bool = True,
        retrieval_only: bool = False, distill: bool = False,
        episodic_ttl: int = 0, reuse: bool = False, flat_context: bool = False,
        facts_last: bool = False, windows: int | None = None,
        window_chars: int | None = None, limit: int | None,
        verbose: bool, out: Path | None = None) -> dict:
    from silica.config import CONFIG
    from silica.kernel import perception

    data = data[:limit] if limit else data
    rows: list[dict] = []
    doc = {
        "generated_at": datetime.date.today().isoformat(),
        "benchmark": "longmemeval",
        "config": {"answer_model": None if retrieval_only else model,
                   "judge_model": None if retrieval_only else judge_model,
                   "retrieval": "stuff-all" if stuff else f"facade-top{k}",
                   "retrieval_only": retrieval_only,
                   "distill": distill,
                   "reuse": reuse,
                   "context": "flat" if flat_context else "windowed",
                   # Effective values, so arm A/B reports stay distinguishable.
                   "windows": windows if windows is not None else perception.DEFAULT_WINDOWS,
                   "window_chars": (window_chars if window_chars is not None
                                    else perception.WINDOW_CHARS),
                   "facts_position": "last" if facts_last else "first",
                   # TTL defaults OFF here: Mem0 and Zep do not evaporate
                   # memories, so the headline comparable run must not either.
                   "episodic_ttl": episodic_ttl,
                   "embedder": use_embedder and not stuff,
                   "reranker": (getattr(CONFIG, "rerank_model", None) or None)
                               if use_rerank and not stuff else None},
        "metrics": {},
        "questions": rows,
    }
    for i, inst in enumerate(data):
        row = run_instance(inst, run_root, model=model, judge_model=judge_model,
                           k=k, stuff=stuff, use_embedder=use_embedder,
                           use_rerank=use_rerank, retrieval_only=retrieval_only,
                           distill=distill, episodic_ttl=episodic_ttl, reuse=reuse,
                           flat_context=flat_context, facts_last=facts_last,
                           windows=windows, window_chars=window_chars)
        rows.append(row)
        if verbose:
            mark = (f"sr={row['session_recall']}" if retrieval_only
                    else ("OK " if row["correct"] else "XX "))
            print(f"  [{i+1}/{len(data)}] {row['question_id']:<28} {row['question_type']:<26} "
                  f"{mark} {'(abs)' if row['abstention'] else ''}", flush=True)
        if out:
            # Checkpoint after every question: a killed/hung run keeps
            # everything scored so far, marked partial until the last row.
            doc["partial"] = f"{i + 1}/{len(data)}"
            doc["metrics"] = aggregate(rows)
            out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    doc.pop("partial", None)
    doc["metrics"] = aggregate(rows)
    return doc


def _print_summary(doc: dict) -> None:
    m, cfg = doc["metrics"], doc["config"]
    print(f"\nlongmemeval — answer={cfg['answer_model']} judge={cfg['judge_model']} "
          f"retrieval={cfg['retrieval']}")
    print(f"  overall accuracy   {m['overall_accuracy']}  (n={m['answerable_n']})")
    print(f"  abstention         {m['abstention_accuracy']}  (n={m['abstention_n']})")
    if m["session_recall_mean"] is not None:
        print(f"  session recall     {m['session_recall_mean']}")
    if m.get("ephemeral_hit_mean") is not None:
        print(f"  ephemeral hit      {m['ephemeral_hit_mean']}")
    for qt, s in m["by_type"].items():
        sr = f"  sr={s['session_recall']}" if s.get("session_recall") is not None else ""
        print(f"  {qt:<28} acc={s['accuracy']}  (n={s['n']}){sr}")


def main(argv=None) -> int:
    from silica.config import CONFIG

    ap = argparse.ArgumentParser(prog="python -m tests.eval.longmemeval")
    ap.add_argument("--data", required=True, help="longmemeval_{oracle,s,m}.json")
    ap.add_argument("--run-root", required=True, help="dir for the per-question vaults")
    ap.add_argument("--model", default=CONFIG.model, help="answer model (litellm string)")
    ap.add_argument("--judge-model", default=CONFIG.model, help="judge model (litellm string)")
    ap.add_argument("--stuff", action="store_true",
                    help="feed all sessions, skip retrieval (reasoning ceiling; use for oracle)")
    ap.add_argument("--distill", action="store_true",
                    help="distill each session via the Silica distiller before "
                         "indexing (mem0-comparable LLM ingest; default is verbatim)")
    ap.add_argument("--episodic-ttl", type=int, default=0,
                    help="episodic fact TTL in days for the ablation run "
                         "(default 0 = off, the Mem0/Zep-comparable headline)")
    ap.add_argument("--reuse-vaults", action="store_true",
                    help="adopt existing question vaults as-is (frozen corpus: "
                         "skip re-distillation so A/Bs across runs are causal)")
    ap.add_argument("--flat-context", action="store_true",
                    help="legacy perception: full note bodies, no rank/evidence "
                         "headers, no query-aware windowing")
    ap.add_argument("--facts-last", action="store_true",
                    help="legacy layout: Personal memory block after the sessions")
    ap.add_argument("--windows", type=int,
                    help="query-dense windows per note (default: the perceive() default)")
    ap.add_argument("--window-chars", type=int,
                    help="chars per window (default: the perceive() default)")
    ap.add_argument("--no-embed", action="store_true", help="cooccur retrieval only")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rerank pass")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip answer+judge; report session_recall only (LLM-free retrieval loop)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.retrieval_only:
        args.stuff = False  # nothing to retrieve when every session is stuffed in
    elif not args.model:
        print("no answer model: set SILICA_MODEL or pass --model")
        return 2
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
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
                  window_chars=args.window_chars, limit=args.limit,
                  verbose=args.verbose, out=out)
    finally:
        import silica.driver
        silica.driver._driver = None

    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(doc)
    print(f"\nreport → {out}")
    return 0
