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


# --- Run ---------------------------------------------------------------------

def run_question(qa: dict, qid: str, index: dict[str, dict], *, model: str,
                 judge_model: str, k: int, stuff: bool, use_embedder: bool,
                 use_rerank: bool, retrieval_only: bool, distill: bool,
                 episodic_ttl: int, flat_context: bool, facts_last: bool,
                 windows: int | None, window_chars: int | None,
                 now: str, speakers: tuple[str, str]) -> dict:
    cat = qa.get("category")
    qtype = _CATEGORY.get(cat, f"cat-{cat}")
    is_abs = cat == _ADVERSARIAL
    gold = qa.get("adversarial_answer") if is_abs else qa.get("answer")

    from silica.kernel import perception

    win_kw = {}
    if windows is not None:
        win_kw["windows"] = windows
    if window_chars is not None:
        win_kw["window_chars"] = window_chars
    p = perception.perceive(qa["question"], now=now, k=k,
                            use_embedder=use_embedder, use_rerank=use_rerank,
                            episodic_ttl_days=episodic_ttl, with_facts=distill,
                            paths=list(index.keys()) if stuff else None, **win_kw)
    rels = [b.path for b in p.blocks]

    gold_sessions = evidence_sessions(qa.get("evidence") or []) if not is_abs else set()
    ephemeral_hit: bool | None = None
    if distill and gold_sessions:
        ephemeral_hit = _ephemeral_hit(p.fact_chains, gold_sessions)

    gold_in_ctx: bool | None = None
    if retrieval_only:
        response, correct = "", None
    else:
        context = p.render(facts_first=not facts_last, windowed=not flat_context)
        if not is_abs:
            gold_in_ctx = _gold_in_context(gold, context)
        response = answer_question(model, qa["question"], now, context, speakers)
        correct = judge(judge_model, qtype, qa["question"], gold, response,
                        is_abs=is_abs)
    retrieved_sessions = {index.get(r, {}).get("session_id") for r in rels}
    return {
        "question_id": qid,
        "question_type": qtype,
        "abstention": is_abs,
        "correct": correct,
        "sessions": len(index),
        "retrieved": len(rels),
        "session_recall": (len(gold_sessions & retrieved_sessions) / len(gold_sessions))
                          if gold_sessions and not stuff and not is_abs else None,
        "ephemeral_hit": ephemeral_hit,
        "gold_in_context": gold_in_ctx,
        "response": response[:500],
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
        index = load_conversation_vault(vault, inst, distill=distill, reuse=reuse)
        if not stuff:
            build_indexes(embed=use_embedder, force=not reuse)
        now = _conv_now(index)
        for qi, qa in qa_list:
            row = run_question(qa, f"{sample_id}_q{qi}", index, model=model,
                               judge_model=judge_model, k=k, stuff=stuff,
                               use_embedder=use_embedder, use_rerank=use_rerank,
                               retrieval_only=retrieval_only, distill=distill,
                               episodic_ttl=episodic_ttl, flat_context=flat_context,
                               facts_last=facts_last, windows=windows,
                               window_chars=window_chars, now=now, speakers=speakers)
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
                doc["metrics"] = aggregate(rows)
                out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    doc.pop("partial", None)
    doc["metrics"] = aggregate(rows)
    return doc


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

    if args.retrieval_only:
        args.stuff = False  # nothing to retrieve when every session is stuffed in
    elif not args.model:
        print("no answer model: set SILICA_MODEL or pass --model")
        return 2
    categories = ({int(c) for c in args.categories.split(",") if c.strip()}
                  if args.categories else None)
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
                  window_chars=args.window_chars, key_schema=args.key_schema,
                  categories=categories, limit=args.limit, verbose=args.verbose,
                  out=out)
    finally:
        import silica.driver
        silica.driver._driver = None

    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(doc)
    print(f"\nreport → {out}")
    return 0
