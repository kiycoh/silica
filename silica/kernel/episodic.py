# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Episodic memory lane — short-term fact store with supersedes chains and TTL.

Captures the personal, ephemeral facts the distiller used to discard ("my dog
is named Tom"), keeps them with fact-level supersedes chains and a wall-clock
TTL, recalls them at answer time next to vault notes, and surfaces nucleation
candidates (facts reinforced across runs) in the run digest.

ADR-0019 boundary, stated explicitly: "writes never route to the memory vault"
governs vault NOTES going through the FSM write channel. `episodic.json` is
index-layer state, sibling of the embed/cooccur indices, not vault content.

Kernel rule: this module never calls ``datetime.now()`` — every date (`seen`,
`now`) is supplied by the caller. The product path passes the run date; the
LongMemEval adapter passes the simulated session date.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class Fact(BaseModel):
    id: str
    key: str
    text: str
    first_seen: str
    last_seen: str
    runs: list[str] = Field(default_factory=list)
    supersedes: str | None = None
    status: str = "live"
    # ponytail: inline float list, not npz packing — the store is TTL-bounded
    # to hundreds of facts; the 10k scaling fix targeted note stores.
    vec: list[float] | None = None


class NucleationCandidate(BaseModel):
    key: str
    run_count: int
    since: str
    text: str


class FactHit(BaseModel):
    fact: Fact
    score: float


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(
        ch if ch.isalnum() else " " for ch in text.casefold()
    ).split() if len(t) > 1}


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


_FACTS_ADAPTER = TypeAdapter(list[Fact])


def _days_between(earlier: str, later: str) -> int:
    from datetime import date

    try:
        return (date.fromisoformat(later[:10]) - date.fromisoformat(earlier[:10])).days
    except ValueError:
        return 0  # unparseable date: never expire on bad input


def _normalize(text: str) -> str:
    """Casefold + strip punctuation/whitespace, for supersede-vs-reinforce."""
    out = []
    for ch in text.casefold():
        cat = unicodedata.category(ch)
        if cat.startswith("P") or ch.isspace():
            continue
        out.append(ch)
    return "".join(out)


class EpisodicStore:
    """JSON-file-backed fact store. Facts are not notes; they nucleate INTO notes."""

    def __init__(self, path: Path | None = None):
        self.path = path if path is not None else store_path()
        self.next_id = 1
        self.facts: list[Fact] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            self.next_id = int(doc.get("next_id", 1))
            self.facts = _FACTS_ADAPTER.validate_python(doc.get("facts", []))
        except Exception:
            from silica.kernel.paths import quarantine

            quarantine(self.path)
            self.next_id, self.facts = 1, []

    def save(self) -> None:
        from silica.kernel.paths import atomic_write_bytes

        doc = {
            "schema_version": SCHEMA_VERSION,
            "next_id": self.next_id,
            "facts": [f.model_dump(exclude_none=False) for f in self.facts],
        }
        atomic_write_bytes(self.path, json.dumps(doc, ensure_ascii=False).encode("utf-8"))

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self, facts: list[dict], *, run_id: str, seen: str,
                embedder=None) -> None:
        """Merge distiller ephemerals into the store. Mechanical, no LLM.

        Same key + same normalized text reinforces (last_seen, runs); same key
        + different text supersedes the live head; a new key starts a chain.
        New/changed facts are embedded when `embedder` is served; embedding
        failure is silent (recall falls back to lexical).
        """
        heads = {f.key: f for f in self.facts if f.status == "live"}
        created: list[Fact] = []
        for raw in facts:
            key = (raw.get("key") or "").strip()
            text = (raw.get("text") or "").strip()
            if not key or not text:
                continue
            head = heads.get(key)
            if head is not None and _normalize(head.text) == _normalize(text):
                head.last_seen = seen
                if run_id not in head.runs:
                    head.runs.append(run_id)
                continue
            fid = f"f_{self.next_id:04d}"
            self.next_id += 1
            fact = Fact(id=fid, key=key, text=text, first_seen=seen, last_seen=seen,
                        runs=[run_id])
            if head is not None:
                fact.supersedes = head.id
                head.status = "superseded"
            self.facts.append(fact)
            heads[key] = fact
            created.append(fact)
        if embedder is not None and created:
            try:
                vecs = embedder.embed([f.text for f in created])
                for fact, vec in zip(created, vecs):
                    fact.vec = list(vec)
            except Exception as e:
                logger.debug("episodic capture: embedding skipped (%s)", e)
        self.save()

    # ------------------------------------------------------------------
    # TTL sweep
    # ------------------------------------------------------------------

    def sweep(self, now: str, *, ttl_days: int | None = None) -> int:
        """Delete chains whose HEAD's last_seen is older than ttl_days at `now`.

        Superseded ancestors live exactly as long as their head; expired chains
        are deleted, not archived. Returns the number of chains removed.
        ttl_days=0 means never expire. Persists when anything was removed.
        """
        if ttl_days is None:
            from silica.config import CONFIG

            ttl_days = int(getattr(CONFIG, "episodic_ttl_days", 90))
        if ttl_days <= 0:
            return 0
        expired_ids: set[str] = set()
        removed = 0
        for head in self.live_facts():
            if _days_between(head.last_seen, now) <= ttl_days:
                continue
            removed += 1
            expired_ids.update(self._chain_ids(head))
        if expired_ids:
            self.facts = [f for f in self.facts if f.id not in expired_ids]
            self.save()
        return removed

    def chain(self, head: Fact) -> list[Fact]:
        """The supersede chain from `head` back to its oldest ancestor."""
        by_id = {f.id: f for f in self.facts}
        out, cur = [], head
        while cur is not None:
            out.append(cur)
            cur = by_id.get(cur.supersedes) if cur.supersedes else None
        return out

    def _chain_ids(self, head: Fact) -> list[str]:
        return [f.id for f in self.chain(head)]

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(self, query_text: str, query_vec: list[float] | None = None, *,
               k: int = 10, now: str, ttl_days: int | None = None) -> list["FactHit"]:
        """Top-k LIVE facts for a query. Never mutates the store.

        A fact is scored by the embed leg (cosine) when both vectors exist,
        else lexically (token overlap with text + key segments). The two never
        fuse. `now` filters chains whose head is past TTL without deleting —
        sweep at digest time is the only deleter.
        """
        if ttl_days is None:
            from silica.config import CONFIG

            ttl_days = int(getattr(CONFIG, "episodic_ttl_days", 90))
        q_tokens = _tokens(query_text)
        hits: list[FactHit] = []
        for fact in self.live_facts():
            if ttl_days > 0 and _days_between(fact.last_seen, now) > ttl_days:
                continue
            if query_vec is not None and fact.vec:
                score = _cosine(query_vec, fact.vec)
            else:
                f_tokens = _tokens(fact.text) | _tokens(fact.key.replace(".", " "))
                score = len(q_tokens & f_tokens) / len(q_tokens) if q_tokens else 0.0
            if score > 0.0:
                hits.append(FactHit(fact=fact, score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # ------------------------------------------------------------------
    # Nucleation
    # ------------------------------------------------------------------

    def nucleation_candidates(self, *, min_runs: int | None = None) -> list["NucleationCandidate"]:
        """Keys whose chain accumulated >= min_runs distinct run ids.

        Suggested in the digest, never auto-written: promotion goes through
        the normal write channel when the user or agent acts on it.
        """
        if min_runs is None:
            from silica.config import CONFIG

            min_runs = int(getattr(CONFIG, "episodic_nucleation_runs", 3))
        out: list[NucleationCandidate] = []
        for head in self.live_facts():
            links = self.chain(head)
            runs = {r for f in links for r in f.runs}
            if len(runs) >= min_runs:
                out.append(NucleationCandidate(key=head.key, run_count=len(runs),
                                               since=min(f.first_seen for f in links),
                                               text=head.text))
        return out

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def live_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.status == "live"]


def capture_from_distill(result: dict, *, run_id: str, seen: str) -> None:
    """Route a distiller result's `ephemerals` into the default store.

    Failures here must never fail the ingest: log + continue. The embedder is
    optional — when unavailable, facts are stored unembedded (lexical recall).
    """
    try:
        ephemerals = result.get("ephemerals") or []
        if not ephemerals:
            return
        embedder = None
        try:
            from silica.agent.providers import get_embedder
            from silica.config import CONFIG

            embedder = get_embedder(CONFIG)
        except Exception:
            pass
        EpisodicStore().capture(ephemerals, run_id=run_id, seen=seen,
                                embedder=embedder)
    except Exception as e:
        logger.warning("episodic capture failed (ingest continues): %s", e)


def render(hits: list[FactHit], *, store: EpisodicStore) -> str:
    """Render recalled facts with their supersede history, dates included —
    knowledge-update and temporal-reasoning questions need the chain."""
    lines: list[str] = []
    for hit in hits:
        links = store.chain(hit.fact)
        lines.append(f"- [since {links[0].first_seen}] {links[0].text}")
        for newer, older in zip(links, links[1:]):
            lines.append(
                f"  (previously: {older.text}, {older.first_seen} to {newer.first_seen})"
            )
    return "\n".join(lines)


def episodic_home() -> Path:
    """Home vault for episodic state: CONFIG.memory_vault, default ~/.silica/vault.

    Unlike ``memory_lane.memory_vault()`` there is NO abstain rule — when the
    active vault IS the memory vault, facts still land there.
    """
    from silica.config import CONFIG

    raw = (getattr(CONFIG, "memory_vault", "") or "").strip()
    return (Path(raw).expanduser() if raw else Path.home() / ".silica" / "vault").resolve()


def store_path() -> Path:
    from silica.kernel import paths

    return paths.index_dir_for(str(episodic_home())) / "episodic.json"
