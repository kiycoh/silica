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
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from silica.kernel.embed import _cosine  # noqa: F401 — shared helper, re-exported for probes

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
    # to hundreds of facts; the 10k scaling fix targeted note stores. Pack npz
    # like note stores if a store ever holds ~10^3+ facts or load/save drags.
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


# Change-marker token stems: models bake the CHANGE into the key
# ("aspiration_reinforced", "job_update", "trip.new") despite the prompt's
# key-discipline block — LoCoMo smoke 2026-07-18 showed the instruction being
# ignored with the clean key in view. Folding these at lookup time makes the
# variant MATCH the clean head so supersede chains stay whole.
_CHANGE_MARKER_STEMS = frozenset({"reinforc", "reaffirm", "updat", "new", "chang"})
_VERSION_TOKEN_RE = re.compile(r"v\d+$")


def normalize_key(key: str) -> str:
    """Canonical `entity.attribute` form for MATCHING: casefold, then
    snowball-stem every `_`-token of every `.` segment, dropping change-marker
    tokens (`_reinforced`/`_update`/`.new`/`v2` — the change belongs in the
    text, supersede encodes it). Merges morphological key drift
    (`model_kits.gifts` == `model_kit.gift`); semantic synonyms stay distinct.
    Stored keys are never rewritten — this is lookup identity only.
    """
    # ponytail: lang hardcoded to english — the distiller prompt shapes keys
    # as English-style slugs; non-English-keyed stores under-merge. Upgrade
    # path: a per-store language knob, when a non-English store shows drift.
    from silica.kernel.text import stem_word

    segs: list[str] = []
    for seg in key.casefold().split("."):
        toks = [stem_word(t, lang="english") for t in seg.split("_") if t]
        kept = [t for t in toks
                if t not in _CHANGE_MARKER_STEMS and not _VERSION_TOKEN_RE.fullmatch(t)]
        if kept:
            segs.append("_".join(kept))
    if not segs:  # a key that is nothing but markers: fall back unfiltered
        return ".".join("_".join(stem_word(t, lang="english") for t in s.split("_") if t)
                        for s in key.casefold().split(".") if s.strip("_"))
    return ".".join(segs)


_ENTITY_PREFIXES = {"user", "assist"}  # canonical forms of user. / assistant.


def enforce_key_schema(key: str, schema) -> str:
    """Structural write-time enforcement of the declared key grammar
    (ADR-0021): unknown first segment folds under `schema.default_prefix`,
    segments beyond `schema.max_depth` fold into the last one. Never rejects.

    Distinct from `normalize_key` (lookup-only matching identity): stored
    keys are shaped here but never stemmed — the spelling survives.
    """
    segs = [s for s in key.split(".") if s]
    if not segs:
        return key
    canonical = {normalize_key(p) for p in schema.prefixes}
    if normalize_key(segs[0]) not in canonical:
        segs.insert(0, schema.default_prefix)
    if len(segs) > schema.max_depth:
        segs = segs[:schema.max_depth - 1] + ["_".join(segs[schema.max_depth - 1:])]
    return ".".join(segs)


def _snap_entity(key: str) -> tuple:
    """Entity namespace of a key, the HARD constraint of the snap fallback:
    cosine may never merge across entities (same attribute of two people
    embeds close by construction — superseding across them falsifies
    history). `user.<name>.*` keys are per-person; any other first segment
    is the entity itself (so assistant.* observations may chain across
    sessions)."""
    segs = [s for s in key.casefold().split(".") if s]
    if not segs:
        return ()
    if segs[0] == "user" and len(segs) > 1:
        return ("user", segs[1])
    return (segs[0],)


def key_tokens(key: str) -> set[str]:
    """Stemmed tokens of a key, entity prefix dropped: the shared alphabet
    of the eval key-drift/clustering probes."""
    segs = normalize_key(key).split(".")
    if len(segs) > 1 and segs[0] in _ENTITY_PREFIXES:
        segs = segs[1:]
    return {t for s in segs for t in s.split("_") if len(t) > 1}


def rare_token_components(keys: list[str], *,
                          max_df: int | None = None) -> dict[str, str]:
    """Connected components over shared key tokens: key -> root key.

    With ``max_df``, a token forms edges only while its document frequency
    over the (deduplicated) key set stays <= max_df; None means no filter
    (the naive blob view kept for diagnostics). Pure function of the key
    set: deterministic and order-independent."""
    keys = sorted(set(keys))
    toks = {k: key_tokens(k) for k in keys}
    if max_df is not None:
        df: dict[str, int] = {}
        for ts in toks.values():
            for t in ts:
                df[t] = df.get(t, 0) + 1
        toks = {k: {t for t in ts if df[t] <= max_df} for k, ts in toks.items()}
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    owner: dict[str, str] = {}
    for k in keys:
        for t in toks[k]:
            if t in owner:
                parent[find(k)] = find(owner[t])
            else:
                owner[t] = k
    return {k: find(k) for k in keys}


def key_vocabulary(store: "EpisodicStore", *, cap: int = 60) -> list[str]:
    """Raw keys of live heads, most recently seen first, capped.

    Feeds the distiller's `## Episodic keys` context section so capture snaps
    to the established vocabulary instead of coining synonym keys."""
    heads = sorted(store.live_facts(), key=lambda f: f.last_seen, reverse=True)
    return [f.key for f in heads[:cap]]


def key_vocabulary_section(store: "EpisodicStore") -> str | None:
    """`## Episodic keys` distiller-context section; None on an empty store."""
    keys = key_vocabulary(store)
    if not keys:
        return None
    return (
        "## Episodic keys\n"
        "Live ephemeral keys already in the store. When a fact concerns one "
        "of these attributes, reuse that exact key instead of coining a new "
        "one:\n" + ", ".join(keys)[:600]  # hard token-budget cap
    )


class EpisodicStore:
    """JSON-file-backed fact store. Facts are not notes; they nucleate INTO notes."""

    def __init__(self, path: Path | None = None):
        self.path = path if path is not None else store_path()
        self.next_id = 1
        self.facts: list[Fact] = []
        self._key_vecs: dict[str, list[float]] = {}  # spaced key -> vec (snap cache)
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
                embedder=None, schema=None, snap_tau: float = 0.0) -> None:
        """Merge distiller ephemerals into the store. Mechanical, no LLM.

        Same key + same normalized text reinforces (last_seen, runs); same key
        + different text supersedes the live head; a new key starts a chain.
        New/changed facts are embedded when `embedder` is served; embedding
        failure is silent (recall falls back to lexical).

        `snap_tau` > 0 arms the fallback of the matcher cascade (canonical
        keys, fase 1/2): a coined key with no exact canonical match joins the
        nearest live head by KEY-embedding cosine >= snap_tau. 0 (default)
        is bit-identical to the pre-cascade store.

        `schema` (ADR-0021): an `EpisodicKeySchema` shapes stored keys via
        `enforce_key_schema` before merge; None means no enforcement —
        bit-identical to before the schema existed (frozen-store replays and
        A/B baselines depend on this default).
        """
        # Heads keyed by canonical form: keys written before Layer A still
        # match variant arrivals. On a legacy collision (two live heads with
        # the same canonical form) the later chain wins the lookup; TTL
        # retires the other.
        heads = {normalize_key(f.key): f for f in self.facts if f.status == "live"}
        created: list[Fact] = []
        folded = 0
        for raw in facts:
            key = (raw.get("key") or "").strip()
            text = (raw.get("text") or "").strip()
            if not key or not text:
                continue
            if schema is not None:
                shaped = enforce_key_schema(key, schema)
                folded += shaped != key
                key = shaped
            nkey = normalize_key(key)
            head = heads.get(nkey)
            if head is None and snap_tau > 0 and embedder is not None:
                head = self._snap_head(key, heads, embedder, snap_tau)
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
                old_nkey = normalize_key(head.key)
                if old_nkey != nkey and heads.get(old_nkey) is head:
                    del heads[old_nkey]  # snap join: retire the stale lookup
            self.facts.append(fact)
            heads[nkey] = fact
            created.append(fact)
        if embedder is not None and created:
            try:
                vecs = embedder.embed([f.text for f in created])
                for fact, vec in zip(created, vecs):
                    fact.vec = list(vec)
            except Exception as e:
                logger.debug("episodic capture: embedding skipped (%s)", e)
        if folded:
            logger.debug("episodic capture: %d key(s) schema-folded", folded)
        self.save()

    def _snap_head(self, key: str, heads: dict[str, Fact], embedder,
                   tau: float) -> Fact | None:
        """Fallback arm of the capture matcher cascade (canonical keys):
        nearest live head by KEY-embedding cosine, joined when >= tau. Keys
        embed spaced (`a.b_c` -> "a b c") and NEVER the fact text — text mixes
        attribute with value and kills knowledge-update chains (probe-gated,
        bench/locomo_embed_identity_gates.md). Embedding failure is silent:
        the arrival simply starts its own chain."""
        ent = _snap_entity(key)
        candidates = [h for h in heads.values() if _snap_entity(h.key) == ent]
        if not candidates:
            return None
        spaced = {k: k.replace(".", " ").replace("_", " ")
                  for k in (key, *(h.key for h in candidates))}
        missing = [k for k in spaced if spaced[k] not in self._key_vecs]
        # ponytail: one embed batch per fallback arrival, cache per store
        # instance; persist the cache in the store file if capture gets hot.
        if missing:
            try:
                vecs = embedder.embed([spaced[k] for k in missing])
            except Exception as e:
                logger.debug("episodic snap: embedding skipped (%s)", e)
                return None
            for k, v in zip(missing, vecs):
                self._key_vecs[spaced[k]] = list(v)
        kv = self._key_vecs[spaced[key]]
        best, best_cos = None, 0.0
        for h in candidates:
            c = _cosine(kv, self._key_vecs[spaced[h.key]])
            if c > best_cos:
                best, best_cos = h, c
        return best if best_cos >= tau else None

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
        # ADR-0021: the key schema is owned by the MEMORY vault (the store's
        # home), never by the vault active at capture. Absent block ⇒ None ⇒
        # no enforcement.
        schema = None
        try:
            from silica.kernel.vault_manifest import load_manifest

            schema = load_manifest(episodic_home()).conventions.episodic_keys
        except Exception:
            pass
        snap_tau = 0.0
        try:
            from silica.config import CONFIG

            snap_tau = float(getattr(CONFIG, "episodic_embed_snap_tau", 0.0))
        except Exception:
            pass
        EpisodicStore().capture(ephemerals, run_id=run_id, seen=seen,
                                embedder=embedder, schema=schema,
                                snap_tau=snap_tau)
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
