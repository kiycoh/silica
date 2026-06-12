"""Deterministic co-occurrence graph.

L1 kernel: no LLM, no API, NO embedder dependency. Turns note prose into a
weighted concept graph via a sliding window with decaying weights.

This module is a CANDIDATE GENERATOR (a second PROPOSE-layer alongside
embeddings). It is never authoritative about vault structure; that role
belongs to graph_diff / the driver. Fusion with embeddings lives in a
separate relatedness facade — NOT here.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import orjson

# --- algorithm constants -------------------------------------
NARRATIVE_WEIGHT = 3
LANDSCAPE_WEIGHT = 3
GAP = 4               # effective look-back of GAP - 1 = 3 tokens
MIN_TOKEN_LEN = 3
EVIDENCE = "cooccur"

_LEGACY_INDEX_PATH = Path.home() / ".silica" / "index" / "cooccurrence.json"


def _index_path() -> Path:
    # Function, not constant: resolves per current vault; tests monkeypatch it.
    from silica.kernel import paths

    return paths.index_dir() / "cooccurrence.json"

# Compact function-word stopword sets. Filtered at BUILD time (not promoted to
# nodes); raw text is never mutated. Italian set seeds from a function-word core.
_STOPWORDS: dict[str, set[str]] = {
    "english": {
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
        "at", "by", "for", "with", "as", "is", "are", "was", "were", "be",
        "been", "being", "it", "its", "this", "that", "these", "those", "he",
        "she", "they", "we", "you", "his", "her", "their", "our", "your",
        "from", "into", "than", "then", "so", "not", "no", "do", "does", "did",
        "has", "have", "had", "can", "could", "would", "should", "will", "shall",
        "may", "might", "must", "about", "which", "who", "whom", "what", "when",
        "where", "how", "why", "all", "any", "some", "such", "more", "most",
    },
    "italian": {
        "di", "da", "in", "con", "su", "per", "tra", "fra", "a", "e", "o", "ma",
        "se", "anche", "come", "il", "lo", "la", "i", "gli", "le", "un", "uno",
        "una", "del", "dello", "della", "dei", "degli", "delle", "al", "allo",
        "alla", "ai", "agli", "alle", "dal", "dalla", "nel", "nella", "sul",
        "sulla", "che", "chi", "cui", "non", "ne", "ci", "vi", "si", "ho", "hai",
        "ha", "abbiamo", "hanno", "sono", "sei", "siamo", "siete", "era", "essere",
        "questo", "questa", "questi", "queste", "quello", "quella", "suo", "sua",
        "loro", "nostro", "vostro", "mio", "tuo",
    },
}

_SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")
_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")

# Cache stemmers per language (snowballstemmer objects are reusable).
_STEMMERS: dict[str, Any] = {}


def _get_stemmer(lang: str) -> Any:
    if lang not in _STEMMERS:
        import snowballstemmer
        _STEMMERS[lang] = snowballstemmer.stemmer(lang)
    return _STEMMERS[lang]


def _split_sentences(text: str) -> list[str]:
    """Split on sentence terminators and newlines; drop empties/whitespace."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def tokenize(text: str, lang: str = "english") -> list[list[tuple[str, str]]]:
    """Return sentences, each a list of (stem, surface) token pairs.

    Pipeline per sentence: extract word tokens → lowercase → drop stopwords and
    tokens shorter than MIN_TOKEN_LEN → stem (Snowball). The window never
    crosses a sentence boundary.
    """
    stopwords = _STOPWORDS.get(lang, set())
    stemmer = _get_stemmer(lang)
    out: list[list[tuple[str, str]]] = []
    for sentence in _split_sentences(text):
        toks: list[tuple[str, str]] = []
        for raw in _TOKEN_RE.findall(sentence):
            surface = raw.lower()
            if len(surface) < MIN_TOKEN_LEN or surface in stopwords:
                continue
            toks.append((stemmer.stemWord(surface), surface))
        if toks:
            out.append(toks)
    return out


def build_contribution(
    name: str,
    body: str,
    lang: str = "english",
    concepts: list[str] | None = None,
) -> dict[str, Any]:
    """Turn a note's text into its per-note co-occurrence contribution.

    Strips frontmatter + media, tokenizes per sentence, then for each token
    links it back to the previous `GAP - 1` tokens with a decaying weight:
        distance 1 -> 3 (narrative), distance 2 -> 2, distance 3 -> 1 (gap scan)
    Edges are directed earlier->later. Weights accumulate.

    `concepts` (#9, Marwitz et al. 2026): optional LLM-extracted, normalized
    concept phrases. Each is appended as its own sentence so its stems become
    nodes and its words co-occur through the same window logic — reinforcing
    LLM-validated concepts above rule-based body noise. `None`/`[]` leaves the
    contribution byte-identical to a body-only build (graceful degradation).
    """
    from silica.kernel import frontmatter
    from silica.kernel.media import preprocess_text

    _data, _fm, body_only = frontmatter.split(body) if body else (None, "", "")
    text = f"{name}\n\n{preprocess_text(body_only)}"
    if concepts:
        concept_sentences = ". ".join(c.strip() for c in concepts if c and c.strip())
        if concept_sentences:
            text = f"{text}\n\n{concept_sentences}."

    node_surface_counts: dict[str, dict[str, int]] = {}
    edge_acc: dict[tuple[str, str], float] = {}

    for sentence in tokenize(text, lang=lang):
        stems = [stem for (stem, _s) in sentence]
        for i, (stem, surface) in enumerate(sentence):
            node_surface_counts.setdefault(stem, {})
            node_surface_counts[stem][surface] = node_surface_counts[stem].get(surface, 0) + 1
            # link back up to GAP - 1 tokens with decaying weight (3, 2, 1)
            for dist in range(1, GAP):
                j = i - dist
                if j < 0:
                    break
                src = stems[j]
                if src == stem:
                    continue
                weight = float(NARRATIVE_WEIGHT + 1 - dist)  # 3, 2, 1
                if weight <= 0:
                    continue
                edge_acc[(src, stem)] = edge_acc.get((src, stem), 0.0) + weight

    nodes = {
        stem: {
            "label": max(surfaces.items(), key=lambda kv: kv[1])[0],
            "count": sum(surfaces.values()),
        }
        for stem, surfaces in node_surface_counts.items()
    }
    edges = [[f, t, w] for (f, t), w in edge_acc.items()]
    return {"nodes": nodes, "edges": edges}


class CooccurStore:
    """orjson-backed store of per-note co-occurrence contributions.

    The global graph is a lazy, cached aggregation over the per-note
    contributions (mirrors EmbedStore's lazy search matrix). Storing per-note
    contributions makes refresh a dict replacement — no fragile weight
    subtraction.
    """

    def __init__(self, path: Path | None = None, lang: str = "english"):
        self._path = path if path is not None else _index_path()
        self._notes: dict[str, dict[str, Any]] = {}
        self.lang = lang
        # lazy aggregated graph caches (scope=None only)
        self._adj: dict[str, dict[str, float]] | None = None
        self._labels: dict[str, str] | None = None
        self._load()

    # --- caches ---
    def _invalidate(self) -> None:
        self._adj = None
        self._labels = None

    # --- I/O ---
    def _load(self) -> None:
        self._invalidate()
        src = self._path
        if not src.exists() and src != _LEGACY_INDEX_PATH and _LEGACY_INDEX_PATH.exists():
            src = _LEGACY_INDEX_PATH  # one-time soft migration: copied forward on next save()
        if src.exists():
            try:
                data = orjson.loads(src.read_bytes())
                self._notes = data.get("notes", {})
                self.lang = data.get("lang", self.lang)
            except Exception:
                self._notes = {}

    def save(self) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(orjson.dumps(
            {"version": 1, "lang": self.lang, "notes": self._notes},
            option=orjson.OPT_INDENT_2,
        ))
        return self._path

    # --- mutation ---
    def upsert_note(self, path: str, contribution: dict[str, Any]) -> None:
        self._notes[path] = contribution
        self._invalidate()

    def delete_note(self, path: str) -> None:
        self._notes.pop(path, None)
        self._invalidate()

    # --- lookup ---
    def paths(self) -> list[str]:
        return list(self._notes.keys())

    def __len__(self) -> int:
        return len(self._notes)

    def note_nodes(self, path: str) -> dict[str, int]:
        """Return {stem: count} for one note's contribution ({} if absent).

        Public read access used by the relatedness facade to build the
        concept->notes inverted index (granularity reconciliation lives there,
        not here).
        """
        contrib = self._notes.get(path)
        if not contrib:
            return {}
        return {
            stem: int(meta.get("count", 1))
            for stem, meta in contrib.get("nodes", {}).items()
        }

    # --- aggregation (lazy, scope=None cached) ---
    def _aggregate(self, scope: str | None = None) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
        """Build undirected adjacency {stem: {neighbor: weight}} + label map.

        scope, if given, restricts to notes whose path == scope or starts with
        scope + "/" (folder scoping, context-level filtering).
        scope=None results are cached; scoped results are computed fresh.
        """
        if scope is None and self._adj is not None and self._labels is not None:
            return self._adj, self._labels

        def _in_scope(p: str) -> bool:
            if not scope:
                return True
            s = scope.strip("/").lower()
            pp = p.strip("/").lower()
            return pp == s or pp.startswith(s + "/")

        adj: dict[str, dict[str, float]] = {}
        label_counts: dict[str, dict[str, int]] = {}
        for path, contrib in self._notes.items():
            if not _in_scope(path):
                continue
            for stem, meta in contrib.get("nodes", {}).items():
                lc = label_counts.setdefault(stem, {})
                lc[meta["label"]] = lc.get(meta["label"], 0) + int(meta.get("count", 1))
            for f, t, w in contrib.get("edges", []):
                adj.setdefault(f, {})[t] = adj.setdefault(f, {}).get(t, 0.0) + w
                adj.setdefault(t, {})[f] = adj.setdefault(t, {}).get(f, 0.0) + w
        labels = {
            stem: max(surfaces.items(), key=lambda kv: kv[1])[0]
            for stem, surfaces in label_counts.items()
        }
        if scope is None:
            self._adj, self._labels = adj, labels
        return adj, labels

    def node_label(self, stem: str) -> str:
        _adj, labels = self._aggregate()
        return labels.get(stem, stem)

    def to_networkx(self, scope: str | None = None):
        """Return an undirected weighted nx.Graph for consumers (centrality,
        delta-vs-wikilink in graph_report). Built from aggregated adjacency.
        """
        import networkx as nx

        adj, _labels = self._aggregate(scope=scope)
        G = nx.Graph()
        for stem, nbrs in adj.items():
            G.add_node(stem)
            for nb, w in nbrs.items():
                # undirected: add once; adjacency already symmetric
                if not G.has_edge(stem, nb):
                    G.add_edge(stem, nb, weight=w)
        return G

    def neighbors(self, concept: str, k: int = 10, scope: str | None = None) -> list[dict[str, Any]]:
        """Top-k co-occurring concepts for `concept`. Returns [] if absent.

        Never raises and never touches the embedder — this is the stable leg of
        the relatedness fusion (works with LM Studio down).
        """
        concept = concept.strip()
        if not concept:
            return []
        stem = _get_stemmer(self.lang).stemWord(concept.lower())
        adj, labels = self._aggregate(scope=scope)
        nbrs = adj.get(stem)
        if not nbrs:
            return []
        ranked = sorted(nbrs.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            {"concept": labels.get(nb_stem, nb_stem), "weight": w, "evidence": EVIDENCE}
            for nb_stem, w in ranked[:k]
        ]


def build_index(
    notes: list[tuple[str, str, str]],
    *,
    store: CooccurStore | None = None,
    lang: str | None = None,
    force: bool = False,
    concepts_by_path: dict[str, list[str]] | None = None,
) -> CooccurStore:
    """Build/refresh the co-occurrence store from (path, name, body) tuples.

    Mirrors embed.build_index. Incremental: skips notes already present unless
    `force`. Returns the saved store.

    `concepts_by_path` (#9): optional map of note path -> LLM-extracted concept
    phrases, forwarded into build_contribution to reinforce those concepts.
    """
    if store is None:
        store = CooccurStore(lang=lang or "english")
    use_lang = lang or store.lang
    cmap = concepts_by_path or {}
    for path, name, body in notes:
        if not force and path in store.paths():
            continue
        store.upsert_note(
            path,
            build_contribution(name, body, lang=use_lang, concepts=cmap.get(path)),
        )
    store.save()
    return store


def refresh_note(
    path: str,
    name: str,
    body: str,
    *,
    store: CooccurStore | None = None,
    lang: str | None = None,
) -> CooccurStore:
    """Re-build a single note's contribution and persist (freshness hook).

    Replacement, not accumulation: the note's prior contribution is overwritten,
    so weights never inflate on re-processing.
    """
    if store is None:
        store = CooccurStore(lang=lang or "english")
    use_lang = lang or store.lang
    store.upsert_note(path, build_contribution(name, body, lang=use_lang))
    store.save()
    return store
