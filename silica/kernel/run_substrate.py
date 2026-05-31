"""Per-chunk semantic substrate builder — Block 4 / Phase 3+ of the plan.

embeddings PROPOSE — graph DISPOSES.

build_substrate() generates a compact '## Related Notes (candidates)' section
for the distiller context so the model can choose a `parent` from notes that
are semantically close to the current chunk but not yet directly linked in the
graph.  A 'graph-far' flag marks such candidates: high cosine but not a direct
link of any note already written in this run.

The function is best-effort: returns None on any error (embedder down, empty
index) so callers can safely skip the section.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def build_substrate(
    chunk: dict,
    *,
    manifest_titles: list[str],
    k: int = 6,
    tau: float = 0.0,
    exclude: set[str] | None = None,
) -> str | None:
    """Return a formatted candidate list for the distiller context, or None.

    Args:
        chunk:           The current chunk dict (schema_version + batches).
        manifest_titles: Titles already injected in this run (from RunManifest).
                         Excluded from results to avoid re-proposing known notes.
        k:               Maximum number of candidates to surface.
        tau:             Minimum cosine score threshold (0.0 = no filter).
        exclude:         Additional path stems to exclude from results.

    Returns:
        Formatted string for the '## Related Notes (candidates)' section,
        or None if the substrate cannot be built.
    """
    try:
        from silica.agent.providers import get_embedder
        from silica.config import CONFIG
        from silica.kernel.embed import EmbedStore
        from silica.driver import DRIVER
        from silica.driver.base import NoteRef

        store = EmbedStore()
        if len(store) == 0:
            return None

        embedder = get_embedder(CONFIG)

        # Collect concept texts (name + excerpt) from the chunk for query embedding
        texts: list[str] = []
        for batch in chunk.get("batches", []):
            for c in batch.get("concepts", []):
                name = c.get("name", "") if isinstance(c, dict) else str(c)
                excerpt = c.get("inbox_excerpt", "") if isinstance(c, dict) else ""
                combined = f"{name}\n{excerpt[:300]}" if excerpt else name
                if combined.strip():
                    texts.append(combined)

        if not texts:
            return None

        vecs = embedder.embed(texts[:8])
        if not vecs:
            return None

        # Centroid of chunk concepts as the query vector
        dim = len(vecs[0])
        query_vec = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

        # Build exclusion set: manifest titles + caller-supplied excludes
        exclude_lower: set[str] = {t.lower() for t in manifest_titles}
        if exclude:
            exclude_lower.update(s.lower() for s in exclude)

        results = store.cosine_top_k(query_vec, k=k)
        if not results:
            return None

        manifest_lower = {t.lower() for t in manifest_titles}

        lines: list[str] = []
        for r in results:
            score = r.get("score", 0.0)
            if score < tau:
                continue
            name = r.get("name", "")
            path = r.get("path", "")
            if not name or name.lower() in exclude_lower:
                continue

            # Graph-far flag: semantically close but not already adjacent to run notes.
            # Light check (1-hop links of this candidate) — best-effort.
            graph_far = False
            try:
                path_with_ext = path + ".md" if not path.endswith(".md") else path
                ref = NoteRef(name=name, path=path_with_ext)
                neighbour_names = {lr.name.lower() for lr in DRIVER.links(ref)}
                # graph-far = this candidate doesn't directly link to any run note
                graph_far = not neighbour_names.intersection(manifest_lower)
            except Exception:
                pass

            flag = " [graph-far]" if graph_far else ""
            lines.append(f"- [[{name}]] (score={score:.3f}){flag}")

        return "\n".join(lines) if lines else None

    except Exception as _e:
        logger.debug("build_substrate: failed (non-fatal): %s", _e)
        return None
