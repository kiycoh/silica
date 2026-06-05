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

logger = logging.getLogger(__name__)


def build_substrate(
    chunk: dict,
    *,
    manifest_titles: list[str],
    k: int = 6,
    tau: float = 0.0,
    exclude: set[str] | None = None,
    cleared_parents: list[dict] | None = None,
) -> str | None:
    """Return a formatted candidate list for the distiller context, or None.

    Args:
        chunk:            The current chunk dict (schema_version + batches).
        manifest_titles:  Titles already injected in this run (from RunManifest).
                          Excluded from results to avoid re-proposing known notes.
        k:                Maximum number of candidates to surface.
        tau:              Minimum cosine score threshold (0.0 = no filter).
        exclude:          Additional path stems to exclude from results.
        cleared_parents:  Forward-reference hints from validate: parent notes that
                          were referenced but don't exist yet in the vault.  These
                          are likely to be created in the current or next run and
                          should be used for wikilinks rather than new notes.

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

        from silica.kernel.cooccurrence import CooccurStore
        from silica.kernel.relatedness import related_notes_for_query

        store = EmbedStore()

        # Embedder is OPTIONAL now: if it is down, the embed leg abstains and the
        # deterministic co-occurrence leg carries the substrate on its own.
        embedder = None
        try:
            embedder = get_embedder(CONFIG)
        except Exception as _emb_e:
            logger.debug("build_substrate: embedder unavailable (%s) — co-occurrence only", _emb_e)

        # Collect concept texts (name + excerpt) from the chunk
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

        # Embed-leg query vector: centroid of chunk concepts (None when the
        # embedder is down or the index is empty — the leg then abstains).
        query_vec = None
        if embedder is not None and len(store) > 0:
            try:
                vecs = embedder.embed(texts[:8])
                if vecs:
                    dim = len(vecs[0])
                    query_vec = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
            except Exception as _ee:
                logger.debug("build_substrate: query embed failed (%s)", _ee)

        # Build exclusion set: manifest titles + caller-supplied excludes
        exclude_lower: set[str] = {t.lower() for t in manifest_titles}
        if exclude:
            exclude_lower.update(s.lower() for s in exclude)

        cooccur_store = CooccurStore(lang=CONFIG.cooccurrence_lang)
        if len(cooccur_store) == 0:
            cooccur_store = None

        related = related_notes_for_query(
            query_vec=query_vec,
            query_text="\n".join(texts[:8]),
            embed_store=store,
            cooccur_store=cooccur_store,
            k=k,
        )
        if not related:
            return None

        manifest_lower = {t.lower() for t in manifest_titles}

        lines: list[str] = []
        for r in related:
            # The cosine threshold gates only the embedding leg; pure
            # co-occurrence candidates are a different signal and pass through.
            if r.embed_score is not None and r.embed_score < tau:
                continue
            name = r.name
            path = r.path
            if not name or name.lower() in exclude_lower:
                continue

            # Graph-far flag: related but not already adjacent to run notes.
            # Light check (1-hop links of this candidate) — best-effort.
            graph_far = False
            try:
                path_with_ext = path + ".md" if not path.endswith(".md") else path
                ref = NoteRef(name=name, path=path_with_ext)
                neighbour_names = {lr.name.lower() for lr in DRIVER.links(ref)}
                graph_far = not neighbour_names.intersection(manifest_lower)
            except Exception:
                pass

            if r.embed_score is not None:
                score_label = f"score={r.embed_score:.3f}"
            else:
                score_label = f"cooccur~w{int(round(r.cooccur_weight or 0))}"
            flag = " [graph-far]" if graph_far else ""
            lines.append(f"- [[{name}]] ({score_label}){flag}")

        # Append forward-reference hints: parent notes cleared by validate because
        # they don't exist yet.  High probability of appearing in future injections —
        # the distiller should use [[name]] links to them rather than creating duplicates.
        if cleared_parents:
            seen: set[str] = set()
            fwd_lines: list[str] = []
            for cp in cleared_parents:
                name = cp.get("cleared_parent", "")
                if not name or name in seen:
                    continue
                seen.add(name)
                ref = cp.get("note_heading") or cp.get("note_path", "")
                fwd_lines.append(
                    f"- [[{name}]] ← forward-reference (not yet in vault; "
                    f"referenced as parent by '{ref}'; likely created in a future injection)"
                )
            if fwd_lines:
                if lines:
                    lines.append("")
                lines.append("## Forward-reference parents (create wikilinks, not new notes)")
                lines.extend(fwd_lines)

        return "\n".join(lines) if lines else None

    except Exception as _e:
        logger.debug("build_substrate: failed (non-fatal): %s", _e)
        return None
