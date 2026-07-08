# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Vault map — a compact semantic self-model of the corpus for recall at session start.

CoALA: consolidates the persistent co-occurrence index + the folder structure
into a short Markdown block, injected into working memory at startup, so the
agent starts oriented instead of rediscovering the vault via tools.

Deterministic, zero LLM. Best-effort: any sub-block that fails is omitted;
an empty vault or cooccur index → None (the caller injects nothing).
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silica.kernel.cooccurrence import CooccurStore

logger = logging.getLogger(__name__)


def build_vault_map(
    *,
    store: "CooccurStore | None" = None,
    max_folders: int = 8,
    max_clusters: int = 8,
    max_vocab: int = 15,
    max_hubs: int = 8,
    max_contested: int = 8,
    log_tail: int = 5,
) -> str | None:
    try:
        from silica.config import CONFIG
        from silica.kernel.cooccurrence import get_cooccur_store

        store = store if store is not None else get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        if len(store) == 0:
            return None

        lines: list[str] = ["## Vault map  (corpus snapshot at session start)"]


        # A single vault pass: refs feeds both the folders block and the
        # contested scan (one list_files call, not two).
        refs: list = []
        try:
            from silica.driver import DRIVER

            refs = DRIVER.list_files()
        except Exception as e:  # best-effort
            logger.debug("build_vault_map: list_files failed: %s", e)

        # Note count + top folders
        try:
            if refs:
                folder_counts: Counter[str] = Counter(
                    (r.path.rsplit("/", 1)[0] if "/" in r.path else "(root)")
                    for r in refs
                    if getattr(r, "path", "")
                )
                lines.append(f"- Notes: {len(refs)} in {len(folder_counts)} folders")
                top = folder_counts.most_common(max_folders)
                if top:
                    lines.append(
                        "- Top folders: "
                        + ", ".join(f"{f} ({c})" for f, c in top)
                    )
        except Exception as e:  # best-effort
            logger.debug("build_vault_map: folders block skipped: %s", e)

        # Contested notes (spec-hermes-coherence §1 leftover): frontmatter
        # `contested: true`, same scan pattern as graph_report/compute.py
        # but via props_of (frontmatter-only, no body) — embedder-free,
        # kernel-only. No line emitted if N == 0.
        try:
            from silica.driver import DRIVER

            contested_names: list[str] = []
            for ref in refs:
                try:
                    props = DRIVER.props_of(ref)
                except Exception:
                    continue
                if props and props.get("contested"):
                    contested_names.append(
                        ref.path.rsplit("/", 1)[-1].removesuffix(".md")
                    )
            if contested_names:
                shown = ", ".join(f"[[{n}]]" for n in contested_names[:max_contested])
                extra = len(contested_names) - max_contested
                if extra > 0:
                    shown += f" … +{extra}"
                lines.append(f"⚠ {len(contested_names)} contested notes: {shown}")
        except Exception as e:  # best-effort
            logger.debug("build_vault_map: contested block skipped: %s", e)

        # Top clusters (Louvain over the concept graph; each community is
        # labelled by its highest-weight stems — community_labels must NOT be
        # used here: it wants communities of note paths, not of stems).
        try:
            from networkx.algorithms.community import louvain_communities

            G = store.to_networkx()
            if G.number_of_nodes():
                deg = dict(G.degree(weight="weight"))
                communities = sorted(
                    louvain_communities(G, seed=42), key=len, reverse=True
                )
                cluster_labels: list[str] = []
                for members in communities[:max_clusters]:
                    top = sorted(
                        members, key=lambda s: deg.get(s, 0.0), reverse=True
                    )[:2]
                    label = " · ".join(store.node_label(s) for s in top)
                    if label:
                        cluster_labels.append(label)
                if cluster_labels:
                    lines.append(
                        "- Top clusters: " + ", ".join(cluster_labels)
                    )
        except Exception as e:  # networkx missing or empty graph → skip
            logger.debug("build_vault_map: cluster block skipped: %s", e)

        # Core vocabulary
        try:
            stems = store.top_stems(max_vocab)
            if stems:
                lines.append("- Core vocabulary: " + ", ".join(stems))
        except Exception as e:
            logger.debug("build_vault_map: vocabulary block skipped: %s", e)

        # Hub notes — proxy: notes that touch the most distinct concepts
        try:
            ranked = sorted(
                store.paths(),
                key=lambda p: len(store.note_nodes(p)),
                reverse=True,
            )[:max_hubs]
            hub_names = [p.rsplit("/", 1)[-1].removesuffix(".md") for p in ranked]
            if hub_names:
                lines.append(
                    "- Hub notes: " + ", ".join(f"[[{h}]]" for h in hub_names)
                )
        except Exception as e:
            logger.debug("build_vault_map: hub block skipped: %s", e)

        # Tail of log.md — the agent sees what happened recently without
        # having to open the run JSON (Task 2: human-readable append-only journal).
        try:
            from silica.kernel.run_log import tail_log

            recent = tail_log(log_tail)
            if recent:
                lines.append("- Recent log:")
                lines.extend(f"  {ln}" for ln in recent)
        except Exception as e:
            logger.debug("build_vault_map: log block skipped: %s", e)

        # Only the header → nothing useful: behave like an empty vault.
        if len(lines) == 1:
            return None
        return "\n".join(lines)

    except Exception as e:  # ponytail: the map must never break the session
        logger.debug("build_vault_map: failed (non-fatal): %s", e)
        return None
