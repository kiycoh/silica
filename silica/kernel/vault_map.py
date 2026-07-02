"""Vault map — self-model semantico compatto del corpus per il recall a inizio sessione.

CoALA: consolida l'indice di co-occorrenza persistente + la struttura delle
cartelle in un blocco Markdown breve, iniettato in working memory all'avvio,
così l'agente parte orientato invece di ri-scoprire il vault via tool.

Deterministico, zero LLM. Best-effort: ogni sotto-blocco che fallisce viene
omesso; vault o indice cooccur vuoto → None (il chiamante non inietta nulla).
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

        lines: list[str] = [
            "## Vault map  (orientamento auto-generato; "
            "puo' non riflettere le scritture di questa sessione)"
        ]

        # Un solo giro di vault: refs alimenta sia il blocco cartelle sia lo
        # scan contested (una sola list_files, non due).
        refs: list = []
        try:
            from silica.driver import DRIVER

            refs = DRIVER.list_files()
        except Exception as e:  # best-effort
            logger.debug("build_vault_map: list_files fallito: %s", e)

        # Conteggio note + cartelle principali
        try:
            if refs:
                folder_counts: Counter[str] = Counter(
                    (r.path.rsplit("/", 1)[0] if "/" in r.path else "(root)")
                    for r in refs
                    if getattr(r, "path", "")
                )
                lines.append(f"- Note: {len(refs)} in {len(folder_counts)} cartelle")
                top = folder_counts.most_common(max_folders)
                if top:
                    lines.append(
                        "- Cartelle principali: "
                        + ", ".join(f"{f} ({c})" for f, c in top)
                    )
        except Exception as e:  # best-effort
            logger.debug("build_vault_map: blocco cartelle saltato: %s", e)

        # Note contestate (spec-hermes-coherence §1 residuo): frontmatter
        # `contested: true`, stesso pattern di scan di graph_report/compute.py
        # ma via props_of (frontmatter-only, niente body) — embedder-free,
        # kernel-only. Nessuna riga se N == 0.
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
                lines.append(f"⚠ {len(contested_names)} note contestate: {shown}")
        except Exception as e:  # best-effort
            logger.debug("build_vault_map: blocco contested saltato: %s", e)

        # Cluster principali (Louvain sul grafo concetti; ogni community e'
        # etichettata dai suoi stem a peso maggiore — community_labels NON va
        # usato qui: vuole community di path di note, non di stem).
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
                        "- Cluster principali: " + ", ".join(cluster_labels)
                    )
        except Exception as e:  # networkx assente o grafo vuoto → salta
            logger.debug("build_vault_map: blocco cluster saltato: %s", e)

        # Vocabolario centrale
        try:
            stems = store.top_stems(max_vocab)
            if stems:
                lines.append("- Vocabolario centrale: " + ", ".join(stems))
        except Exception as e:
            logger.debug("build_vault_map: blocco vocabolario saltato: %s", e)

        # Note hub — proxy: note che toccano piu' concetti distinti
        try:
            ranked = sorted(
                store.paths(),
                key=lambda p: len(store.note_nodes(p)),
                reverse=True,
            )[:max_hubs]
            hub_names = [p.rsplit("/", 1)[-1].removesuffix(".md") for p in ranked]
            if hub_names:
                lines.append(
                    "- Note hub: " + ", ".join(f"[[{h}]]" for h in hub_names)
                )
        except Exception as e:
            logger.debug("build_vault_map: blocco hub saltato: %s", e)

        # Coda del log.md — l'agente vede cosa e' successo di recente senza
        # dover aprire il JSON dei run (Task 2: journal umano append-only).
        try:
            from silica.kernel.run_log import tail_log

            recent = tail_log(log_tail)
            if recent:
                lines.append("- Log recente:")
                lines.extend(f"  {ln}" for ln in recent)
        except Exception as e:
            logger.debug("build_vault_map: blocco log saltato: %s", e)

        # Solo l'header → niente di utile: comportati come vault vuoto.
        if len(lines) == 1:
            return None
        return "\n".join(lines)

    except Exception as e:  # ponytail: la mappa non deve mai rompere la sessione
        logger.debug("build_vault_map: fallito (non-fatale): %s", e)
        return None
