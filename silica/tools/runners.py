# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Runner tools — whole pipelines and leashed sub-agent batches.

Full FSM runs (injector, organizer), taxonomy generation, run-ledger
inspection, and the dedup/refine/enrich sub-agent passes.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool
from silica.tools.graph import _in_folder


class RunInjectorArgs(BaseModel):
    inbox_file: str = Field(default="", description="Path to a single inbox file (legacy; use inbox_files for multiple files)")
    inbox_files: list[str] = Field(default_factory=list, description="Paths to one or more inbox files to nucleate in a single run")
    target_dir: str = Field(description="Destination directory for the extracted concepts")
    hub: str = Field(default="", description="Optional reference hub note")
    resume_run_id: str = Field(default="", description="Run ID to resume (re-processes only failed chunks, skips done ones)")

@tool(RunInjectorArgs, cls="composed", collapse="eager")
def silica_run_injector(
    inbox_file: str = "",
    inbox_files: list[str] | None = None,
    target_dir: str = "",
    hub: str = "",
    resume_run_id: str = "",
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Nucleate one or more inbox files into the vault — the full pipeline with
    quality gates and rollback. This is THE tool for "nucleate/inject this file".

    Per-chunk failure containment: a failed chunk is rolled back and marked
    'failed' while the remaining chunks continue. Pass resume_run_id to re-run
    only the chunks that failed in a previous partial run. Near-duplicate
    merging runs concurrently in the background.
    For a single quick note, silica_write_note/silica_patch_note are cheaper.
    """
    from silica.router.coordinator import Coordinator

    files: list[str] = list(inbox_files or [])
    if inbox_file and inbox_file not in files:
        files.insert(0, inbox_file)
    if not files:
        return {"error": "No inbox file(s) specified"}

    coordinator = Coordinator(
        inbox_files=files,
        target_dir=target_dir,
        hub=hub or None,
        resume_run_id=resume_run_id or None,
        cancel_token=cancel_token,
    )
    return coordinator.run()


class LedgerDigestArgs(BaseModel):
    run_id: str = Field(default="", description="Run ID to inspect (latest saved run if empty)")

@tool(LedgerDigestArgs, cls="composed")
def silica_ledger_digest(run_id: str = "") -> dict[str, Any]:
    """Compact summary of a run's plan and progress (< 500 tokens).

    Use to inspect an nucleate/audit run before advancing it with
    silica_ledger_next. Pass run_id="" for the most recently saved run.
    """
    from silica.kernel.progress import ProgressLedger, latest_run_id

    resolved_id = run_id.strip() or (latest_run_id() or "")
    if not resolved_id:
        return {"error": "No runs found in ~/.silica/runs/"}

    try:
        ledger = ProgressLedger.load(resolved_id)
    except FileNotFoundError:
        return {"error": f"Run '{resolved_id}' not found"}
    except Exception as e:
        return {"error": f"Failed to load ledger: {e}"}

    return {"run_id": resolved_id, "digest": ledger.digest()}


class DedupPairsArgs(BaseModel):
    pairs: list[dict] = Field(description="List of duplicate pairs to merge. Each dict must have 'source' and 'target' keys.")

@tool(DedupPairsArgs, cls="composed")
def silica_dedup_pairs(pairs: list[dict]) -> dict[str, Any]:
    """Merge an ALREADY-KNOWN list of duplicate note pairs (e.g. from a ledger task).

    The smaller note's genuinely-new info is appended to the larger note as a
    single patch. To discover duplicate pairs by scanning, use silica_dedup.
    """
    from silica.kernel.workqueue import WorkItem
    from silica.agent.subagent import run_subagent_batch
    
    if not pairs:
        return {"error": "No pairs provided."}
        
    items: list[WorkItem] = []
    
    for pair in pairs:
        source = pair.get("source")
        target = pair.get("target")
        score = pair.get("score", 0.0)
        
        if not source or not target:
            continue
            
        try:
            body_src = DRIVER.read_note(source).content or ""
            body_tgt = DRIVER.read_note(target).content or ""
        except Exception:
            continue
            
        # The larger note is the merge target; the smaller is the source of new info.
        if len(body_tgt) >= len(body_src):
            larger, smaller, smaller_body = target, source, body_src
        else:
            larger, smaller, smaller_body = source, target, body_tgt
            
        items.append(WorkItem(
            kind="dedup",
            target_path=larger,
            context={
                "concept": smaller.rsplit("/", 1)[-1],
                "excerpt": smaller_body[:4000],
                "candidate": larger.rsplit("/", 1)[-1],
                "score": score,
                "inbox_file": smaller,
            },
            reason=f"ledger_dedup score={score:.3f}",
        ))
        
    if not items:
        return {"success": False, "message": "No valid pairs to process"}
        
    res = run_subagent_batch(items)
    res["pairs_found"] = len(items)
    return res


class DedupFolderArgs(BaseModel):
    folder: str = Field(default="", description="Vault folder to scan for near-duplicate notes (empty = whole vault)")


@tool(DedupFolderArgs, cls="composed")
def silica_dedup(folder: str = "", cancel_token: Any = None) -> dict[str, Any]:
    """SCAN a folder (or the vault) for near-duplicate note pairs and merge each
    smaller note into its larger twin.

    Only the smaller note's genuinely-new info is appended to the larger note
    (a single append-only patch) — never rewrites, deletes, or creates notes.
    Requires the embedding index (silica_embed_refresh). If you already know
    the pairs, use silica_dedup_pairs instead.

    A pair is admitted when its body similarity is borderline OR its titles are
    strongly similar — the latter catches cases like "ROS" / "JSON in ROS 2"
    where bodies diverge but titles are clearly related.
    """
    from silica.kernel.embed import get_store, _cosine
    from silica.kernel.workqueue import WorkItem
    from silica.agent.subagent import run_subagent_batch
    from silica.config import CONFIG as _C

    store = get_store()
    if len(store) == 0:
        return {"error": "Embedding index empty — run /embed first."}

    τ_high = getattr(_C, "sim_threshold_high", 0.85)
    τ_low = getattr(_C, "sim_threshold_low", 0.65)
    τ_title = getattr(_C, "sim_title_threshold", 0.80)

    scope = [p for p in store.paths() if _in_folder(p, folder)]
    seen_pairs: set[tuple[str, str]] = set()
    items: list[WorkItem] = []

    for p in scope:
        vec = store.get_vec(p)
        if not vec:
            continue
        candidates = store.cosine_top_k(vec, k=_C.dedup_scan_k, exclude={p})
        for match in candidates:
            score = match.get("score", 0.0)
            other = match.get("path", "")
            if not other or not _in_folder(other, folder):
                continue

            # Title-level similarity gate: catches pairs whose bodies diverge
            # but whose titles share a strong semantic relationship.
            title_vec_p = store.get_title_vec(p)
            title_vec_o = store.get_title_vec(other)
            title_score = (
                _cosine(title_vec_p, title_vec_o)
                if title_vec_p and title_vec_o
                else 0.0
            )

            in_full_window = τ_low < score < τ_high
            in_title_gate  = title_score >= τ_title

            # continue (not break): list is sorted descending; a match above τ_high
            # arrives before borderline ones — break would kill the loop too early.
            if not in_full_window and not in_title_gate:
                continue

            key = tuple(sorted((p, other)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            try:
                body_p = DRIVER.read_note(p).content or ""
                body_o = DRIVER.read_note(other).content or ""
            except Exception:
                continue

            # The larger note is the merge target; the smaller is the source of new info.
            if len(body_o) >= len(body_p):
                larger, smaller, smaller_body = other, p, body_p
            else:
                larger, smaller, smaller_body = p, other, body_o

            effective_score = max(score, title_score)
            items.append(WorkItem(
                kind="dedup",
                target_path=larger,
                context={
                    "concept": smaller.removesuffix(".md").rsplit("/", 1)[-1],
                    "excerpt": smaller_body[:4000],
                    "candidate": larger.removesuffix(".md").rsplit("/", 1)[-1],
                    "score": effective_score,
                    "full_score": score,
                    "title_score": title_score,
                    "inbox_file": smaller,
                },
                reason=f"folder_dedup score={effective_score:.3f} (full={score:.3f} title={title_score:.3f})",
            ))

    res = run_subagent_batch(items, cancel_token=cancel_token)
    res["pairs_found"] = len(items)
    res["folder"] = folder or "(vault)"
    return res


class RefineBatchArgs(BaseModel):
    note_paths: list[str] = Field(description="List of vault-relative paths to stylistically refine.")

@tool(RefineBatchArgs, cls="composed")
def silica_refine_batch(note_paths: list[str], cancel_token: Any = None) -> dict[str, Any]:
    """Stylistically refine a batch of notes: reformat for clarity and Obsidian
    style WITHOUT adding or losing information.

    To add missing content to thin notes, use silica_enrich_batch instead.
    """
    if not note_paths:
        return {"error": "No note paths provided."}

    from silica.kernel.workqueue import WorkItem
    from silica.agent.subagent import run_subagent_batch

    items = [WorkItem(kind="refine", target_path=p, context={}) for p in note_paths]
    res = run_subagent_batch(items, cancel_token=cancel_token)
    res["notes"] = len(items)
    return res


class EnrichBatchArgs(BaseModel):
    note_paths: list[str] = Field(description="List of vault-relative paths to semantically enrich.")

@tool(EnrichBatchArgs, cls="composed")
def silica_enrich_batch(note_paths: list[str], cancel_token: Any = None) -> dict[str, Any]:
    """Semantically enrich a batch of lean or empty notes: adds substantive
    content. To only fix style/formatting without changing content, use
    silica_refine_batch instead."""
    if not note_paths:
        return {"error": "No note paths provided."}

    from silica.kernel.workqueue import WorkItem
    from silica.agent.subagent import run_subagent_batch

    items = [WorkItem(kind="enrich", target_path=p, context={}) for p in note_paths]
    res = run_subagent_batch(items, cancel_token=cancel_token)
    res["notes"] = len(items)
    return res


class GenerateTaxonomyArgs(BaseModel):
    user_intent: str = Field(description="Natural-language description of how the user wants to organize their vault")
    scope: str = Field(
        default="",
        description="Vault-relative subfolder to restrict taxonomy generation and scanning to",
    )
    save_path: str = Field(
        default="",
        description=(
            "Vault-relative path where the taxonomy YAML should be written. "
            "Defaults to 'taxonomy.yaml' inside the configured vault."
        ),
    )
    merge: bool = Field(
        default=False,
        description=(
            "If True, feed the existing taxonomy to the LLM as standing rules and "
            "update it incrementally instead of regenerating it from scratch."
        ),
    )

@tool(GenerateTaxonomyArgs, cls="composed")
def silica_generate_taxonomy(
    user_intent: str, scope: str = "", save_path: str = "", merge: bool = False
) -> dict[str, Any]:
    """Generate a taxonomy YAML from a natural-language organization intent.

    Uses the LLM to translate the user's description into a structured
    FolderRule list, validates it with Pydantic, and writes it to disk
    at taxonomy.yaml (or the specified path).

    With merge=True the existing taxonomy (if any) is treated as standing
    directives: the LLM preserves its rules and only adds/updates what the
    new intent requires.

    Returns the validated taxonomy dict and the path it was written to.
    The user should review the output before running silica_run_organizer.
    """
    from pathlib import Path

    from silica.agent.llm import call_llm
    from silica.config import CONFIG
    from silica.kernel.sanitize import parse_json
    from silica.kernel.taxonomy import (
        TAXONOMY_GENERATION_PROMPT,
        TAXONOMY_MERGE_BLOCK,
        Taxonomy,
        default_taxonomy_path,
    )
    from silica.driver import DRIVER

    # Resolve the destination first — merge mode reads the current file from there.
    if save_path:
        dest = Path(save_path)
        if not dest.is_absolute() and CONFIG.vault_path:
            dest = Path(CONFIG.vault_path) / dest
    else:
        dest = default_taxonomy_path()

    note_titles: list[str] = []
    try:
        refs = DRIVER.list_files(scope or "")
        note_titles = [
            Path(ref.path or ref.name).stem
            for ref in refs
            if (ref.path or ref.name).endswith(".md")
        ]
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("silica_generate_taxonomy: failed to list files for scope %s: %s", scope, exc)

    # Format the titles clearly (e.g., max 400 titles to prevent prompt bloat)
    titles_summary = "\n".join(f"- {t}" for t in note_titles[:400])
    if len(note_titles) > 400:
        titles_summary += f"\n- ... and {len(note_titles) - 400} more notes."

    system_prompt = "You are an expert knowledge manager. Follow the user instructions exactly."
    user_msg = TAXONOMY_GENERATION_PROMPT.format(
        user_intent=user_intent,
        scope=scope or "Entire Vault",
        note_titles=titles_summary or "(No notes found in scope)",
    )

    if merge and dest.exists():
        try:
            existing = Taxonomy.from_yaml(dest)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "silica_generate_taxonomy: cannot parse existing taxonomy at %s (%s) — generating from scratch",
                dest, exc,
            )
            existing = None
        if existing is not None and existing.rules:
            import yaml as _y
            existing_yaml = _y.dump(
                existing.model_dump(), allow_unicode=True, sort_keys=False, default_flow_style=False
            )
            user_msg += TAXONOMY_MERGE_BLOCK.format(existing_yaml=existing_yaml)

    try:
        response = call_llm(
            model=CONFIG.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            tools=None,
        )
        raw = (response.text or "").strip()
    except Exception as exc:
        return {"error": f"LLM call failed: {exc}"}

    # The LLM is instructed to return raw YAML; try yaml.safe_load first,
    # then fall back to parse_json for robustness.
    import yaml as _yaml

    taxonomy_dict: dict | None = None
    try:
        taxonomy_dict = _yaml.safe_load(raw)
    except Exception:
        pass

    if not isinstance(taxonomy_dict, dict):
        parsed, _ = parse_json(raw, strict=False)
        if isinstance(parsed, dict):
            taxonomy_dict = parsed
        else:
            return {"error": f"LLM returned unparseable output: {raw[:300]}"}

    try:
        taxonomy = Taxonomy.from_dict(taxonomy_dict)
    except Exception as exc:
        return {"error": f"Taxonomy validation failed: {exc}", "raw": taxonomy_dict}

    try:
        taxonomy.to_yaml(dest)
    except Exception as exc:
        return {"error": f"Failed to write taxonomy to {dest}: {exc}"}

    return {
        "success": True,
        "taxonomy_path": str(dest),
        "taxonomy": taxonomy.model_dump(),
        "rules_count": len(taxonomy.rules),
    }


class RunOrganizerArgs(BaseModel):
    taxonomy_path: str = Field(
        default="",
        description="Path to the taxonomy YAML file. Defaults to 'taxonomy.yaml' in the vault.",
    )
    scope: str = Field(
        default="",
        description="Vault-relative subfolder to restrict organization to (empty = vault-wide)",
    )
    dry_run: bool = Field(
        default=True,
        description=(
            "If True (default), compute and return the move plan without executing any moves. "
            "Set to False to actually move notes."
        ),
    )
    llm_arbiter: bool = Field(
        default=True,
        description="If True, use the LLM to classify borderline notes (ambiguous band)",
    )
    move_uncategorized: bool = Field(
        default=False,
        description=(
            "If True, notes matching no taxonomy rule are moved to the uncategorized "
            "folder. Default False: unmatched notes stay where they are."
        ),
    )

@tool(RunOrganizerArgs, cls="composed")
def silica_run_organizer(
    taxonomy_path: str = "",
    scope: str = "",
    dry_run: bool = True,
    llm_arbiter: bool = True,
    move_uncategorized: bool = False,
) -> dict[str, Any]:
    """Classify vault notes against the taxonomy and move them into its folders.

    Requires a taxonomy (silica_generate_taxonomy first). Two-phase:
    dry_run=True (default) returns the move plan without touching anything;
    dry_run=False executes the moves graph-safely (wikilinks updated), with
    automatic rollback if the post-move lint gate fails. To move a single note
    directly, use silica_move instead.
    """
    from silica.kernel.taxonomy import load_taxonomy
    from silica.router.organize_fsm import OrganizerFSM

    taxonomy = load_taxonomy(taxonomy_path or None)
    if not taxonomy.rules:
        return {
            "error": (
                "Taxonomy has no rules. Run silica_generate_taxonomy first or "
                "create taxonomy.yaml manually."
            )
        }

    fsm = OrganizerFSM(
        taxonomy=taxonomy,
        scope=scope,
        dry_run=dry_run,
        llm_arbiter=llm_arbiter,
        move_uncategorized=move_uncategorized,
    )
    return fsm.run()
