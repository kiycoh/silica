"""Injector write states: SNAPSHOT, WRITE, HUB_UPDATE (+ MOC helpers).

Handler bodies for InjectorFSM, extracted from orchestrator.py: each function
takes the FSM instance and mutates its context/state exactly as the former
method did. Patchable collaborators (DRIVER, CONFIG, tools, load_ops, time)
are resolved through the orchestrator module namespace (orch.X) so tests that
patch silica.router.orchestrator.* keep working.
"""
from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from silica.router import orchestrator as orch

if TYPE_CHECKING:
    from silica.router.orchestrator import InjectorFSM

logger = logging.getLogger(__name__)


from silica.kernel.ops import OpType


# Italian function-word markers used for language detection in MOC headings.
_ITALIAN_MARKERS_RE = re.compile(
    r'\b(della|dello|degli|delle|del|dal|nel|sul|per|con|una|questo|questa|sono|hanno|viene|vengono)\b',
    re.IGNORECASE,
)


def _moc_prefix(sample: str) -> str:
    """Return 'Da' (Italian) or 'From' (English) based on marker density in sample."""
    return "Da" if len(_ITALIAN_MARKERS_RE.findall(sample)) >= 3 else "From"


def _moc_heading(source_name: str, sample: str) -> str:
    """Language-aware MOC section heading: '## Da: {name}' or '## From: {name}'."""
    return f"## {_moc_prefix(sample)}: {source_name}"


def _merge_moc_section(content: str, heading: str, note_lines: list[str]) -> str:
    """Append note_lines to an existing MOC section or create a new one.

    When the same source file produces multiple chunks, each chunk calls
    HUB_UPDATE.  Rather than duplicating the heading, new links are appended
    inside the existing section.
    """
    if heading + "\n" in content or heading + "\r\n" in content:
        # Append new links just before the next same-level heading or end of file.
        pattern = re.compile(re.escape(heading) + r'(.*?)(?=\n##\s|\Z)', re.DOTALL)
        def _append(m: re.Match) -> str:
            return m.group(0).rstrip() + "\n" + "\n".join(note_lines) + "\n"
        return pattern.sub(_append, content, count=1)
    moc_block = f"\n{heading}\n\n" + "\n".join(note_lines) + "\n"
    return content.rstrip() + "\n" + moc_block


def _resolve_vault_path(name: str) -> str | None:
    """Real vault-relative path of an existing note by name, searched vault-wide.

    Mirrors validate._resolve_parent's `search_names` check so HUB_UPDATE writes
    the parent where it actually lives instead of assuming `target_dir/<name>.md`.
    Returns None when the note doesn't exist yet (caller falls back to target_dir).
    """
    name_l = name.lower()
    try:
        matches = [r for r in orch.DRIVER.search_names(name) if r.name.lower() == name_l and r.path]
    except Exception:
        return None
    if not matches:
        return None
    # Deterministic pick on duplicate names: shallowest path, then lexical.
    return sorted(matches, key=lambda r: (r.path.count("/"), r.path.lower()))[0].path


def handle_snapshot(fsm: "InjectorFSM") -> None:
    idx = fsm._current_chunk_idx
    fsm._progress_note(fsm._chunk_task_id("snapshot"), "snapshot", "running")
    from silica.tools.wrapped import silica_snapshot
    res = silica_snapshot(fsm._chunk_ctx["ops_path"])
    if "error" in res:
        raise RuntimeError(f"SNAPSHOT failed: {res['error']}")

    fsm._chunk_ctx["snapshot"] = res
    fsm._chunk_ctx["txn_id"] = res["txn_id"]
    try:
        from silica.driver.base import NoteRef, Txn
        from silica.kernel.ops import InverseOp
        inv = [InverseOp(**d) for d in res["inverses"]]
        
        # Reconstruct refs for Txn from inverses
        refs = []
        for d in res["inverses"]:
            if d.get("kind") == "restore_version":
                path = d.get("path")
                name = path.rsplit("/", 1)[-1].removesuffix(".md")
                refs.append(NoteRef(name=name, path=path))
                
        fsm._txn = Txn(
            id=res["txn_id"],
            refs=refs,
            versions=res.get("versions", {}),
            created_paths=res.get("created_paths", []),
            inverses=inv
        )
    except Exception as e:
        raise RuntimeError(f"SNAPSHOT rebuild failed: {e}")

    # S3.2: Take pre-write graph snapshot incrementally
    try:
        from silica.kernel.ast import extract_links as _extract_links
        ops = orch.load_ops(fsm._chunk_ctx["ops_path"])
        touched_refs = []
        snapshot_domain = set()

        for op in ops:
            path = op.touched_ref()
            if path:
                name = os.path.splitext(os.path.basename(path))[0]
                ref = NoteRef(name=name, path=path)
                touched_refs.append(ref)
                snapshot_domain.add(ref)

                if op.op in (OpType.patch, OpType.overwrite, OpType.delete):
                    # Capture current outgoing targets so we can detect orphaning.
                    try:
                        for target_ref in orch.DRIVER.links(ref):
                            snapshot_domain.add(target_ref)
                    except Exception as ex:
                        logger.warning("Failed to fetch pre-write links for %s: %s", path, ex)

                elif op.op == OpType.write:
                    # A write op creates a new note that didn't exist at pre-snapshot
                    # orch.time.  After the write, graph_snapshot expands its neighborhood
                    # to include every vault note the new note links to.  If those
                    # linked notes carry pre-existing unresolved links, they appear as
                    # new_unres in graph_diff Rule 2 — a false positive.
                    # Fix: add those link targets to the pre-snapshot domain now so
                    # their existing ghost links cancel out in the diff.
                    content = op.snippet or op.content or ""
                    for link_target in _extract_links(content):
                        target_stem = link_target.removesuffix(".md")
                        target_key = target_stem.lower()
                        try:
                            if "/" in target_stem:
                                target_name = os.path.splitext(os.path.basename(target_stem))[0]
                                snapshot_domain.add(NoteRef(name=target_name, path=target_stem + ".md"))
                            else:
                                for match in orch.DRIVER.search_names(target_stem):
                                    if match.name.lower() == target_key:
                                        snapshot_domain.add(match)
                        except Exception as ex:
                            logger.debug("Snapshot domain expansion: could not resolve '%s': %s", link_target, ex)

        snapshot_domain_list = list(snapshot_domain)
        fsm._chunk_ctx["snapshot_domain"] = [{"name": r.name, "path": r.path} for r in snapshot_domain_list]
        fsm._pre_graph = orch.DRIVER.graph_snapshot(snapshot_domain_list)
    except Exception as e:
        logger.error("Failed to take pre-write graph snapshot: %s", e)
        raise RuntimeError(f"Pre-write graph snapshot failed: {e}")

    fsm._progress_note(fsm._chunk_task_id("snapshot"), "snapshot", "done")
    fsm._transition_success()


def _attach_section_images(fsm: "InjectorFSM", ops: list) -> None:
    """Re-attach source images to the notes distilled from their section.

    The distiller never saw these embeds (media.strip_images strips them from
    its payload), so the produced notes would otherwise lose every figure. Here
    we read the chunk's raw source file and, for each new/patched note, append
    the images from the section whose heading matches the note's concept — the
    same heading→section join payload.py uses to build the excerpt.
    """
    # One source file per chunk (same derivation as HUB_UPDATE below).
    _fi, _ci = fsm._chunk_flat_to_fi_ci.get(fsm._current_chunk_idx, (0, 0))
    source_file = (
        fsm._file_chunks[_fi]["source_file"]
        if fsm._file_chunks and _fi < len(fsm._file_chunks)
        else fsm.inbox_file
    )
    if not source_file:
        return
    try:
        source = orch.DRIVER.read_note(source_file).content or ""
    except Exception:
        return
    if "![" not in source:  # fast bail: no embeds at all
        return
    from silica.kernel.media import append_section_images
    for op in ops:
        if op.op not in (OpType.write, OpType.patch):
            continue
        path = op.touched_ref()
        if not path:
            continue
        # ponytail: overwrite (collision-enrich) ops left alone — re-attaching to
        # a rewritten vault note risks dupes; handle if it ever bites.
        concept = os.path.splitext(os.path.basename(path))[0]
        op.snippet = append_section_images(op.snippet or "", source, concept)


def handle_write(fsm: "InjectorFSM") -> None:
    from silica.kernel.atomic_write import bulk_write_atomic
    fsm._progress_note(fsm._chunk_task_id("write"), "write", "running")

    ops = orch.load_ops(fsm._chunk_ctx["ops_path"])
    try:
        _attach_section_images(fsm, ops)
    except Exception as _ie:
        logger.debug("WRITE: image re-attach skipped (%s)", _ie)
    result = bulk_write_atomic(ops, hub=fsm.hub, lint=True)

    # Accumulate surviving notes' inverses for the undo journal (recorded at
    # CLEANUP after autolink finalises content → correct version-guard hashes).
    for r in result.committed:
        for inv in r.inverses:
            fsm._run_inverses.append((r.path, inv, None))

    # Brief run-yield for the TUI summary: count NEW notes actually committed
    # (created_paths ∩ committed), accumulated across chunks/files.
    created = {p.removesuffix(".md") for p in (fsm._txn.created_paths or [])} if fsm._txn else set()
    new_notes = sum(1 for r in result.committed if r.path.removesuffix(".md") in created)
    fsm.context["yield_notes"] = fsm.context.get("yield_notes", 0) + new_notes

    if result.failed:
        try:
            _deferred = [
                next(o for o in ops if o.touched_ref() == r.path).model_dump()
                for r in result.failed
            ]
            _errors = {r.path: (r.error or "lint/write failed") for r in result.failed}
            fsm._defer_ops(_deferred, _errors, phase="WRITE")
            logger.warning(
                "WRITE: %d op(s) failed lint/write — deferred. "
                "Continuing with %d committed op(s).",
                len(result.failed), len(result.committed),
            )
            # Mark failed/deferred ops as skip so subsequent phases (lint, autolink, backlink, cleanup, etc.) skip them
            failed_paths = {r.path for r in result.failed}
            for op in ops:
                if op.touched_ref() in failed_paths:
                    op.op = OpType.skip
                    op.reason = "Deferred because write/lint failed"
            from silica.kernel.ops_io import dump_ops
            dump_ops(fsm._chunk_ctx["ops_path"], ops)
        except Exception as _de:
            logger.debug("WRITE: deferred save/update failed (non-fatal): %s", _de)
        fsm.context["has_partial_failure"] = True

    if not result.committed and result.failed:
        fsm._progress_note(fsm._chunk_task_id("write"), "write", "failed",
                            error=f"all {len(result.failed)} ops failed lint/write")

    # Synthesise the legacy `write` result shape that downstream code reads.
    fsm.context["write"] = {
        "success": True,
        "successful": len(result.committed),
        "failed": [{"path": r.path, "error": r.error} for r in result.failed],
        "total": result.total,
    }

    committed_paths = {r.path for r in result.committed}
    _deferred_stems: frozenset[str] = frozenset(
        os.path.splitext(os.path.basename(r.path))[0].lower()
        for r in result.failed if r.path
    )
    fsm._chunk_ctx["deferred_stems"] = list(_deferred_stems)

    # Register committed notes in the RunManifest and refresh embedding index.
    try:
        from silica.kernel.progress import RunManifestEntry
        vault_ctx = fsm.context.get("vault_graph_ctx", {})
        for op in ops:
            path = op.touched_ref()
            if op.op not in (OpType.write, OpType.patch) or not path:
                continue
            if path not in committed_paths:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            path_key = path.removesuffix(".md")
            cluster_id = vault_ctx.get(path_key, {}).get("cluster_id", -1)
            fsm.manifest.record(RunManifestEntry(
                title=stem,
                path=path_key,
                parent=op.parent,
                cluster_id=cluster_id,
                source_basename=op.source_basename or "",
                op=op.op.value,
            ))
        fsm.manifest.save()

        # Best-effort incremental embed index refresh
        try:
            from silica.agent.providers import get_embedder
            from silica.kernel.embed import get_store, refresh_note
            embedder = get_embedder(orch.CONFIG)
            store = get_store()
            for op in ops:
                path = op.touched_ref()
                if op.op not in (OpType.write, OpType.patch) or not path:
                    continue
                if path not in committed_paths:
                    continue
                stem = os.path.splitext(os.path.basename(path))[0]
                idx_path = path.removesuffix(".md")
                try:
                    body = orch.DRIVER.read_note(path).content or ""
                    # Fix A: defer the whole-index write to a single end-of-run
                    # flush (_run_loop); upsert into the shared in-memory store only.
                    refresh_note(embedder, idx_path, stem, body, store=store, save=False)
                    # Mark the in-memory index dirty so the end-of-run flush knows
                    # to persist (and only then — no flush when nothing changed).
                    fsm.context["_embed_dirty"] = True
                except Exception as _re:
                    logger.debug("WRITE: embed refresh failed for '%s': %s", path, _re)
        except Exception as _ee:
            logger.debug("WRITE: embed refresh skipped (%s)", _ee)

        # Best-effort incremental co-occurrence refresh — the STABLE leg.
        # Separate from the embed block above so it stays fresh even when the
        # embedder is down (no network, pure local compute). Also deferred (Fix A).
        _n_cooccur = orch._refresh_cooccurrence_for_ops(
            ops,
            committed_paths,
            read_body=lambda p: orch.DRIVER.read_note(p).content or "",
            lang=orch.CONFIG.cooccurrence_lang,
            save=False,
        )
        if _n_cooccur:
            fsm.context["_cooccur_dirty"] = True
    except Exception as _me:
        logger.debug("WRITE: manifest update failed (non-fatal): %s", _me)

    # Git safety net (SILICA_GIT_COMMIT=auto): snapshot the write batch.
    try:
        orch._commit_docs_for_ops(
            ops, committed_paths,
            vault=orch.CONFIG.vault_path, git_commit=orch.CONFIG.git_commit,
        )
    except Exception as _ge:
        logger.debug("WRITE: git auto-commit skipped (%s)", _ge)

    if not result.committed and result.failed:
        pass
    else:
        fsm._progress_note(fsm._chunk_task_id("write"), "write", "done")
    fsm._transition_success()


def handle_hub_update(fsm: "InjectorFSM") -> None:
    """Append MOC links to the Hub note for all newly written notes."""
    idx = fsm._current_chunk_idx
    fsm._progress_note(fsm._chunk_task_id("hub_update"), "hub_update", "running")
    if not fsm.hub:
        logger.info("HUB_UPDATE: no hub configured, skipping")
        fsm._progress_note(fsm._chunk_task_id("hub_update"), "hub_update", "done")
        fsm._transition_success()
        return

    try:
        ops = orch.load_ops(fsm._chunk_ctx["ops_path"])
    except Exception as e:
        raise RuntimeError(f"HUB_UPDATE: failed to read ops: {e}")

    hub_name = fsm.hub.strip("[]")
    hub_name_lower = hub_name.lower()

    # Collect write ops grouped by effective parent:
    # notes with op.parent set go to that parent note; others fall back to hub.
    hub_notes: list[tuple[str, str]] = []       # (note_name, desc)
    parent_notes: dict[str, list[tuple[str, str]]] = {}  # parent_name → [(note_name, desc)]
    for op in ops:
        if op.op != OpType.write:
            continue
        path = op.touched_ref()
        if not path:
            continue
        note_name = os.path.splitext(os.path.basename(path))[0]
        if note_name.lower() == hub_name_lower:
            continue
        snippet = (op.snippet or "").strip()
        desc = snippet.split("\n")[0] if snippet else ""
        effective_parent = (op.parent.strip("[]") if op.parent else None) or hub_name
        if effective_parent.lower() == hub_name_lower:
            hub_notes.append((note_name, desc))
        else:
            parent_notes.setdefault(effective_parent, []).append((note_name, desc))

    # Flatten for backward-compat references below
    new_notes = hub_notes

    if not new_notes and not parent_notes:
        logger.info("HUB_UPDATE: no new notes to link, skipping")
        fsm._progress_note(fsm._chunk_task_id("hub_update"), "hub_update", "done")
        fsm._transition_success()
        return

    hub_path = f"{fsm.target_dir}/{hub_name}.md".replace("//", "/")
    from silica.driver.base import NoteRef
    hub_ref = NoteRef(name=hub_name, path=hub_path)

    try:
        hub_note = orch.DRIVER.read_note(hub_ref)
    except Exception as e:
        logger.warning("HUB_UPDATE: hub '%s' not readable: %s — skipping", hub_path, e)
        fsm._progress_note(fsm._chunk_task_id("hub_update"), "hub_update", "done")
        fsm._transition_success()
        return

    # If hub pre-existed (not created in this txn), register a content-based
    # rollback inverse using the content we just read — more reliable than
    # history:restore whose version positions shift after each new write.
    hub_path_norm = hub_path.replace("\\", "/")
    hub_is_new = fsm._txn is not None and any(
        p.replace("\\", "/") == hub_path_norm
        for p in (fsm._txn.created_paths or [])
    )
    if not hub_is_new and fsm._txn is not None:
        from silica.kernel.ops import InverseOp, InverseOpKind
        hub_inverse = InverseOp(
            kind=InverseOpKind.restore_version,
            path=hub_path,
            prior_content=hub_note.content,
        )
        fsm._txn.inverses.append(hub_inverse)

    # Cross-cluster integrity check: warn when new notes land in a different
    # cluster from the hub.  This is informational only — the MOC link is
    # still written, but the log helps identify structural drift.
    _gctx = fsm.context.get("vault_graph_ctx", {})
    _hub_key = hub_path.removesuffix(".md")
    _hub_cluster = _gctx.get(_hub_key, {}).get("cluster_id", -1)
    if _gctx and _hub_cluster >= 0:
        for note_name, _ in new_notes:
            _note_key = f"{fsm.target_dir}/{note_name}".replace("//", "/")
            _note_cluster = _gctx.get(_note_key, {}).get("cluster_id", -1)
            if _note_cluster >= 0 and _note_cluster != _hub_cluster:
                logger.warning(
                    "HUB_UPDATE: '%s' (cluster %d) linked to hub '%s' (cluster %d) — cross-cluster MOC",
                    note_name, _note_cluster, hub_name, _hub_cluster,
                )

    # Derive the actual source file for this chunk (fsm.inbox_file is always
    # the first file and never updates in multi-file runs — use the flat index map).
    _fi, _ci = fsm._chunk_flat_to_fi_ci.get(fsm._current_chunk_idx, (0, 0))
    _source_file = (
        fsm._file_chunks[_fi]["source_file"]
        if fsm._file_chunks and _fi < len(fsm._file_chunks)
        else fsm.inbox_file
    )
    source_name = os.path.splitext(os.path.basename(_source_file))[0]

    # Language-aware heading: "## Da: {name}" (Italian) or "## From: {name}" (English).
    # Sample the hub content + first snippet to detect language.
    _lang_sample = hub_note.content + " ".join(d for _, d in new_notes[:3])
    moc_heading = _moc_heading(source_name, _lang_sample)

    # Build note link lines.
    note_lines = [
        f"- [[{n}]] — {d}" if d else f"- [[{n}]]"
        for n, d in new_notes
    ]

    # Merge: append to existing section if present (same file, multiple chunks),
    # otherwise create a new section.  Use overwrite to avoid the settle race.
    new_hub_content = _merge_moc_section(hub_note.content, moc_heading, note_lines)

    try:
        orch.DRIVER.overwrite(hub_path, new_hub_content)
        # Explicitly wait until the section header is readable.
        _deadline = orch.time.monotonic() + 5.0
        while orch.time.monotonic() < _deadline:
            try:
                if moc_heading in orch.DRIVER.read_note(hub_ref).content:
                    break
            except Exception:
                pass
            orch.time.sleep(0.15)
        else:
            logger.warning("HUB_UPDATE: MOC block settle timeout for hub '%s' — graph may lag", hub_path)
        logger.info("HUB_UPDATE: updated hub '%s' with %d links", hub_path, len(new_notes))
    except Exception as e:
        raise RuntimeError(f"HUB_UPDATE: failed to update hub '{hub_path}': {e}")

    # Extend snapshot_domain so LINT's graph regression check covers the hub's new links
    existing_paths = {d["path"] for d in fsm._chunk_ctx.get("snapshot_domain", [])}
    if hub_path not in existing_paths:
        fsm._chunk_ctx.setdefault("snapshot_domain", []).append({"name": hub_name, "path": hub_path})

    # Write MOC sections to specific parent notes (best-effort — only active when
    # the distiller emits op.parent, which requires the Block 4 prompt update).
    if parent_notes:
        from silica.kernel.ops import InverseOp, InverseOpKind
        for parent_name, p_new_notes in parent_notes.items():
            # Resolve the parent wherever it actually lives in the vault (mirrors
            # validate's search_names check) instead of assuming it sits under
            # target_dir — otherwise an existing parent in another folder reads
            # as "File not found". Fall back to target_dir only for a new note.
            parent_path = _resolve_vault_path(parent_name) or f"{fsm.target_dir}/{parent_name}.md".replace("//", "/")
            try:
                from silica.driver.base import NoteRef as _NR
                p_ref = _NR(name=parent_name, path=parent_path)
                p_note = orch.DRIVER.read_note(p_ref)
                # Register rollback inverse for parent note
                if fsm._txn is not None:
                    p_inverse = InverseOp(
                        kind=InverseOpKind.restore_version,
                        path=parent_path,
                        prior_content=p_note.content,
                    )
                    fsm._txn.inverses.append(p_inverse)
                # Build and write parent MOC block (same language-aware heading,
                # same deduplication logic as the hub section above).
                p_heading = _moc_heading(source_name, p_note.content)
                p_note_lines = [
                    f"- [[{n}]] — {d}" if d else f"- [[{n}]]"
                    for n, d in p_new_notes
                ]
                new_p_content = _merge_moc_section(p_note.content, p_heading, p_note_lines)
                orch.DRIVER.overwrite(parent_path, new_p_content)
                existing_paths = {d["path"] for d in fsm._chunk_ctx.get("snapshot_domain", [])}
                if parent_path not in existing_paths:
                    fsm._chunk_ctx.setdefault("snapshot_domain", []).append(
                        {"name": parent_name, "path": parent_path}
                    )
                logger.info(
                    "HUB_UPDATE: updated parent '%s' with %d sub-spoke link(s)",
                    parent_path, len(p_new_notes),
                )
            except Exception as _pe:
                logger.warning("HUB_UPDATE: failed to update parent '%s': %s", parent_path, _pe)

    fsm._progress_note(fsm._chunk_task_id("hub_update"), "hub_update", "done")
    fsm._transition_success()
