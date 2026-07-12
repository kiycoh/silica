# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

import os
import logging
from pydantic import BaseModel
from silica.driver import DRIVER
from silica.kernel.ops import Op, OpType
from silica.kernel.templates import slugify
from silica.kernel.ast import extract_links

logger = logging.getLogger(__name__)

# Precision gate: a write op whose snippet is shorter than this is deferred
# instead of written — execute_write would otherwise fill the note with a
# "(da espandere)" placeholder (real incident: run 5d0a3350, 2026-07-04, the
# distiller returned whole chunks with snippet="" despite full inbox excerpts).
# Rejection routes through the existing defer + steer path, so the distiller
# gets re-prompted with the reason.
MIN_WRITE_SNIPPET_CHARS = 100


class Rejection(BaseModel):
    op: Op
    reason: str


def validate_operations(
    ops: list[Op] | list[dict],
    payloads: list,
    target_dir: str,
    hub: str | None = None,
    cleared_parents_out: list | None = None,
    future_ref_whitelist: list[str] | None = None,
    cleared_links_out: list | None = None,
    ungrounded_out: list | None = None,
) -> tuple[list[Op], list[Rejection]]:
    """Validates operations against payloads and target_dir using DRIVER.

    ungrounded_out (optional): collects warn-only span-grounding hits —
    write/patch ops whose math/code spans can't be located in their source
    excerpt (fabrication candidates). Never causes a rejection.
    """
    from silica.kernel.ops_io import parse_ops
    ops_parsed = parse_ops(ops)
    ops = [op.model_copy(deep=True) for op in ops_parsed]

    # Sanitize filesystem-illegal characters (e.g. ':') from path filenames.
    # When a write op carries a `title` field, rebuild the path from title so
    # the note is filed under the clean concept name rather than the heading.
    for op in ops:
        if op.op == OpType.write and op.title and op.path and target_dir:
            clean_title = slugify(op.title)
            if clean_title:
                new_path = f"{target_dir.rstrip('/')}/{clean_title}.md"
                if new_path != op.path:
                    logger.debug("validate: title-derived path '%s' → '%s'", op.path, new_path)
                    op.path = new_path

        if op.path:
            folder, filename = os.path.split(op.path)
            name, ext = os.path.splitext(filename)
            sanitized = slugify(name) + ext
            if sanitized != filename:
                new_path = (os.path.join(folder, sanitized) if folder else sanitized).replace("\\", "/")
                logger.debug("validate: sanitized path '%s' → '%s'", op.path, new_path)
                op.path = new_path

    valid_concepts: dict[str, set[str]] = {}
    expected_collision_paths: dict[tuple[str, str], str | None] = {}
    concept_excerpts: dict[tuple[str, str], str] = {}
    collision_excerpts: dict[tuple[str, str], str] = {}
    inbox_folders = set()
    has_payloads = bool(payloads)

    # Index payloads
    if has_payloads:
        for payload_data in payloads:
            batches = payload_data.get("batches", [])
            for batch in batches:
                inbox_file = batch.get("inbox_file")
                if not inbox_file:
                    continue
                    
                source_basename = os.path.basename(inbox_file)
                inbox_dir = os.path.dirname(os.path.abspath(inbox_file))
                inbox_folders.add(inbox_dir)
                
                if source_basename not in valid_concepts:
                    valid_concepts[source_basename] = set()
                    
                for c in batch.get("concepts", []):
                    name = c.get("name")
                    if not name:
                        continue
                    valid_concepts[source_basename].add(name)
                    concept_excerpts[(source_basename, name)] = c.get("inbox_excerpt", "") or ""

                    collision = c.get("vault_collision")
                    if collision and isinstance(collision, dict) and collision.get("path"):
                        expected_collision_paths[(source_basename, name)] = collision["path"]
                        collision_excerpts[(source_basename, name)] = collision.get("excerpt", "") or ""
                    else:
                        expected_collision_paths[(source_basename, name)] = None

    _existence_cache: dict[str, bool] = {}
    def path_exists(p: str) -> bool:
        norm = os.path.abspath(p)
        if norm not in _existence_cache:
            try:
                DRIVER.read_note(p)
                _existence_cache[norm] = True
            except RuntimeError:
                _existence_cache[norm] = False
        return _existence_cache[norm]

    # 1. Global deduplication (executed before coercion to ensure correct richest op type determination)
    path_groups: dict[str, list[Op]] = {}
    for op in ops:
        path = op.touched_ref()
        if path:
            norm_path = os.path.abspath(path)
            if norm_path not in path_groups:
                path_groups[norm_path] = []
            path_groups[norm_path].append(op)

    for norm_path, group in path_groups.items():
        if len(group) > 1:
            richest_op = max(group, key=lambda o: len(o.snippet or o.content or ""))
            has_write = any(o.op in (OpType.write, OpType.overwrite) for o in group)
            for op in group:
                if op is not richest_op:
                    op.op = OpType.skip
                    op.reason = f"Duplicate operation to the same path '{op.path}'"
            if has_write:
                # If there's an overwrite in the group, richest_op becomes overwrite
                if any(o.op == OpType.overwrite for o in group):
                    richest_op.op = OpType.overwrite
                else:
                    richest_op.op = OpType.write

    # C3 title-identity gate: existing note titles in target_dir, keyed by
    # title_key, built lazily once per call. Empty on any driver failure —
    # the gate abstains, never blocks the pipeline.
    _title_gate_cache: dict[str, tuple[str, str]] | None = None  # key -> (title, path)

    def _target_dir_titles() -> dict[str, tuple[str, str]]:
        nonlocal _title_gate_cache
        if _title_gate_cache is not None:
            return _title_gate_cache
        from silica.kernel.title import title_key
        out: dict[str, tuple[str, str]] = {}
        try:
            norm_dir = (target_dir or "").replace("\\", "/").strip("/")
            for ref in DRIVER.search_names(""):
                ref_dir = os.path.dirname((ref.path or "").replace("\\", "/")).strip("/")
                if ref_dir != norm_dir:
                    continue
                key = title_key(ref.name)
                if key:
                    out[key] = (ref.name, ref.path)
        except Exception as e:
            logger.debug("validate: title gate enumeration failed (abstaining): %s", e)
        _title_gate_cache = out
        return out

    # 2. Coerce write <-> patch and enforce default hub fallback
    if not hub and target_dir:
        hub = os.path.basename(target_dir.rstrip("/\\"))

    for op in ops:
        if op.op == OpType.skip:
            continue
        if op.op in (OpType.write, OpType.patch, OpType.overwrite) and hub:
            op.hub = hub

        if op.op == OpType.write and op.path and path_exists(op.path):
            op.op = OpType.patch
        elif op.op == OpType.write and op.path:
            # C3 gate, band 1: a title key-equal to an existing note in the
            # target folder is the SAME note under a cosmetic variant
            # («Machine Learning (9 CFU)») — mechanical coercion to patch,
            # extending the exact-path coercion above.
            from silica.kernel.title import title_key
            stem = os.path.splitext(os.path.basename(op.path))[0]
            match = _target_dir_titles().get(title_key(stem))
            if match is not None:
                logger.info(
                    "validate: title '%s' key-equal to existing '%s' — coercing write→patch",
                    stem, match[0],
                )
                op.op = OpType.patch
                op.path = match[1] if match[1].endswith(".md") else f"{match[1]}.md"
        elif op.op == OpType.patch and op.path and not path_exists(op.path):
            if has_payloads:
                expected_path = expected_collision_paths.get((op.source_basename, op.heading))
                if not expected_path or os.path.abspath(op.path) == os.path.abspath(expected_path):
                    op.op = OpType.write
            else:
                op.op = OpType.write

    # Pre-compute note stems created in this run so parent validation can allow
    # forward references to notes being written in the same batch.
    _run_write_stems: set[str] = {
        os.path.splitext(os.path.basename(op.path))[0].lower()
        for op in ops
        if op.op in (OpType.write, OpType.overwrite) and op.path
    }

    def _resolve_parent(op: Op, cleared_out: list | None = None) -> None:
        """Neutralise an unresolvable parent — fall back to hub, no Rejection.

        If cleared_out is provided, records the cleared reference as a forward-link
        hint so the distiller can anticipate the note in future iterations.
        """
        if not op.parent:
            return
        p_key = op.parent.lower()
        if p_key in _run_write_stems:
            return
        matches = DRIVER.search_names(op.parent)
        if not any(r.name.lower() == p_key for r in matches):
            logger.warning(
                "validate: parent '%s' not found in vault or current run — clearing to hub fallback",
                op.parent,
            )
            if cleared_out is not None:
                cleared_out.append({
                    "cleared_parent": op.parent,
                    "note_heading": op.heading or "",
                    "note_path": op.path or "",
                })
            op.parent = None

    def _check_grounding(op: Op) -> None:
        """Warn-only verbatim gate (never rejects): math/code spans in the body
        that can't be located in the source excerpt are fabrication candidates."""
        # Ground against everything the distiller legitimately saw for this
        # concept: inbox excerpt + colliding vault note excerpt — a patch
        # restating a vault formula for coherence is not fabrication.
        excerpt = concept_excerpts.get((op.source_basename, op.heading), "")
        collision = collision_excerpts.get((op.source_basename, op.heading), "")
        source_text = f"{excerpt}\n{collision}" if collision else excerpt
        body = op.snippet or op.content or ""
        if not source_text.strip() or not body:
            return
        from silica.kernel.provenance import ungrounded_spans
        spans = ungrounded_spans(body, source_text)
        if spans:
            logger.warning(
                "validate: '%s' — %d verbatim span(s) not grounded in source excerpt: %s",
                op.path, len(spans), " | ".join(s[:60] for s in spans),
            )
            if ungrounded_out is not None:
                ungrounded_out.append({
                    "path": op.path,
                    "heading": op.heading,
                    "source_basename": op.source_basename,
                    "spans": spans,
                })

    validated_ops = []
    rejected_ops = []

    target_dir_abs = os.path.abspath(target_dir) if target_dir else ""

    def _is_within_dir(path_abs: str, dir_abs: str) -> bool:
        if not dir_abs:
            return True
        try:
            return os.path.commonpath([path_abs, dir_abs]) == dir_abs
        except ValueError:
            return False

    for op in ops:
        heading = op.heading
        op_type = op.op
        source_basename = op.source_basename
        path = op.path

        if has_payloads:
            if not heading:
                rejected_ops.append(Rejection(op=op, reason="Missing 'heading' field"))
                continue
            if not source_basename:
                rejected_ops.append(Rejection(op=op, reason="Missing 'source_basename' field"))
                continue
            if source_basename not in valid_concepts:
                rejected_ops.append(Rejection(op=op, reason=f"Unknown source_basename '{source_basename}'"))
                continue
            if heading not in valid_concepts[source_basename]:
                rejected_ops.append(Rejection(op=op, reason=f"Heading '{heading}' not present in payload concepts"))
                continue

        if path:
            path_abs = os.path.abspath(path)
            forbidden = any(_is_within_dir(path_abs, folder) for folder in inbox_folders)
            if "/0 Inbox/" in path or "/0 inbox/" in path.lower() or forbidden:
                rejected_ops.append(Rejection(op=op, reason=f"Target path '{path}' contains forbidden inbox segment"))
                continue

        if op_type == OpType.skip:
            continue
            
        elif op_type == OpType.patch:
            if not path:
                rejected_ops.append(Rejection(op=op, reason="Missing 'path' field for patch operation"))
                continue
                
            path_abs = os.path.abspath(path)
            if has_payloads:
                expected_path = expected_collision_paths.get((source_basename, heading))
                if expected_path:
                    if path_abs != os.path.abspath(expected_path):
                        rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' does not match expected collision '{expected_path}'"))
                        continue
                else:
                    if not _is_within_dir(path_abs, target_dir_abs):
                        rejected_ops.append(Rejection(op=op, reason=f"Coerced patch path '{path}' not in target folder"))
                        continue
            elif not _is_within_dir(path_abs, target_dir_abs):
                rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' not in target folder"))
                continue

            if not path_exists(path):
                rejected_ops.append(Rejection(op=op, reason=f"Collision path '{path}' does not exist in vault"))
                continue

            _resolve_parent(op, cleared_parents_out)
            _check_grounding(op)
            validated_ops.append(op)

        elif op_type == OpType.write:
            if not path:
                rejected_ops.append(Rejection(op=op, reason="Missing 'path' field for write operation"))
                continue

            path_abs = os.path.abspath(path)
            if not _is_within_dir(path_abs, target_dir_abs):
                rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' not in target folder"))
                continue

            if path_exists(path):
                rejected_ops.append(Rejection(op=op, reason=f"Target path '{path}' already exists (should be patch/overwrite)"))
                continue

            # C3 gate, band 2: fuzzy-near an existing title (Descriptor vs
            # Description) → defer to the review queue so the dedup judge
            # decides — never a hard block, never a silent fourth duplicate.
            from silica.kernel.title import near_titles
            stem = os.path.splitext(os.path.basename(path))[0]
            near = near_titles(stem, [t for (t, _p) in _target_dir_titles().values()])
            if near:
                cand_title, ratio = near[0]
                cand_path = next(
                    (p for (t, p) in _target_dir_titles().values() if t == cand_title), ""
                )
                rejected_ops.append(Rejection(
                    op=op,
                    reason=(
                        f"near_title candidate='{cand_title}' path='{cand_path}' "
                        f"ratio={ratio:.2f} — deferred for dedup review"
                    ),
                ))
                continue

            body_len = len((op.snippet or "").strip())
            if body_len == 0 and has_payloads:
                # Distinguish two ways a write lands with an empty body:
                #  (a) the source excerpt itself is empty — the concept was only
                #      *mentioned*, never defined. Nothing to distill or expand;
                #      deferring only churns (and a whole chunk of these drives the
                #      rejection rate to 100% and aborts the run). Skip it as a
                #      forward-reference — it stays linked from the notes that
                #      mention it, to be authored when a later source defines it.
                #  (b) the excerpt HAD content but the distiller dropped the body
                #      (run 5d0a3350 regression) — that must still be rejected so
                #      the expand arc retries it. Falls through below.
                excerpt = concept_excerpts.get((source_basename, heading))
                if excerpt is not None and not excerpt.strip():
                    logger.info(
                        "validate: write '%s' — empty source excerpt, skipped as a "
                        "forward-reference (nothing to distill)", op.path,
                    )
                    continue
            if body_len < MIN_WRITE_SNIPPET_CHARS:
                rejected_ops.append(Rejection(
                    op=op,
                    reason=(
                        f"snippet too short ({body_len} < {MIN_WRITE_SNIPPET_CHARS} chars) "
                        f"— would write a placeholder note, deferred for retry"
                    ),
                ))
                continue

            _resolve_parent(op, cleared_parents_out)
            _check_grounding(op)
            validated_ops.append(op)

        elif op_type == OpType.overwrite:
            if not path:
                rejected_ops.append(Rejection(op=op, reason="Missing 'path' field for overwrite operation"))
                continue

            path_abs = os.path.abspath(path)
            if not _is_within_dir(path_abs, target_dir_abs):
                rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' not in target folder"))
                continue

            if not path_exists(path):
                # If target note doesn't exist, overwrite degrades to write gracefully
                op.op = OpType.write
            elif op.base_content is None:
                # Snapshot the current note so the write path can 3-way-detect
                # a concurrent edit (charter UC6); the refiner snapshots at
                # triage time, every other producer relies on this choke point.
                op.base_content = DRIVER.read_note(path).content

            _resolve_parent(op, cleared_parents_out)
            validated_ops.append(op)

        else:
            rejected_ops.append(Rejection(op=op, reason=f"Unknown operation type '{op_type}'"))

    # 3. Auto-create missing Hub notes
    hubs_to_check = set()
    for op in validated_ops:
        op_type = op.op
        if op_type in (OpType.write, OpType.patch, OpType.overwrite):
            hub = op.hub
            if hub:
                clean_hub = hub.strip("[]")
                if clean_hub:
                    hubs_to_check.add(clean_hub)

    hub_ops = []
    for hub in sorted(hubs_to_check):
        if not path_exists(hub):
            hub_filename = f"{hub}.md"
            hub_path = os.path.join(target_dir, hub_filename).replace("\\", "/")
            
            already_creating = any(
                (o.op == OpType.write and o.heading == hub) or
                (o.path and os.path.abspath(o.path) == os.path.abspath(hub_path))
                for o in validated_ops
            )
            
            if not already_creating:
                source_basename = "auto_generated"
                if validated_ops:
                    source_basename = validated_ops[0].source_basename or "auto_generated"
                
                hub_op = Op(
                    op=OpType.write,
                    heading=hub,
                    path=hub_path,
                    snippet="Hub automatically generated by the Injector pipeline.",
                    hub=hub,
                    source_basename=source_basename
                )
                hub_ops.append(hub_op)
                logger.info("Validation: hub '%s' does not exist. Injected creation operation at %s", hub, hub_path)

    if hub_ops:
        validated_ops = hub_ops + validated_ops

    # 4. Prospective link check: surface wikilinks introduced by write/patch/overwrite
    # ops that cannot be resolved in the current vault, within this batch, or via the
    # future_ref_whitelist.  Unlike parents (see _resolve_parent), an unresolved inline
    # link does NOT reject the op — it is kept verbatim as a dangling forward-reference
    # and recorded in cleared_links_out, symmetric with cleared_parents.  This mirrors
    # Obsidian semantics (dangling links are first-class) and prevents a self-referential
    # source from losing whole notes to the rejection-rate gate.
    # Any op that leaves a note at op.path after this batch (write/patch/overwrite) is a
    # valid in-batch link target — not just freshly-written notes.
    batch_created_names: set[str] = {
        os.path.splitext(os.path.basename(op.path))[0].lower()
        for op in validated_ops
        if op.op in (OpType.write, OpType.patch, OpType.overwrite) and op.path
    }

    _link_resolve_cache: dict[str, bool] = {}
    whitelist_lower = {w.lower() for w in (future_ref_whitelist or [])}

    def _link_resolves(target: str) -> bool:
        stem = target.removesuffix(".md")
        key = stem.lower()
        if key in _link_resolve_cache:
            return _link_resolve_cache[key]
        if key in batch_created_names:
            _link_resolve_cache[key] = True
            return True
        if key in whitelist_lower:
            _link_resolve_cache[key] = True
            return True
        if "/" in stem:
            result = path_exists(stem + ".md") or path_exists(stem)
        else:
            matches = DRIVER.search_names(stem)
            result = any(r.name.lower() == key for r in matches)
        _link_resolve_cache[key] = result
        return result

    prospective_valid: list[Op] = []
    for op in validated_ops:
        if op.op not in (OpType.write, OpType.patch, OpType.overwrite):
            prospective_valid.append(op)
            continue
        text = op.snippet or op.content or ""
        if not text:
            prospective_valid.append(op)
            continue
        links = extract_links(text)
        broken = [lnk for lnk in links if not _link_resolves(lnk)]
        if broken:
            logger.debug(
                "validate: %d unresolved wikilink(s) kept as forward-ref in '%s': %r",
                len(broken), op.path or op.heading or "?", broken,
            )
            if cleared_links_out is not None:
                for lnk in broken:
                    cleared_links_out.append({
                        "cleared_link": lnk,
                        "note_heading": op.heading or "",
                        "note_path": op.path or "",
                    })
        prospective_valid.append(op)
    validated_ops = prospective_valid

    return validated_ops, rejected_ops
