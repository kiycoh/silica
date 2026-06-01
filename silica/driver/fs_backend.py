"""Obsidian FS Backend — L0 implementation via direct filesystem access.

From SILICA.md §3 L0:
  Headless fallback and oracle for non-regression testing. Directly reads the
  filesystem and builds an in-memory graph index.

Note:
  This backend is independent of the Obsidian app, making it suitable for CI
  and headless cron jobs. It manages its own graph index which is refreshed
  as needed.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any
import networkx as nx
from silica.kernel.wikilink import extract_links

from silica.driver.base import (
    GraphSnapshot,
    Heading,
    Hit,
    Link,
    NoteContent,
    NoteRef,
    Txn,
)
from silica.kernel import frontmatter as fm
from silica.kernel import ofm
logger = logging.getLogger(__name__)

class ObsidianFSBackend:
    """ObsidianDriver implementation using direct filesystem access."""

    def __init__(self, vault_path: str):
        if not vault_path:
            raise ValueError("FS backend requires a valid vault_path")
        self.vault_path = Path(vault_path).resolve()
        
        # In-memory index
        self._notes: dict[str, NoteRef] = {}          # path -> NoteRef
        self._notes_by_name: dict[str, list[NoteRef]] = {}  # lower_name -> list of NoteRefs
        self._links: dict[str, set[str]] = {}         # source_path -> set(target_name)
        self._graph = nx.DiGraph()
        self._unresolved_links: set[tuple[str, str]] = set() # (source_path, raw_target)
        self._mention_index: dict[str, set[str]] = {}        # title_lower -> set(path)
        self._needs_reindex: bool = True
        self._dirty_paths: set[str] = set()           # paths patched since last full rebuild

    def _path_of(self, ref: NoteRef | str) -> str | None:
        if isinstance(ref, NoteRef):
            return ref.path
        if ref.endswith(".md"):
            return ref
        matched = self._notes_by_name.get(ref.lower(), [])
        if matched:
            return matched[0].path
        return None

    def _node_ref(self, path: str) -> NoteRef:
        if path in self._notes:
            return self._notes[path]
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        return NoteRef(name=name, path=path)

    # ------------------------------------------------------------------
    # Indexing (in-memory graph)
    # ------------------------------------------------------------------
    
    def _ensure_index(self):
        if self._needs_reindex:
            self._rebuild_index()

    def _resolve_target(self, target: str, source_path: str = "") -> NoteRef | None:
        """Resolve a link target to an existing NoteRef or None if unresolved.

        Obsidian link resolution rules:
        1. If target starts with '#' or '^', it is an internal link. It resolves to the
           source file itself.
        2. If target contains '/', it is a path link. We check if target (or target + '.md')
           matches the end of the path of any existing note (with a leading slash or exact match).
        3. If target does not contain '/', it is a name link. We check if target matches
           the name of any note in the vault. If multiple exist, we prioritize the one in
           the same directory as source_path, then by shortest path.
        """
        if target.startswith('#') or target.startswith('^'):
            if source_path:
                return self._notes.get(source_path)
            return None

        target_no_ext = target.removesuffix(".md")
        if "/" in target:
            p1 = target_no_ext + ".md"
            p1_norm = os.path.normcase(p1.replace("\\", "/").strip("/")).lower()
            
            # Try exact match first
            for path, ref in self._notes.items():
                path_norm = os.path.normcase(path.replace("\\", "/").strip("/")).lower()
                if path_norm == p1_norm:
                    return ref
                    
            # Try matching end of path (suffix matching with /)
            suffix = "/" + p1_norm
            candidates = []
            for path, ref in self._notes.items():
                path_norm = os.path.normcase(path.replace("\\", "/").strip("/")).lower()
                if path_norm.endswith(suffix):
                    candidates.append(ref)
            if candidates:
                # Prioritize same directory as source_path if available
                if source_path and "/" in source_path:
                    source_dir = source_path.rsplit("/", 1)[0]
                    same_dir_candidates = [c for c in candidates if c.path.startswith(source_dir + "/")]
                    if same_dir_candidates:
                        sorted_same = sorted(same_dir_candidates, key=lambda r: (r.path.count("/"), r.path.lower()))
                        return sorted_same[0]
                sorted_candidates = sorted(candidates, key=lambda r: (r.path.count("/"), r.path.lower()))
                return sorted_candidates[0]
            return None
        else:
            refs = self._notes_by_name.get(target_no_ext.lower(), [])
            if not refs:
                return None
            if len(refs) == 1:
                return refs[0]
                
            # Prioritize the one in the same directory as source_path
            if source_path and "/" in source_path:
                source_dir = source_path.rsplit("/", 1)[0]
                same_dir_refs = [r for r in refs if r.path.startswith(source_dir + "/")]
                if same_dir_refs:
                    sorted_same = sorted(same_dir_refs, key=lambda r: (r.path.count("/"), r.path.lower()))
                    return sorted_same[0]
                    
            # Prioritize the one with the shortest vault-relative path
            sorted_refs = sorted(refs, key=lambda r: (r.path.count("/"), r.path.lower()))
            return sorted_refs[0]

    def _rebuild_index(self):
        logger.debug("Rebuilding FS graph index...")
        self._notes.clear()
        self._notes_by_name.clear()
        self._links.clear()
        self._graph.clear()
        self._unresolved_links.clear()
        self._mention_index.clear()

        from silica.config import CONFIG
        inbox_norm = os.path.normcase(CONFIG.inbox_dir.replace("\\", "/").strip("/")) if CONFIG.inbox_dir else None

        files_to_process = []

        # Pass 1: Find all markdown files and populate self._notes and self._graph nodes
        for root, dirs, files in os.walk(self.vault_path):
            rel_path = Path(root).relative_to(self.vault_path).as_posix()

            # Skip hidden folders
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            # Skip inbox directory if configured
            if inbox_norm:
                new_dirs = []
                for d in dirs:
                    sub_rel_path = (Path(rel_path) / d).as_posix().strip(".")
                    sub_rel_norm = os.path.normcase(sub_rel_path.replace("\\", "/").strip("/"))
                    if sub_rel_norm == inbox_norm or sub_rel_norm.startswith(inbox_norm + "/"):
                        logger.debug("Skipping indexing for inbox directory: %s", sub_rel_path)
                    else:
                        new_dirs.append(d)
                dirs[:] = new_dirs
                
            for file in files:
                if not file.endswith(".md"):
                    continue
                
                path = Path(root) / file
                rel_path_file = path.relative_to(self.vault_path).as_posix()

                # Double safety check: skip if rel_path_file is in inbox
                if inbox_norm:
                    rel_path_norm = os.path.normcase(rel_path_file.replace("\\", "/").strip("/"))
                    if rel_path_norm == inbox_norm or rel_path_norm.startswith(inbox_norm + "/"):
                        continue
                
                name = file[:-3]
                ref = NoteRef(name=name, path=rel_path_file)
                self._notes[rel_path_file] = ref
                self._graph.add_node(rel_path_file, ref=ref)
                
                name_lower = name.lower()
                if name_lower not in self._notes_by_name:
                    self._notes_by_name[name_lower] = []
                self._notes_by_name[name_lower].append(ref)
                
                files_to_process.append((rel_path_file, path))

        # Pass 2: Parse and resolve links + build mention index
        for rel_path_file, path in files_to_process:
            try:
                content = path.read_text(encoding="utf-8")
                targets = set(extract_links(content))
                self._links[rel_path_file] = targets
                for target in targets:
                    ref = self._resolve_target(target, source_path=rel_path_file)
                    if ref:
                        self._graph.add_edge(rel_path_file, ref.path)
                    else:
                        self._unresolved_links.add((rel_path_file, target))
                
                # Mention index
                content_lower = content.lower()
                for title_lower in self._notes_by_name:
                    if len(title_lower) >= 2 and title_lower in content_lower:
                        self._mention_index.setdefault(title_lower, set()).add(rel_path_file)
            except Exception as e:
                logger.warning("Failed to index %s: %s", rel_path_file, e)

        self._needs_reindex = False
        self._dirty_paths.clear()
        logger.debug("Indexed %d notes", len(self._notes))

    def _patch_index(self, rel_path: str, content: str | None) -> None:
        """Incrementally update the graph index for a single changed path.

        If content is None the note was deleted — remove it from the index.
        Call this instead of setting _needs_reindex = True for single-file writes.
        """
        # --- remove stale data for this path ---
        if rel_path in self._graph:
            self._graph.remove_edges_from(list(self._graph.out_edges(rel_path)))
        self._unresolved_links = {(s, t) for s, t in self._unresolved_links if s != rel_path}
        for paths_set in self._mention_index.values():
            paths_set.discard(rel_path)

        if content is None:
            # deletion path
            if rel_path in self._graph:
                self._graph.remove_node(rel_path)
            old_ref = self._notes.pop(rel_path, None)
            if old_ref:
                name_lower = old_ref.name.lower()
                if name_lower in self._notes_by_name:
                    self._notes_by_name[name_lower] = [
                        r for r in self._notes_by_name[name_lower] if r.path != rel_path
                    ]
            self._dirty_paths.discard(rel_path)
            return

        # --- upsert node ---
        name = rel_path.rsplit("/", 1)[-1].removesuffix(".md")
        ref = NoteRef(name=name, path=rel_path)
        self._notes[rel_path] = ref
        self._graph.add_node(rel_path, ref=ref)
        name_lower = name.lower()
        if name_lower not in self._notes_by_name:
            self._notes_by_name[name_lower] = []
        if ref not in self._notes_by_name[name_lower]:
            self._notes_by_name[name_lower].append(ref)

        # --- rebuild edges for this path ---
        targets = set(extract_links(content))
        self._links[rel_path] = targets
        for target in targets:
            target_ref = self._resolve_target(target, source_path=rel_path)
            if target_ref:
                self._graph.add_edge(rel_path, target_ref.path)
            else:
                self._unresolved_links.add((rel_path, target))

        # --- rebuild mention index for this path ---
        content_lower = content.lower()
        for title_lower in self._notes_by_name:
            if len(title_lower) >= 2 and title_lower in content_lower:
                self._mention_index.setdefault(title_lower, set()).add(rel_path)

        self._dirty_paths.add(rel_path)

    def _resolve_path(self, ref: NoteRef | str) -> Path:
        """Resolve a NoteRef or name to a full filesystem path."""
        self._ensure_index()
        
        if isinstance(ref, NoteRef) and ref.path:
            p = Path(ref.path)
            if p.is_absolute():
                return p
            return self.vault_path / ref.path
            
        name = ref if isinstance(ref, str) else ref.name
        
        # Strip .md if passed in string
        if name.endswith(".md"):
            name = name[:-3]
        
        # Look up in index
        matched = self._notes_by_name.get(name.lower(), [])
        if matched:
            return self.vault_path / matched[0].path
            
        # Check if the name/ref is actually a path pointing directly to an existing file
        p = Path(name + ".md")
        if p.exists():
            return p.resolve()
        p = Path(name)
        if p.exists():
            return p.resolve()
            
        # Fallback for new files not yet in index
        return self.vault_path / f"{name}.md"

    # ------------------------------------------------------------------
    # Discovery / Read
    # ------------------------------------------------------------------

    def search_names(self, query: str) -> list[NoteRef]:
        """Search vault note names matching query."""
        self._ensure_index()
        query = query.lower()
        results = []
        for ref in self._notes.values():
            if query in ref.name.lower():
                results.append(ref)
        return results

    def mentions_of(self, title: str) -> list[str]:
        """Return vault-relative paths of notes whose body mentions `title`.
        
        O(1) lookup into the inverted text index built during indexing.
        """
        self._ensure_index()
        return list(self._mention_index.get(title.lower(), set()))

    def search_context(self, query: str) -> list[Hit]:
        """Search vault content with line-level context snippets."""
        self._ensure_index()
        query_lower = query.lower()
        results = []
        
        for name, ref in self._notes.items():
            path = self.vault_path / ref.path
            try:
                content = path.read_text(encoding="utf-8")
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if query_lower in line.lower():
                        # Extract a snippet around the match
                        results.append(Hit(
                            ref=ref,
                            line=i + 1,
                            snippet=line.strip()
                        ))
            except Exception:
                continue
                
        return results

    def read_note(self, ref: NoteRef | str) -> NoteContent:
        """Read a note's full content by name or ref."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
            
        content = path.read_text(encoding="utf-8")
        name = ref if isinstance(ref, str) else ref.name
        
        try:
            rel_path = path.relative_to(self.vault_path).as_posix()
        except ValueError:
            # Fallback for external files outside the vault
            rel_path = path.resolve().as_posix()
            
        return NoteContent(
            ref=NoteRef(name=name, path=rel_path),
            content=content,
            size=len(content)
        )

    def props_of(self, ref: NoteRef | str) -> dict:
        """Read frontmatter properties."""
        try:
            nc = self.read_note(ref)
            data, _, _ = fm.split(nc.content)
            return data or {}
        except RuntimeError:
            return {}

    def outline(self, ref: NoteRef | str) -> list[Heading]:
        """Get the heading tree of a note."""
        try:
            nc = self.read_note(ref)
            raw_headings = ofm.parse_headings(nc.content)
            return [
                Heading(
                    level=h["level"],
                    text=h["text"],
                    position=h["pos"]
                ) for h in raw_headings
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def links(self, ref: NoteRef | str) -> list[NoteRef]:
        """Outgoing links from a note."""
        self._ensure_index()
        path = self._path_of(ref)
        if not path:
            return []
        
        # Resolved outgoing links from graph
        results = []
        if path in self._graph:
            results.extend([self._node_ref(t) for t in self._graph.successors(path)])
        
        # Unresolved outgoing links
        for s, t in self._unresolved_links:
            if s == path:
                t_name = t.rsplit("/", 1)[-1].removesuffix(".md")
                results.append(NoteRef(name=t_name, path=f"{t_name}.md"))
                
        return results

    def backlinks(self, ref: NoteRef | str) -> list[NoteRef]:
        """Incoming links to a note."""
        self._ensure_index()
        path = self._path_of(ref)
        if not path or path not in self._graph:
            return []
        return [self._node_ref(s) for s in self._graph.predecessors(path)]

    def orphans(self) -> list[NoteRef]:
        """Notes with no incoming links."""
        self._ensure_index()
        return [self._graph.nodes[n]["ref"] for n, d in self._graph.in_degree() if d == 0]

    def unresolved(self) -> list[Link]:
        """Unresolved wikilinks in the vault."""
        self._ensure_index()
        results = []
        for s, t in self._unresolved_links:
            results.append(Link(source=self._node_ref(s), target=t.removesuffix(".md")))
        return results

    def graph_data(self, folder: str = "") -> tuple[dict, set, Any]:
        """Return (notes, unresolved_links, graph) for in-process consumers."""
        self._ensure_index()
        return self._notes, self._unresolved_links, self._graph

    def graph_snapshot(self, refs: list[NoteRef] | None = None) -> GraphSnapshot:
        """Graph snapshot for non-regression gating.

        If refs is provided, performs an incremental snapshot covering only
        the touched notes and their 1-hop neighborhood.
        """
        self._ensure_index()
        if refs is None:
            link_counts = {}
            for path, ref in self._notes.items():
                resolved_count = self._graph.out_degree(path) if path in self._graph else 0
                unresolved_count = sum(1 for s, t in self._unresolved_links if s == path)
                # Key by canonical path (no .md) — unique even with duplicate basenames.
                # graph_diff.normalize_path() strips .md and lowercases, so path-keyed
                # snapshots compare identically to name-keyed ones in the diff.
                key = path.removesuffix(".md")
                link_counts[key] = resolved_count + unresolved_count

            backlink_counts = {
                path.removesuffix(".md"): d
                for path, d in self._graph.in_degree()
            }
            return GraphSnapshot(
                orphans=self.orphans(),
                unresolved=self.unresolved(),
                link_counts=link_counts,
                backlink_counts=backlink_counts
            )

        # Incremental snapshot
        neighborhood = set()
        for r in refs:
            if r.path:
                neighborhood.add(r.path)
                # Add outgoing
                if r.path in self._graph:
                    for t in self._graph.successors(r.path):
                        neighborhood.add(t)
                # Add incoming
                if r.path in self._graph:
                    for s in self._graph.predecessors(r.path):
                        neighborhood.add(s)

        link_counts = {}
        backlink_counts = {}
        for path in neighborhood:
            note = self._notes.get(path)
            if note:
                resolved_count = self._graph.out_degree(path) if path in self._graph else 0
                unresolved_count = sum(1 for s, t in self._unresolved_links if s == path)
                key = path.removesuffix(".md")
                link_counts[key] = resolved_count + unresolved_count
                backlink_counts[key] = self._graph.in_degree(path) if path in self._graph else 0

        # Filter orphans & unresolved to neighborhood incrementally
        orphans = [
            self._notes[path] for path in neighborhood
            if path in self._notes and (path not in self._graph or self._graph.in_degree(path) == 0)
        ]
        unresolved = []
        for path in neighborhood:
            if path in self._notes:
                source_ref = self._notes[path]
                for s, t in self._unresolved_links:
                    if s == path:
                        unresolved.append(Link(source=source_ref, target=t.removesuffix(".md")))

        return GraphSnapshot(
            orphans=orphans,
            unresolved=unresolved,
            link_counts=link_counts,
            backlink_counts=backlink_counts
        )

    # ------------------------------------------------------------------
    # Write (graph-safe)
    # ------------------------------------------------------------------

    def create(self, path: str, content: str) -> NoteRef:
        """Create a new note at the given vault-relative path."""
        p = Path(path)
        if p.is_absolute():
            try:
                rel_path = p.relative_to(self.vault_path).as_posix()
            except ValueError:
                rel_path = p.as_posix()
        else:
            rel_path = p.as_posix()

        full_path = self.vault_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        full_path.write_text(content, encoding="utf-8")
        name = rel_path.rsplit("/", 1)[-1].removesuffix(".md")
        if self._needs_reindex:
            self._rebuild_index()
        else:
            self._patch_index(rel_path, content)
        return NoteRef(name=name, path=rel_path)

    def overwrite(self, path: str, content: str) -> NoteRef:
        """Overwrite an existing note in-place.

        The FS backend does this as a direct write — history is not tracked
        in FS mode, so overwrite and patch rollback via versions is a no-op
        (see restore()). For write-op rollback, created_paths is used instead.
        """
        p = Path(path)
        if p.is_absolute():
            try:
                rel_path = p.relative_to(self.vault_path).as_posix()
            except ValueError:
                rel_path = p.as_posix()
        else:
            rel_path = p.as_posix()

        full_path = self.vault_path / rel_path
        if not full_path.exists():
            raise RuntimeError(f"Cannot overwrite non-existent file: {path}")

        full_path.write_text(content, encoding="utf-8")
        name = rel_path.rsplit("/", 1)[-1].removesuffix(".md")
        if self._needs_reindex:
            self._rebuild_index()
        else:
            self._patch_index(rel_path, content)
        return NoteRef(name=name, path=rel_path)

    def append(self, ref: NoteRef | str, content: str) -> None:
        """Append content to an existing note."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
            
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)

        rel_path_str = path.relative_to(self.vault_path).as_posix()
        if self._needs_reindex:
            self._rebuild_index()
        else:
            full_content = path.read_text(encoding="utf-8")
            self._patch_index(rel_path_str, full_content)

    def set_prop(self, ref: NoteRef | str, name: str, value: Any, type_: str = "text") -> None:
        """Set a frontmatter property on a note."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
            
        content = path.read_text(encoding="utf-8")
        data, delim, body = fm.split(content)
        
        if data is None:
            data = {}
            
        data[name] = value
        
        new_content = fm.dump(data, body)
        path.write_text(new_content, encoding="utf-8")

    def move(self, ref: NoteRef | str, to: str) -> None:
        """Move/rename a note. 
        
        Note: The FS backend currently does NOT update wikilinks like Obsidian does.
        This is a known limitation compared to the CLI backend.
        """
        src = self._resolve_path(ref)
        if not src.exists():
            raise RuntimeError(f"File not found: {src}")
            
        dst = self.vault_path / to
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        src_rel = src.relative_to(self.vault_path).as_posix()
        src.rename(dst)
        if self._needs_reindex:
            self._rebuild_index()
        else:
            dst_content = dst.read_text(encoding="utf-8")
            self._patch_index(src_rel, None)
            dst_rel = dst.relative_to(self.vault_path).as_posix()
            self._patch_index(dst_rel, dst_content)

    def delete(self, ref: NoteRef | str) -> None:
        """Delete a note from the vault."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
            
        rel_path_str = path.relative_to(self.vault_path).as_posix()
        path.unlink()
        if self._needs_reindex:
            self._rebuild_index()
        else:
            self._patch_index(rel_path_str, None)

    # ------------------------------------------------------------------
    # Advanced
    # ------------------------------------------------------------------

    def list_files(self, folder: str = "") -> list[NoteRef]:
        """List all markdown files, optionally filtered by folder."""
        self._ensure_index()
        
        results = []
        for ref in self._notes.values():
            if not folder or ref.path.startswith(folder):
                results.append(ref)
                
        return results

    def list_inbox_files(self) -> list[NoteRef]:
        """List all files in the inbox directory."""
        from silica.config import CONFIG
        if not CONFIG.inbox_dir:
            return []
        inbox_path = self.vault_path / CONFIG.inbox_dir
        if not inbox_path.exists() or not inbox_path.is_dir():
            return []
        results = []
        for root, _, files in os.walk(inbox_path):
            for file in files:
                if file.endswith(".md"):
                    full_p = Path(root) / file
                    try:
                        rel_p = full_p.relative_to(self.vault_path).as_posix()
                    except ValueError:
                        rel_p = full_p.resolve().as_posix()
                    name = file.removesuffix(".md")
                    results.append(NoteRef(name=name, path=rel_p))
        return results

    def base_query(self, base: str, view: str) -> list[dict]:
        """Query an Obsidian Base (not implemented in FS backend)."""
        logger.warning("base_query not implemented in FS backend")
        return []

    # ------------------------------------------------------------------
    # Transactionality
    # ------------------------------------------------------------------

    def snapshot_versions(self, refs: list[NoteRef]) -> Txn:
        """Snapshot current versions for later rollback.

        The FS backend does not track version history, so `versions` is always
        empty. Rollback of patch ops is a no-op in FS mode. Rollback of write
        ops works via `created_paths` (delete the created notes).
        """
        txn_id = f"txn_fs_{int(time.time())}"
        return Txn(id=txn_id, refs=refs, versions={})

    def restore(self, txn: Txn) -> None:
        """Rollback a transaction.

        - versions: no-op in FS backend (no history tracking).
        - created_paths: deletes newly-created notes to undo write ops.
        """
        if txn.versions:
            logger.warning(
                "FS backend cannot restore note history (versions). "
                "Patch rollback is a no-op. Consider using the CLI backend for full rollback support."
            )

        for path in txn.created_paths:
            try:
                full_path = self.vault_path / path
                if full_path.exists():
                    full_path.unlink()
                    logger.info("Rolled back created note: %s", path)
                    if self._needs_reindex:
                        pass  # full rebuild will happen on next _ensure_index
                    else:
                        self._patch_index(path, None)
            except Exception as e:
                logger.error("Failed to delete created note %s during rollback: %s", path, e)
