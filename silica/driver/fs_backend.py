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
import re
import time
from pathlib import Path
from typing import Any

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

# Basic wikilink extraction: [[target]] or [[target|display]]
WIKILINK_RE = re.compile(r'\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]')

# Heading extraction fallback if OFM fails
HEADING_RE = re.compile(r'^(#{1,6})\s+(.*?)\s*$', re.MULTILINE)

class ObsidianFSBackend:
    """ObsidianDriver implementation using direct filesystem access."""

    def __init__(self, vault_path: str):
        if not vault_path:
            raise ValueError("FS backend requires a valid vault_path")
        self.vault_path = Path(vault_path).resolve()
        
        # In-memory index
        self._notes: dict[str, NoteRef] = {}          # name -> NoteRef
        self._links: dict[str, set[str]] = {}         # source_name -> set(target_name)
        self._backlinks: dict[str, set[str]] = {}     # target_name -> set(source_name)
        self._last_index_time: float = 0.0
        self._needs_reindex: bool = True

    # ------------------------------------------------------------------
    # Indexing (in-memory graph)
    # ------------------------------------------------------------------
    
    def _ensure_index(self):
        if self._needs_reindex:
            self._rebuild_index()

    def _rebuild_index(self):
        logger.debug("Rebuilding FS graph index...")
        self._notes.clear()
        self._links.clear()
        self._backlinks.clear()

        # Find all markdown files
        for root, _, files in os.walk(self.vault_path):
            # Skip hidden folders
            if any(p.startswith(".") for p in Path(root).relative_to(self.vault_path).parts):
                continue
                
            for file in files:
                if not file.endswith(".md"):
                    continue
                
                path = Path(root) / file
                rel_path = path.relative_to(self.vault_path).as_posix()
                name = file[:-3]
                
                ref = NoteRef(name=name, path=rel_path)
                self._notes[name] = ref
                
                try:
                    content = path.read_text(encoding="utf-8")
                    targets = set(WIKILINK_RE.findall(content))
                    
                    self._links[name] = targets
                    for target in targets:
                        if target not in self._backlinks:
                            self._backlinks[target] = set()
                        self._backlinks[target].add(name)
                        
                except Exception as e:
                    logger.warning("Failed to index %s: %s", rel_path, e)

        self._needs_reindex = False
        self._last_index_time = time.time()
        logger.debug("Indexed %d notes", len(self._notes))

    def _resolve_path(self, ref: NoteRef | str) -> Path:
        """Resolve a NoteRef or name to a full filesystem path."""
        self._ensure_index()
        
        if isinstance(ref, NoteRef) and ref.path:
            return self.vault_path / ref.path
            
        name = ref if isinstance(ref, str) else ref.name
        
        # Strip .md if passed in string
        if name.endswith(".md"):
            name = name[:-3]
        
        # Look up in index
        if name in self._notes:
            return self.vault_path / self._notes[name].path
            
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
        for name, ref in self._notes.items():
            if query in name.lower():
                results.append(ref)
        return results

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
        
        return NoteContent(
            ref=NoteRef(name=name, path=path.relative_to(self.vault_path).as_posix()),
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
        name = ref if isinstance(ref, str) else ref.name
        
        targets = self._links.get(name, set())
        results = []
        for t in targets:
            if t in self._notes:
                results.append(self._notes[t])
            else:
                # Unresolved link
                results.append(NoteRef(name=t, path=f"{t}.md"))
        return results

    def backlinks(self, ref: NoteRef | str) -> list[NoteRef]:
        """Incoming links to a note."""
        self._ensure_index()
        name = ref if isinstance(ref, str) else ref.name
        
        sources = self._backlinks.get(name, set())
        results = []
        for s in sources:
            if s in self._notes:
                results.append(self._notes[s])
        return results

    def orphans(self) -> list[NoteRef]:
        """Notes with no incoming links."""
        self._ensure_index()
        results = []
        for name, ref in self._notes.items():
            if name not in self._backlinks or not self._backlinks[name]:
                results.append(ref)
        return results

    def unresolved(self) -> list[Link]:
        """Unresolved wikilinks in the vault."""
        self._ensure_index()
        results = []
        for source, targets in self._links.items():
            source_ref = self._notes.get(source, NoteRef(name=source))
            for target in targets:
                if target not in self._notes:
                    results.append(Link(source=source_ref, target=target))
        return results

    def graph_snapshot(self, refs: list[NoteRef] | None = None) -> GraphSnapshot:
        """Graph snapshot for non-regression gating.

        If refs is provided, performs an incremental snapshot covering only
        the touched notes and their 1-hop neighborhood.
        """
        self._ensure_index()
        if refs is None:
            link_counts = {name: len(targets) for name, targets in self._links.items()}
            backlink_counts = {name: len(sources) for name, sources in self._backlinks.items()}
            return GraphSnapshot(
                orphans=self.orphans(),
                unresolved=self.unresolved(),
                link_counts=link_counts,
                backlink_counts=backlink_counts
            )

        # Incremental snapshot
        neighborhood = set()
        for r in refs:
            name = r.name
            neighborhood.add(name)
            # Add outgoing
            for t in self._links.get(name, []):
                neighborhood.add(t)
            # Add incoming
            for s in self._backlinks.get(name, []):
                neighborhood.add(s)

        link_counts = {}
        backlink_counts = {}
        for name in neighborhood:
            link_counts[name] = len(self._links.get(name, []))
            backlink_counts[name] = len(self._backlinks.get(name, []))

        # Filter orphans & unresolved to neighborhood incrementally
        orphans = [
            self._notes[name] for name in neighborhood
            if name in self._notes and (name not in self._backlinks or not self._backlinks[name])
        ]
        unresolved = []
        for name in neighborhood:
            if name in self._notes:
                source_ref = self._notes[name]
                for target in self._links.get(name, []):
                    if target not in self._notes:
                        unresolved.append(Link(source=source_ref, target=target))

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
        full_path = self.vault_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        full_path.write_text(content, encoding="utf-8")
        self._needs_reindex = True

        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        return NoteRef(name=name, path=path)

    def overwrite(self, path: str, content: str) -> NoteRef:
        """Overwrite an existing note in-place.

        The FS backend does this as a direct write — history is not tracked
        in FS mode, so overwrite and patch rollback via versions is a no-op
        (see restore()). For write-op rollback, created_paths is used instead.
        """
        full_path = self.vault_path / path
        if not full_path.exists():
            raise RuntimeError(f"Cannot overwrite non-existent file: {path}")

        full_path.write_text(content, encoding="utf-8")
        self._needs_reindex = True

        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        return NoteRef(name=name, path=path)

    def append(self, ref: NoteRef | str, content: str) -> None:
        """Append content to an existing note."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
            
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
            
        self._needs_reindex = True

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
        
        src.rename(dst)
        self._needs_reindex = True

    def delete(self, ref: NoteRef | str) -> None:
        """Delete a note from the vault."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
            
        path.unlink()
        self._needs_reindex = True

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
                    self._needs_reindex = True
            except Exception as e:
                logger.error("Failed to delete created note %s during rollback: %s", path, e)
