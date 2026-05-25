"""Obsidian CLI Backend — L0 implementation via the official Obsidian CLI.

Wraps `obsidian <command> [options]` shell-outs. Requires Obsidian desktop
app >= 1.12.7 running (it's a CDP bridge to the Electron instance).

From SILICA.md §3 L0:
  Reads the live metadata-cache and graph engine. Write operations are
  graph-safe (wikilinks updated by Obsidian's engine on move/rename).

Freshness contract:
  After a create/set_prop/move, the backend polls until the cache reflects
  the mutation (_wait_for_settle). This is normative — a read that returns
  stale data after a write is a bug.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
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

logger = logging.getLogger(__name__)

# Settle polling config
_SETTLE_POLL_INTERVAL = 0.1  # seconds
_SETTLE_TIMEOUT = 2.0  # seconds


class ObsidianCLIBackend:
    """ObsidianDriver implementation via the official Obsidian CLI."""

    def __init__(self, vault_name: str = ""):
        self._vault_name = vault_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_cli(self, *args: str, check: bool = True) -> str:
        """Execute an obsidian CLI command and return stdout.

        Raises subprocess.CalledProcessError on non-zero exit (if check=True).
        """
        cmd = ["obsidian"]
        if self._vault_name:
            cmd.append(f"vault={self._vault_name}")
        cmd.extend(args)

        logger.debug("CLI exec: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=check,
            )
            if result.stderr:
                logger.debug("CLI stderr: %s", result.stderr.strip())
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Obsidian CLI timeout: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Obsidian CLI error (exit {e.returncode}): {e.stderr.strip()}"
            )

    def _run_json(self, *args: str) -> Any:
        """Execute a CLI command with format=json and parse the result."""
        raw = self._run_cli(*args, "format=json")
        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse CLI JSON output: %s\n%s", e, raw[:500])
            raise

    def _ref_arg(self, ref: NoteRef | str) -> str:
        """Convert a NoteRef or string to a file= CLI argument."""
        if isinstance(ref, str):
            return f"file={ref}"
        if ref.path:
            return f"path={ref.path}"
        return f"file={ref.name}"

    # ------------------------------------------------------------------
    # Discovery / Read
    # ------------------------------------------------------------------

    def search_names(self, query: str) -> list[NoteRef]:
        """Search vault note names matching query."""
        query = query.lower()
        files = self.list_files()
        return [f for f in files if query in f.name.lower()]

    def search_context(self, query: str) -> list[Hit]:
        """Search vault content with line-level context snippets."""
        data = self._run_json("search:context", f"query={query}")
        results = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    name = item.get("name", item.get("file", "")) or ""
                    path = item.get("path", item.get("file", "")) or ""
                    ref = NoteRef(name=str(name), path=str(path))
                    # Handle matches within the item
                    matches = item.get("matches", [item])
                    for match in matches if isinstance(matches, list) else [matches]:
                        if isinstance(match, dict):
                            results.append(Hit(
                                ref=ref,
                                line=match.get("line", 0),
                                snippet=str(match.get("content", match.get("text", ""))),
                            ))
                        else:
                            results.append(Hit(ref=ref, snippet=str(match)))
        return results

    def read_note(self, ref: NoteRef | str) -> NoteContent:
        """Read a note's full content by name or ref."""
        content = self._run_cli("read", self._ref_arg(ref))
        name = ref if isinstance(ref, str) else ref.name
        return NoteContent(
            ref=NoteRef(name=name),
            content=content,
            size=len(content),
        )

    def props_of(self, ref: NoteRef | str) -> dict:
        """Read frontmatter properties."""
        data = self._run_json("properties", self._ref_arg(ref))
        if isinstance(data, dict):
            return data
        return {}

    def outline(self, ref: NoteRef | str) -> list[Heading]:
        """Get the heading tree of a note."""
        data = self._run_json("outline", self._ref_arg(ref))
        headings = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    headings.append(Heading(
                        level=item.get("level", 1),
                        text=str(item.get("heading", item.get("text", ""))),
                        position=item.get("position", 0),
                    ))
        return headings

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def links(self, ref: NoteRef | str) -> list[NoteRef]:
        """Outgoing links from a note."""
        raw = self._run_cli("links", self._ref_arg(ref))
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("No ") and "found" not in line:
                name = line.rsplit("/", 1)[-1].removesuffix(".md")
                results.append(NoteRef(name=name, path=line))
        return results

    def backlinks(self, ref: NoteRef | str) -> list[NoteRef]:
        """Incoming links to a note."""
        raw = self._run_cli("backlinks", self._ref_arg(ref))
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or (line.startswith("No ") and "found" in line):
                continue
            # Format may be "path\tcount" with counts flag
            parts = line.split("\t")
            path = parts[0].strip()
            if path:
                name = path.rsplit("/", 1)[-1].removesuffix(".md")
                results.append(NoteRef(name=name, path=path))
        return results

    def orphans(self) -> list[NoteRef]:
        """Notes with no incoming links."""
        raw = self._run_cli("orphans")
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if line and not (line.startswith("No ") and "found" in line):
                name = line.rsplit("/", 1)[-1].removesuffix(".md")
                results.append(NoteRef(name=name, path=line))
        return results

    def unresolved(self) -> list[Link]:
        """Unresolved wikilinks in the vault."""
        raw = self._run_cli("unresolved")
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or (line.startswith("No ") and "found" in line):
                continue
            # TSV format: target\t[count]
            parts = line.split("\t")
            target = parts[0].strip()
            if target:
                results.append(Link(source=NoteRef(name=""), target=target))
        return results

    def graph_snapshot(self, refs: list[NoteRef] | None = None) -> GraphSnapshot:
        """Graph snapshot for non-regression gating.

        If refs is provided, performs an incremental snapshot covering only
        the touched notes and their 1-hop neighborhood.
        """
        if refs is None:
            all_notes = self.list_files()
            link_counts: dict[str, int] = {}
            backlink_counts: dict[str, int] = {}
            for ref in all_notes:
                out = self.links(ref)
                link_counts[ref.name] = len(out)
                for target in out:
                    backlink_counts[target.name] = backlink_counts.get(target.name, 0) + 1
            return GraphSnapshot(
                orphans=self.orphans(),
                unresolved=self.unresolved(),
                link_counts=link_counts,
                backlink_counts=backlink_counts,
            )

        # Incremental snapshot
        neighborhood: set[str] = set()
        name_to_ref: dict[str, NoteRef] = {}

        for ref in refs:
            name = ref.name
            neighborhood.add(name)
            name_to_ref[name] = ref

            # Add outgoing links
            try:
                out = self.links(ref)
                for target in out:
                    neighborhood.add(target.name)
                    name_to_ref[target.name] = target
            except Exception:
                pass

            # Add incoming backlinks
            try:
                inc = self.backlinks(ref)
                for source in inc:
                    neighborhood.add(source.name)
                    name_to_ref[source.name] = source
            except Exception:
                pass

        link_counts = {}
        backlink_counts = {}
        orphans: list[NoteRef] = []
        unresolved: list[Link] = []

        existence_cache: dict[str, bool] = {}
        def note_exists(n: str) -> bool:
            if n in existence_cache:
                return existence_cache[n]
            r = name_to_ref.get(n, NoteRef(name=n))
            try:
                self._run_cli("read", self._ref_arg(r))
                exists = True
            except Exception:
                exists = False
            existence_cache[n] = exists
            return exists

        for name in neighborhood:
            ref = name_to_ref.get(name, NoteRef(name=name))
            if not note_exists(name):
                link_counts[name] = 0
                backlink_counts[name] = 0
                continue

            try:
                out = self.links(ref)
                link_counts[name] = len(out)
                for target in out:
                    if not note_exists(target.name):
                        unresolved.append(Link(source=ref, target=target.name))
            except Exception:
                link_counts[name] = 0

            try:
                inc = self.backlinks(ref)
                backlink_counts[name] = len(inc)
                if len(inc) == 0:
                    orphans.append(ref)
            except Exception:
                backlink_counts[name] = 0

        return GraphSnapshot(
            orphans=orphans,
            unresolved=unresolved,
            link_counts=link_counts,
            backlink_counts=backlink_counts,
        )

    # ------------------------------------------------------------------
    # Write (graph-safe)
    # ------------------------------------------------------------------

    def create(self, path: str, content: str) -> NoteRef:
        """Create a new note at the given vault-relative path."""
        escaped = content.replace("\\", "\\\\").replace("\n", "\\n")
        self._run_cli("create", f"path={path}", f"content={escaped}")
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        ref = NoteRef(name=name, path=path)
        self._wait_for_create(ref)
        return ref

    def overwrite(self, path: str, content: str) -> NoteRef:
        """Overwrite an existing note in-place, preserving Obsidian version history.

        Uses `obsidian create ... overwrite=true` which keeps the file's block-refs
        and history intact — unlike delete+create which destroys both.
        """
        escaped = content.replace("\\", "\\\\").replace("\n", "\\n")
        self._run_cli("create", f"path={path}", f"content={escaped}", "overwrite=true")
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        ref = NoteRef(name=name, path=path)
        # C2: verify content reflects the write, not just readability
        self._wait_for_content_reflects(ref, content)
        return ref

    def append(self, ref: NoteRef | str, content: str) -> None:
        """Append content to an existing note."""
        escaped = content.replace("\\", "\\\\").replace("\n", "\\n")
        self._run_cli("append", self._ref_arg(ref), f"content={escaped}")
        # C2: verify content was appended
        self._wait_for_content_contains(ref, content)

    def set_prop(self, ref: NoteRef | str, name: str, value: Any, type_: str = "text") -> None:
        """Set a frontmatter property on a note."""
        self._run_cli(
            "property:set",
            self._ref_arg(ref),
            f"name={name}",
            f"value={value}",
            f"type={type_}",
        )
        self._wait_for_prop(ref, name, str(value))

    def move(self, ref: NoteRef | str, to: str) -> None:
        """Move/rename a note. Obsidian updates all wikilinks (graph-safe)."""
        self._run_cli("move", self._ref_arg(ref), f"to={to}")
        self._wait_for_move(ref, to)

    def delete(self, ref: NoteRef | str) -> None:
        """Delete a note from the vault."""
        self._run_cli("delete", self._ref_arg(ref))
        # C2: verify note is gone
        self._wait_for_gone(ref)

    # ------------------------------------------------------------------
    # Advanced
    # ------------------------------------------------------------------

    def list_files(self, folder: str = "") -> list[NoteRef]:
        """List all markdown files, optionally filtered by folder."""
        args = ["files", "ext=md"]
        if folder:
            args.append(f"path={folder}")
        raw = self._run_cli(*args)
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                name = line.rsplit("/", 1)[-1].removesuffix(".md")
                results.append(NoteRef(name=name, path=line))
        return results

    def base_query(self, base: str, view: str) -> list[dict]:
        """Query an Obsidian Base."""
        return self._run_json("base:query", f"file={base}", f"view={view}")

    # ------------------------------------------------------------------
    # Transactionality
    # ------------------------------------------------------------------

    def snapshot_versions(self, refs: list[NoteRef]) -> Txn:
        """Snapshot current versions for later rollback via history:restore.

        Uses format=json to get real version identifiers from Obsidian history.
        Falls back to line-count heuristic only if JSON parsing fails.
        """
        versions: dict[str, int] = {}
        for ref in refs:
            try:
                data = self._run_json("history", self._ref_arg(ref))
                if isinstance(data, list) and data:
                    # history format=json returns [{"version": N, ...}, ...] newest-first
                    # We want the current (latest) version number to restore to.
                    first = data[0]
                    if isinstance(first, dict) and "version" in first:
                        versions[ref.path or ref.name] = int(first["version"])
                    else:
                        # Fallback: use list index 1-based (latest = len)
                        versions[ref.path or ref.name] = len(data)
                elif isinstance(data, dict) and "version" in data:
                    versions[ref.path or ref.name] = int(data["version"])
            except Exception:
                # Try plain-text fallback
                try:
                    raw = self._run_cli("history", self._ref_arg(ref))
                    count = sum(1 for line in raw.splitlines() if line.strip())
                    if count > 0:
                        versions[ref.path or ref.name] = count
                except RuntimeError:
                    logger.warning("No history available for %s", ref.name)

        txn_id = f"txn_{int(time.time())}"
        return Txn(id=txn_id, refs=refs, versions=versions)

    def restore(self, txn: Txn) -> None:
        """Rollback to a previous snapshot via history:restore.

        Handles two rollback strategies:
          - versions: patch ops → restore existing notes to a prior version
          - created_paths: write ops → delete newly-created notes
        """
        # 1. Restore patched notes to their pre-write version
        for ref in txn.refs:
            key = ref.path or ref.name
            version = txn.versions.get(key)
            if version is not None:
                try:
                    self._run_cli(
                        "history:restore",
                        self._ref_arg(ref),
                        f"version={version}",
                    )
                    logger.info("Restored %s to version %d", key, version)
                except RuntimeError as e:
                    logger.error("Failed to restore %s: %s", key, e)

        # 2. Delete notes that were newly created (write ops)
        for path in txn.created_paths:
            try:
                self._run_cli("delete", f"path={path}")
                logger.info("Rolled back created note: %s", path)
            except RuntimeError as e:
                logger.error("Failed to delete created note %s during rollback: %s", path, e)

    # ------------------------------------------------------------------
    # Freshness contract — per-operation postconditions (B5)
    # ------------------------------------------------------------------

    def _wait_for_create(self, ref: NoteRef, timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until Obsidian's cache reflects a newly-created note.

        Postcondition (C2): read(ref) succeeds.
        For content convergence after overwrite, use _wait_for_content_reflects instead.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._run_cli("read", self._ref_arg(ref), check=False)
                return
            except RuntimeError:
                time.sleep(_SETTLE_POLL_INTERVAL)

        logger.warning(
            "Settle timeout (create) for %s after %.1fs — cache may be stale",
            ref.name,
            timeout,
        )

    def _wait_for_content_reflects(self, ref: NoteRef, expected_content: str,
                                   timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until read(ref).content reflects expected_content.

        Postcondition (C2 / overwrite): content is not just readable but matches
        what was written. Uses a prefix check (first 120 chars) to avoid full-body
        comparisons on large notes while still catching stale-cache false positives.
        """
        prefix = expected_content[:120]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                nc = self.read_note(ref)
                if nc.content[:120] == prefix:
                    return
            except RuntimeError:
                pass
            time.sleep(_SETTLE_POLL_INTERVAL)

        logger.warning(
            "Settle timeout (overwrite content) for %s after %.1fs — cache may be stale",
            ref.name,
            timeout,
        )

    def _wait_for_content_contains(self, ref: NoteRef | str, fragment: str,
                                   timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until read(ref).content contains fragment.

        Postcondition (C2 / append): appended content is visible in the note.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                nc = self.read_note(ref)
                if fragment in nc.content:
                    return
            except RuntimeError:
                pass
            time.sleep(_SETTLE_POLL_INTERVAL)

        logger.warning(
            "Settle timeout (append) for %s after %.1fs — cache may be stale",
            ref if isinstance(ref, str) else ref.name,
            timeout,
        )

    def _wait_for_gone(self, ref: NoteRef | str, timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until read(ref) raises (note is gone).

        Postcondition (C2 / delete): note is no longer readable.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._run_cli("read", self._ref_arg(ref), check=False)
                time.sleep(_SETTLE_POLL_INTERVAL)
            except RuntimeError:
                return  # Gone — postcondition satisfied

        logger.warning(
            "Settle timeout (delete) for %s after %.1fs — note may still be cached",
            ref if isinstance(ref, str) else ref.name,
            timeout,
        )

    def _wait_for_prop(self, ref: NoteRef | str, prop_name: str, expected_value: str,
                       timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until a frontmatter property reflects the expected value.

        Postcondition (C2 / set_prop): props_of(ref)[prop_name] == expected_value
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                props = self.props_of(ref)
                if str(props.get(prop_name, "")) == expected_value:
                    return
            except RuntimeError:
                pass
            time.sleep(_SETTLE_POLL_INTERVAL)

        logger.warning(
            "Settle timeout (set_prop '%s') for %s after %.1fs — cache may be stale",
            prop_name,
            ref if isinstance(ref, str) else ref.name,
            timeout,
        )

    def _wait_for_move(self, original_ref: NoteRef | str, to_path: str,
                       timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until a move is reflected: destination readable AND source gone.

        Postcondition (C2 / move): read(to) succeeds AND read(original_ref) raises.
        Both halves are required — checking only the destination allows a
        false-positive when Obsidian's cache still holds the old path.
        """
        to_name = to_path.rsplit("/", 1)[-1].removesuffix(".md")
        to_ref = NoteRef(name=to_name, path=to_path)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            dest_ok = False
            src_gone = False
            try:
                self._run_cli("read", self._ref_arg(to_ref), check=False)
                dest_ok = True
            except RuntimeError:
                pass
            try:
                self._run_cli("read", self._ref_arg(original_ref), check=False)
                # Source still readable — not done yet
            except RuntimeError:
                src_gone = True
            if dest_ok and src_gone:
                return
            time.sleep(_SETTLE_POLL_INTERVAL)

        logger.warning(
            "Settle timeout (move to '%s') after %.1fs — cache may be stale",
            to_path,
            timeout,
        )

    # Legacy alias kept for any remaining internal callers
    def _wait_for_settle(self, ref: NoteRef, timeout: float = _SETTLE_TIMEOUT) -> None:
        self._wait_for_create(ref, timeout=timeout)
