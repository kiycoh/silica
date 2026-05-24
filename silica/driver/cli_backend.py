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
                    name = item.get("name", item.get("file", ""))
                    path = item.get("path", item.get("file", ""))
                    ref = NoteRef(name=name, path=path)
                    # Handle matches within the item
                    matches = item.get("matches", [item])
                    for match in matches if isinstance(matches, list) else [matches]:
                        if isinstance(match, dict):
                            results.append(Hit(
                                ref=ref,
                                line=match.get("line", 0),
                                snippet=match.get("content", match.get("text", "")),
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
                        text=item.get("heading", item.get("text", "")),
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

    def graph_snapshot(self) -> GraphSnapshot:
        """Full graph snapshot for non-regression gating."""
        return GraphSnapshot(
            orphans=self.orphans(),
            unresolved=self.unresolved(),
        )

    # ------------------------------------------------------------------
    # Write (graph-safe)
    # ------------------------------------------------------------------

    def create(self, path: str, content: str) -> NoteRef:
        """Create a new note at the given vault-relative path."""
        # Escape content for CLI (replace newlines with literal \n)
        escaped = content.replace("\\", "\\\\").replace("\n", "\\n")
        self._run_cli("create", f"path={path}", f"content={escaped}")
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        ref = NoteRef(name=name, path=path)
        self._wait_for_settle(ref)
        return ref

    def append(self, ref: NoteRef | str, content: str) -> None:
        """Append content to an existing note."""
        escaped = content.replace("\\", "\\\\").replace("\n", "\\n")
        self._run_cli("append", self._ref_arg(ref), f"content={escaped}")

    def set_prop(self, ref: NoteRef | str, name: str, value: Any, type_: str = "text") -> None:
        """Set a frontmatter property on a note."""
        self._run_cli(
            "property:set",
            self._ref_arg(ref),
            f"name={name}",
            f"value={value}",
            f"type={type_}",
        )

    def move(self, ref: NoteRef | str, to: str) -> None:
        """Move/rename a note. Obsidian updates all wikilinks (graph-safe)."""
        self._run_cli("move", self._ref_arg(ref), f"to={to}")

    def delete(self, ref: NoteRef | str) -> None:
        """Delete a note from the vault."""
        self._run_cli("delete", self._ref_arg(ref))

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
        """Snapshot current versions for later rollback via history:restore."""
        versions: dict[str, int] = {}
        for ref in refs:
            try:
                raw = self._run_cli("history", self._ref_arg(ref))
                # Count versions — latest version number is the count
                count = len([l for l in raw.splitlines() if l.strip()])
                if count > 0:
                    versions[ref.path or ref.name] = count
            except RuntimeError:
                logger.warning("No history available for %s", ref.name)

        txn_id = f"txn_{int(time.time())}"
        return Txn(id=txn_id, refs=refs, versions=versions)

    def restore(self, txn: Txn) -> None:
        """Rollback to a previous snapshot via history:restore."""
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

    # ------------------------------------------------------------------
    # Freshness contract
    # ------------------------------------------------------------------

    def _wait_for_settle(self, ref: NoteRef, timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until Obsidian's cache reflects a recent write.

        From SILICA.md §3 L0 freshness contract:
          After a create/set_prop/move, the Driver guarantees that the next
          read reflects the mutation. If the cache updates asynchronously,
          the backend MUST wait/poll until settled.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._run_cli("read", self._ref_arg(ref), check=False)
                return  # File is readable — cache has settled
            except RuntimeError:
                time.sleep(_SETTLE_POLL_INTERVAL)

        logger.warning(
            "Settle timeout for %s after %.1fs — cache may be stale",
            ref.name,
            timeout,
        )
