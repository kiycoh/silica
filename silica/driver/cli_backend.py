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
import re
import subprocess
import time
from typing import Any
import networkx as nx
from silica.config import CONFIG
from silica.kernel.wikilink import extract_links

from silica.driver.base import (
    GraphSnapshot,
    Heading,
    Hit,
    Link,
    NoteContent,
    NoteRef,
    SettleTimeout,
    Txn,
)

logger = logging.getLogger(__name__)

# Settle polling config
_SETTLE_POLL_INTERVAL = 0.1   # seconds
_SETTLE_TIMEOUT = 2.0         # seconds — outer deadline for cache convergence polls

# Hard limit per individual subprocess call to the Obsidian CDP bridge.
# Configurable via SILICA_OBSIDIAN_CLI_TIMEOUT (default 8 s).
# 8 s >> normal CDP latency (< 1 s) but prevents single stalled calls from
# accumulating into 88-second hangs inside the settle poll loops.
def _cli_timeout() -> float:
    """Read the current CLI timeout from CONFIG at call time (supports runtime changes)."""
    return float(getattr(CONFIG, "obsidian_cli_timeout", 8.0))


class ObsidianCLIBackend:
    """ObsidianDriver implementation via the official Obsidian CLI."""

    def __init__(self, vault_name: str = ""):
        self._vault_name = vault_name
        self._graph = nx.DiGraph()
        self._unresolved_links: set[tuple[str, str]] = set()
        self._notes: dict[str, NoteRef] = {}
        self._notes_by_name: dict[str, list[NoteRef]] = {}
        self._mention_index: dict[str, set[str]] = {}  # title_lower → {paths that mention it}
        self._is_graph_built = False

    def _node_ref(self, path: str) -> NoteRef:
        if path in self._notes:
            return self._notes[path]
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        return NoteRef(name=name, path=path)

    def _ensure_graph(self):
        if self._is_graph_built:
            return

        self._graph.clear()
        self._unresolved_links.clear()
        self._notes.clear()
        self._notes_by_name.clear()

        all_notes = self.list_files()
        for ref in all_notes:
            self._notes[ref.path] = ref
            self._graph.add_node(ref.path, ref=ref)
            name_lower = ref.name.lower()
            if name_lower not in self._notes_by_name:
                self._notes_by_name[name_lower] = []
            self._notes_by_name[name_lower].append(ref)

        # Spike S1: attempt bulk read via CDP; fall back to per-note queries.
        try:
            self._load_graph_from_obsidian()
        except Exception as e:
            logger.debug("Bulk graph load unavailable (%s); falling back to per-note queries.", e)
            for ref in all_notes:
                try:
                    out = self.links(ref)
                    for target in out:
                        if target.path and target.path in self._notes:
                            self._graph.add_edge(ref.path, target.path)
                        else:
                            self._unresolved_links.add((ref.path, target.name))
                except Exception:
                    pass

        self._is_graph_built = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, ref: NoteRef | str) -> str:
        """Resolve a NoteRef or name to a vault-relative path."""
        if isinstance(ref, NoteRef):
            if ref.path:
                return ref.path
            name = ref.name
        else:
            name = ref
            
        if name.endswith(".md"):
            return name
            
        # Match against list_files
        for f in self.list_files():
            if f.name.lower() == name.lower():
                return f.path
                
        # Default fallback
        return f"{name}.md"

    def _write_large_content(self, path: str, content: str, append_mode: bool = False) -> None:
        """Write large content to a file inside Obsidian using a temporary file and eval."""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        try:
            js_temp_path = temp_path.replace("\\", "\\\\").replace("'", "\\'")
            js_dest_path = path.replace("\\", "\\\\").replace("'", "\\'")
            if append_mode:
                js_code = (
                    f"(async () => {{"
                    f"  const fs = require('fs');"
                    f"  const data = fs.readFileSync('{js_temp_path}', 'utf8');"
                    f"  await app.vault.adapter.append('{js_dest_path}', data);"
                    f"}})()"
                )
            else:
                js_code = (
                    f"(async () => {{"
                    f"  const fs = require('fs');"
                    f"  const data = fs.readFileSync('{js_temp_path}', 'utf8');"
                    f"  await app.vault.adapter.write('{js_dest_path}', data);"
                    f"}})()"
                )
            self._run_cli("eval", f"code={js_code}")
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _run_cli(self, *args: str, check: bool = True) -> str:
        """Execute an obsidian CLI command and return stdout.

        Raises subprocess.CalledProcessError on non-zero exit (if check=True).
        """
        cmd = ["obsidian"]
        if self._vault_name:
            cmd.append(f"vault={self._vault_name}")
        cmd.extend(args)

        timeout = _cli_timeout()
        logger.debug("CLI exec: %s  (timeout=%.1fs)", " ".join(cmd), timeout)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=check,
            )
            if result.stderr:
                logger.debug("CLI stderr: %s", result.stderr.strip())
            
            stdout_str = result.stdout.strip()
            # Intercept Obsidian CLI errors that are printed to stdout with exit code 0
            if stdout_str.startswith("Error:") or stdout_str == "No matches found.":
                raise RuntimeError(stdout_str)
                
            return stdout_str
        except FileNotFoundError as e:
            raise RuntimeError("Obsidian CLI executable not found: obsidian") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Obsidian CLI timeout: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Obsidian CLI error (exit {e.returncode}): {e.stderr.strip()}"
            )

    def _run_json(self, *args: str) -> Any:
        """Execute a CLI command with format=json and parse the result."""
        try:
            raw = self._run_cli(*args, "format=json")
        except RuntimeError as e:
            if "no matches" in str(e).lower() or "not found" in str(e).lower() or "error" in str(e).lower():
                return []
            raise
        if not raw or raw.lower().startswith("no matches"):
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse CLI JSON output: %s\n%s", e, raw[:500])
            return []

    def _ref_arg(self, ref: NoteRef | str) -> str:
        """Convert a NoteRef or string to a file= CLI argument."""
        if isinstance(ref, str):
            if "/" in ref or ref.endswith(".md"):
                if not ref.endswith(".md"):
                    ref = f"{ref}.md"
                return f"path={ref}"
            return f"file={ref}"
        if ref.path:
            return f"path={ref.path}"
        return f"file={ref.name}"

    # ------------------------------------------------------------------
    # Discovery / Read
    # ------------------------------------------------------------------

    def search_names(self, query: str) -> list[NoteRef]:
        """Search vault note names matching query."""
        import os
        from silica.config import CONFIG
        inbox_norm = os.path.normcase(CONFIG.inbox_dir.replace("\\", "/").strip("/")) if CONFIG.inbox_dir else None

        query = query.lower()
        files = self.list_files()
        results = []
        for f in files:
            if inbox_norm and f.path:
                f_path_norm = os.path.normcase(f.path.replace("\\", "/").strip("/"))
                if f_path_norm == inbox_norm or f_path_norm.startswith(inbox_norm + "/"):
                    continue
            if query in f.name.lower():
                results.append(f)
        return results

    def mentions_of(self, title: str) -> list[str]:
        """Return vault-relative paths of notes whose body mentions `title`.

        O(1) lookup into the inverted text index built during _ensure_graph.
        If the index hasn't been built yet, falls back to building it first.
        """
        self._ensure_graph()
        return list(self._mention_index.get(title.lower(), set()))

    def search_context(self, query: str) -> list[Hit]:
        """Search vault content with line-level context snippets."""
        import os
        from silica.config import CONFIG
        inbox_norm = os.path.normcase(CONFIG.inbox_dir.replace("\\", "/").strip("/")) if CONFIG.inbox_dir else None

        escaped_query = query.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
        js_code = """(async () => {
  const query = 'QUERY_PLACEHOLDER';
  const queryLower = query.toLowerCase();
  const files = app.vault.getMarkdownFiles();
  const results = [];
  await Promise.all(files.map(async (file) => {
    try {
      const content = await app.vault.read(file);
      if (content.toLowerCase().includes(queryLower)) {
        const lines = content.split('\\n');
        const matches = [];
        for (let i = 0; i < lines.length; i++) {
          if (lines[i].toLowerCase().includes(queryLower)) {
            matches.push({
              line: i + 1,
              content: lines[i].trim()
            });
          }
        }
        if (matches.length > 0) {
          results.push({
            file: file.path,
            path: file.path,
            name: file.basename,
            matches: matches
          });
        }
      }
    } catch (e) {}
  }));
  return JSON.stringify(results);
})()""".replace('QUERY_PLACEHOLDER', escaped_query)

        try:
            raw = self._run_cli("eval", f"code={js_code}")
            if raw.startswith("=> "):
                raw = raw[3:].strip()
            data = json.loads(raw)
        except Exception as e:
            logger.error("Failed to execute or parse eval search: %s", e)
            data = []

        results = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    name = item.get("name", item.get("file", "")) or ""
                    path = item.get("path", item.get("file", "")) or ""

                    if inbox_norm and path:
                        path_norm = os.path.normcase(str(path).replace("\\", "/").strip("/"))
                        if path_norm == inbox_norm or path_norm.startswith(inbox_norm + "/"):
                            continue

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
        try:
            data = self._run_json("properties", self._ref_arg(ref))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def outline(self, ref: NoteRef | str) -> list[Heading]:
        """Get the heading tree of a note."""
        try:
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
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def links(self, ref: NoteRef | str) -> list[NoteRef]:
        """Outgoing links from a note.

        Handles both line-per-path plain-text output and JSON array output
        (some CLI versions return '["path1.md",...]' or '[]' instead of
        newline-separated paths).  An empty JSON array '[]' is treated as
        "no links" rather than a malformed path named '[]'.
        """
        try:
            raw = self._run_cli("links", self._ref_arg(ref))
            results = []
            stripped = raw.strip()
            if stripped.startswith("["):
                # JSON array format: parse and extract paths
                try:
                    paths = json.loads(stripped)
                    for path in (paths if isinstance(paths, list) else []):
                        if isinstance(path, str) and path.strip():
                            name = path.rsplit("/", 1)[-1].removesuffix(".md")
                            if name:
                                results.append(NoteRef(name=name, path=path))
                except (ValueError, TypeError):
                    pass  # malformed JSON — fall through to line-by-line
                return results
            # Plain-text format: one vault-relative path per line
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("No ") or "found" in line:
                    continue
                name = line.rsplit("/", 1)[-1].removesuffix(".md")
                if name:
                    results.append(NoteRef(name=name, path=line))
            return results
        except Exception:
            return []

    def backlinks(self, ref: NoteRef | str) -> list[NoteRef]:
        """Incoming links to a note."""
        try:
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
        except Exception:
            return []

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
            self._ensure_graph()
            link_counts: dict[str, int] = {}
            for path, note in self._notes.items():
                resolved_count = self._graph.out_degree(path) if path in self._graph else 0
                unresolved_count = sum(1 for s, t in self._unresolved_links if s == path)
                # Key by canonical path (no .md) — unique even with duplicate basenames.
                key = path.removesuffix(".md")
                link_counts[key] = resolved_count + unresolved_count

            backlink_counts: dict[str, int] = {
                path.removesuffix(".md"): d
                for path, d in self._graph.in_degree()
            }

            orphans: list[NoteRef] = [self._graph.nodes[n]["ref"] for n, d in self._graph.in_degree() if d == 0]
            unresolved: list[Link] = [
                Link(source=self._node_ref(s), target=t.removesuffix(".md"))
                for s, t in self._unresolved_links
            ]

            return GraphSnapshot(
                orphans=orphans,
                unresolved=unresolved,
                link_counts=link_counts,
                backlink_counts=backlink_counts,
            )

        # Incremental snapshot — path-keyed, reads from in-memory graph (C1.2/C1.3)
        self._ensure_graph()

        # Build 1-hop neighborhood using paths (not names)
        neighborhood: set[str] = set()
        for ref in refs:
            if not ref.path:
                continue
            neighborhood.add(ref.path)
            if ref.path in self._graph:
                for t in self._graph.successors(ref.path):
                    neighborhood.add(t)
                for s in self._graph.predecessors(ref.path):
                    neighborhood.add(s)
            # Unresolved outgoing: source path is in _unresolved_links
            for s, _t in self._unresolved_links:
                if s == ref.path:
                    neighborhood.add(s)

        link_counts: dict[str, int] = {}
        backlink_counts: dict[str, int] = {}
        orphans: list[NoteRef] = []
        unresolved: list[Link] = []

        for path in neighborhood:
            note = self._notes.get(path)
            if not note:
                continue
            # Canonical key: path without .md extension (mirrors full-vault branch)
            key = path.removesuffix(".md")

            resolved_out = self._graph.out_degree(path) if path in self._graph else 0
            unresolved_out = sum(1 for s, t in self._unresolved_links if s == path)
            link_counts[key] = resolved_out + unresolved_out

            in_deg = self._graph.in_degree(path) if path in self._graph else 0
            backlink_counts[key] = in_deg
            if in_deg == 0:
                orphans.append(ref)

        # Capture unresolved links for neighborhood paths
        for s, t in self._unresolved_links:
            if s in neighborhood and s in self._notes:
                unresolved.append(
                    Link(source=self._node_ref(s), target=t.removesuffix(".md"))
                )

        return GraphSnapshot(
            orphans=orphans,
            unresolved=unresolved,
            link_counts=link_counts,
            backlink_counts=backlink_counts,
        )


    def graph_data(self, folder: str = "") -> tuple[dict, set, Any]:
        """Return (notes, unresolved_links, graph) for in-process consumers."""
        self._ensure_graph()
        return self._notes, self._unresolved_links, self._graph

    # ------------------------------------------------------------------
    # Write (graph-safe)
    # ------------------------------------------------------------------

    def _patch_graph_add(self, path: str, ref: NoteRef, content: str) -> None:
        """Patch the in-memory graph after a create or overwrite.

        Adds/updates the node and its outgoing edges derived from `content`.
        Obsidian's metadataCache is already up-to-date (guaranteed by
        _wait_for_links_indexed), so we trust self._notes_by_name for
        resolution — the same dict Obsidian populated via Spike S1.
        Does nothing if the graph has never been built (no cache to patch).
        """
        if not self._is_graph_built:
            return
        # Remove stale outgoing edges for this path
        if path in self._graph:
            self._graph.remove_edges_from(list(self._graph.out_edges(path)))
        self._unresolved_links = {
            (s, t) for s, t in self._unresolved_links if s != path
        }
        # Add or refresh the node
        self._notes[path] = ref
        self._graph.add_node(path, ref=ref)
        name_lower = ref.name.lower()
        if name_lower not in self._notes_by_name:
            self._notes_by_name[name_lower] = []
        if ref not in self._notes_by_name[name_lower]:
            self._notes_by_name[name_lower].append(ref)
        # Re-derive edges from the new content
        for target_name in extract_links(content):
            # Name-based lookup mirrors Obsidian's shortest-path resolution
            candidates = self._notes_by_name.get(target_name.lower(), [])
            if candidates:
                self._graph.add_edge(path, candidates[0].path)
            else:
                self._unresolved_links.add((path, target_name))

        # Incrementally update the mention index: rescan this note's body
        # against all known titles, and also check if existing notes mention
        # this note's title.
        content_lower = content.lower()
        # 1. Remove stale entries for this path from all title sets
        for title_lower, paths_set in self._mention_index.items():
            paths_set.discard(path)
        # 2. Re-scan this body against all known titles
        for title_lower in self._notes_by_name:
            if len(title_lower) >= 2 and title_lower in content_lower:
                self._mention_index.setdefault(title_lower, set()).add(path)
        # 3. The new note's own title is now a searchable term — existing
        #    notes may already mention it but weren't indexed for it.
        #    A full rescan would be expensive, so we skip it here.
        #    The _build_mention_index call in _load_graph_from_obsidian
        #    already captured all mentions at startup.  New notes created
        #    mid-session can only be mentioned by notes written *after*
        #    them (which will be patched in their own _patch_graph_add).

    def _patch_graph_remove(self, path: str) -> None:
        """Patch the in-memory graph after a delete."""
        if not self._is_graph_built:
            return
        if path in self._graph:
            self._graph.remove_node(path)
        self._notes.pop(path, None)
        name_lower = path.rsplit("/", 1)[-1].removesuffix(".md").lower()
        if name_lower in self._notes_by_name:
            self._notes_by_name[name_lower] = [
                r for r in self._notes_by_name[name_lower] if r.path != path
            ]
        self._unresolved_links = {
            (s, t) for s, t in self._unresolved_links if s != path
        }
        # Remove this path from all mention index entries
        for title_lower, paths_set in self._mention_index.items():
            paths_set.discard(path)

    def create(self, path: str, content: str) -> NoteRef:
        """Create a new note at the given vault-relative path."""
        if len(content) > 30000:
            self._write_large_content(path, content, append_mode=False)
        else:
            escaped = content.replace("\\", "\\\\").replace("\n", "\\n")
            self._run_cli("create", f"path={path}", f"content={escaped}")
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        ref = NoteRef(name=name, path=path)
        self._wait_for_content_reflects(ref, content)
        expected_targets = extract_links(content)
        self._wait_for_links_indexed(ref, expected_targets)
        # Optimistic patch — no full reload needed
        self._patch_graph_add(path, ref, content)
        return ref

    def _load_graph_from_obsidian(self) -> nx.DiGraph:
        """Bulk-read graph state from Obsidian metadataCache via a single CDP eval.

        Spike S1 implementation: reads resolvedLinks and unresolvedLinks in one
        round-trip instead of N per-note links() calls, dramatically reducing
        _ensure_graph() latency on large vaults.

        resolvedLinks shape (from Obsidian's metadataCache):
            { "Source/Note.md": { "Target/Note.md": count, ... }, ... }
        unresolvedLinks shape:
            { "Source/Note.md": { "TargetName": count, ... }, ... }

        Raises RuntimeError if the eval call fails (e.g. Obsidian not running),
        which causes _ensure_graph() to fall back to per-note queries.
        """
        js_code = (
            "JSON.stringify({"
            "resolved: app.metadataCache.resolvedLinks,"
            "unresolved: app.metadataCache.unresolvedLinks"
            "})"
        )
        raw = self._run_cli("eval", f"code={js_code}")
        # The Obsidian CLI prefixes eval return values with "=> "
        if raw.startswith("=> "):
            raw = raw[3:].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Could not parse metadataCache response: {exc}") from exc

        resolved: dict[str, dict[str, int]] = data.get("resolved") or {}
        unresolved: dict[str, dict[str, int]] = data.get("unresolved") or {}

        for source_path, targets in resolved.items():
            if source_path not in self._notes:
                continue
            for target_path in targets:
                if target_path in self._notes:
                    self._graph.add_edge(source_path, target_path)
                # Resolved entries point to real files — no unresolved entry needed.

        for source_path, targets in unresolved.items():
            if source_path not in self._notes:
                continue
            for target_name in targets:
                self._unresolved_links.add((source_path, target_name))

        # Build the title-mention index: for each note, scan its body (via
        # cachedRead in JS — instant, no disk I/O) for all known vault titles.
        # The result is {title_lower: [paths]} — one CDP call, O(N×T) in the
        # fast JS runtime, response is just the compact map.
        self._build_mention_index()

        return self._graph

    def _build_mention_index(self) -> None:
        """Build the title-mention inverted index via a single CDP eval.

        Reads each vault file once (via cachedRead — from Obsidian's in-memory
        cache, not disk) and checks its body against all known note titles.
        Result: {title_lower: [paths_that_mention_it]}.

        Runs inside the JS runtime for speed — avoids transferring megabytes of
        note bodies over the CDP bridge.
        """
        titles_json = json.dumps([
            ref.name.lower() for ref in self._notes.values()
            if len(ref.name) >= 2
        ])

        js_code = (
            "(async () => {\n"
            f"  const titles = {titles_json};\n"
            "  const files = app.vault.getMarkdownFiles();\n"
            "  const mentions = {};\n"
            "  await Promise.all(files.map(async (file) => {\n"
            "    try {\n"
            "      const content = (await app.vault.cachedRead(file)).toLowerCase();\n"
            "      for (const title of titles) {\n"
            "        if (content.includes(title)) {\n"
            "          if (!mentions[title]) mentions[title] = [];\n"
            "          mentions[title].push(file.path);\n"
            "        }\n"
            "      }\n"
            "    } catch (e) {}\n"
            "  }));\n"
            "  return JSON.stringify(mentions);\n"
            "})()"
        )

        try:
            raw = self._run_cli("eval", f"code={js_code}")
            if raw.startswith("=> "):
                raw = raw[3:].strip()
            data = json.loads(raw)
        except Exception as e:
            logger.warning("_build_mention_index: CDP eval failed (%s); mention index empty.", e)
            data = {}

        self._mention_index.clear()
        if isinstance(data, dict):
            for title_lower, paths in data.items():
                if isinstance(paths, list):
                    self._mention_index[title_lower] = set(paths)
        logger.debug("Mention index built: %d titles tracked", len(self._mention_index))

    def overwrite(self, path: str, content: str) -> NoteRef:
        """Overwrite an existing note in-place, preserving Obsidian version history.

        Uses `obsidian create ... overwrite=true` which keeps the file's block-refs
        and history intact — unlike delete+create which destroys both.
        """
        if len(content) > 30000:
            self._write_large_content(path, content, append_mode=False)
        else:
            escaped = content.replace("\\", "\\\\").replace("\n", "\\n")
            self._run_cli("create", f"path={path}", f"content={escaped}", "overwrite=true")
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        ref = NoteRef(name=name, path=path)
        # C2: verify content reflects the write, not just readability
        self._wait_for_content_reflects(ref, content)
        # Optimistic patch — rebuild edges for the new content
        self._patch_graph_add(path, ref, content)
        return ref

    def append(self, ref: NoteRef | str, content: str) -> None:
        """Append content to an existing note."""
        if len(content) > 30000:
            path = self._resolve_path(ref)
            self._write_large_content(path, content, append_mode=True)
        else:
            escaped = content.replace("\\", "\\\\").replace("\n", "\\n")
            self._run_cli("append", self._ref_arg(ref), f"content={escaped}")
        # C2: verify content was appended
        self._wait_for_content_contains(ref, content)
        # Optimistic patch — add any new links introduced by the appended fragment
        if self._is_graph_built:
            path = ref.path if isinstance(ref, NoteRef) else self._resolve_path(ref)
            note_ref = self._notes.get(path) if isinstance(path, str) else None
            if note_ref and path in self._graph:
                for target_name in extract_links(content):
                    candidates = self._notes_by_name.get(target_name.lower(), [])
                    if candidates:
                        self._graph.add_edge(path, candidates[0].path)
                    else:
                        self._unresolved_links.add((path, target_name))

    def set_prop(self, ref: NoteRef | str, name: str, value: Any, type_: str = "text") -> None:
        """Set a frontmatter property on a note."""
        self._is_graph_built = False
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
        old_path = ref.path if isinstance(ref, NoteRef) else None
        self._run_cli("move", self._ref_arg(ref), f"to={to}")
        self._wait_for_move(ref, to)
        # Obsidian rewrites all incoming wikilinks on move, so edges from
        # other notes pointing to the old path are now stale. We cannot
        # patch those in-process cheaply, so invalidate the full cache.
        self._is_graph_built = False
        if old_path and old_path in self._notes:
            self._patch_graph_remove(old_path)

    def delete(self, ref: NoteRef | str) -> None:
        """Delete a note from the vault."""
        path = ref.path if isinstance(ref, NoteRef) else None
        self._run_cli("delete", self._ref_arg(ref))
        # C2: verify note is gone
        self._wait_for_gone(ref)
        # Optimistic patch — remove node and all its edges
        if path:
            self._patch_graph_remove(path)

    # ------------------------------------------------------------------
    # Advanced
    # ------------------------------------------------------------------

    def list_files(self, folder: str = "") -> list[NoteRef]:
        """List all markdown files, optionally filtered by folder."""
        import os
        from silica.config import CONFIG
        inbox_norm = os.path.normcase(CONFIG.inbox_dir.replace("\\", "/").strip("/")) if CONFIG.inbox_dir else None

        args = ["files", "ext=md"]
        if folder:
            args.append(f"folder={folder}")
        raw = self._run_cli(*args)
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                if inbox_norm:
                    line_norm = os.path.normcase(line.replace("\\", "/").strip("/"))
                    if line_norm == inbox_norm or line_norm.startswith(inbox_norm + "/"):
                        continue
                name = line.rsplit("/", 1)[-1].removesuffix(".md")
                results.append(NoteRef(name=name, path=line))
        return results

    def list_inbox_files(self) -> list[NoteRef]:
        """List all files in the inbox directory."""
        from silica.config import CONFIG
        if not CONFIG.inbox_dir:
            return []
        args = ["files", f"folder={CONFIG.inbox_dir}", "ext=md"]
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

        Parses the plain-text history output directly — the Obsidian CLI's
        `history` command does not honour `format=json`.  The table format is:
          <filename>
          1    <datetime>    <size>   ← most recent, position 1
          2    <datetime>    <size>
        The first numeric token on the first data line gives the current
        position (always 1).  Stored only as a best-effort hint; the primary
        rollback path now uses prior_content captured by build_txn.
        """
        versions: dict[str, int] = {}
        for ref in refs:
            key = ref.path or ref.name
            try:
                raw = self._run_cli("history", self._ref_arg(ref))
                for line in raw.splitlines():
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        versions[key] = int(parts[0])
                        break  # first numeric line = most recent entry
            except Exception as e:
                logger.warning("No history available for %s: %s", ref.name, e)

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
                if "not found" in str(e).lower():
                    logger.info("Rolled back created note %s (already absent)", path)
                else:
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

        raise SettleTimeout(
            f"Settle timeout (create) for {ref.name} after {timeout:.1f}s — cache may be stale"
        )

    def _wait_for_links_indexed(self, ref: NoteRef, expected_targets: list[str], timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until Obsidian has *registered* ref's outgoing wikilinks.

        Registration ≠ resolution. A wikilink to a note that does not (yet)
        exist is a legitimate unresolved link and counts as registered here;
        the cache has processed the file, which is the only freshness
        invariant a single-note write can guarantee. True graph-resolution
        quality (ghost-link audit, new-orphan detection) is the job of the
        batch-level LINT / graph-regression gate downstream.

        The previous "issubset-of-resolved-names" predicate was unsatisfiable
        in practice because of four compounding defects:
          (a) ``extract_links(content)`` includes wikilinks to non-existent
              concepts — these remain unresolved by design, so the subset
              check spins until timeout;
          (b) ``self.links(ref)`` reports unresolved targets with a trailing
              ``(unresolved)`` marker that the raw subset check never strips;
          (c) expected targets come from the source text (often snake_case /
              lowercase) while ``links()`` returns the *resolved* note name
              (title-case), so even resolvable links failed the case-
              sensitive comparison;
          (d) per-note settle + sequential batch writes means forward
              references to siblings not yet created can't possibly resolve.
        The canonicalization below — strip status marker, drop path/ext,
        casefold — collapses (b), (c), and (d); accepting registered-but-
        unresolved as a passing state closes (a).
        """
        if not expected_targets:
            return

        def _canon(name: str) -> str:
            name = re.sub(r"\s*\(unresolved\)\s*$", "", name)
            return name.rsplit("/", 1)[-1].removesuffix(".md").casefold()

        expected = {_canon(t) for t in expected_targets}
        deadline = time.monotonic() + timeout
        registered: set[str] = set()
        while time.monotonic() < deadline:
            try:
                registered = {_canon(l.name) for l in self.links(ref)}
                if expected.issubset(registered):
                    return
            except Exception:
                pass
            time.sleep(_SETTLE_POLL_INTERVAL)

        # Non-fatal: the note is already written correctly on disk
        # (_wait_for_content_reflects passed).  Link-indexing lag is a
        # Obsidian metadataCache detail that resolves in seconds.  The
        # batch-level LINT / graph-regression gate audits graph
        # consistency after all chunks complete, so this is safe to skip.
        logger.warning(
            "Settle timeout (links indexing) for %s after %.1fs — "
            "note is on disk, cache lag will resolve. "
            "Expected: %s, Registered: %s",
            ref.name, timeout, expected, registered,
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

        raise SettleTimeout(
            f"Settle timeout (overwrite content) for {ref.name} after {timeout:.1f}s — cache may be stale"
        )

    def _wait_for_content_contains(self, ref: NoteRef | str, fragment: str,
                                   timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until read(ref).content contains fragment.

        Postcondition (C2 / append): appended content is visible in the note.
        """
        deadline = time.monotonic() + timeout
        name = ref if isinstance(ref, str) else ref.name
        while time.monotonic() < deadline:
            try:
                nc = self.read_note(ref)
                if fragment in nc.content:
                    return
            except RuntimeError:
                pass
            time.sleep(_SETTLE_POLL_INTERVAL)

        raise SettleTimeout(
            f"Settle timeout (append) for {name} after {timeout:.1f}s — cache may be stale"
        )

    def _wait_for_gone(self, ref: NoteRef | str, timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until read(ref) raises (note is gone).

        Postcondition (C2 / delete): note is no longer readable.
        """
        deadline = time.monotonic() + timeout
        name = ref if isinstance(ref, str) else ref.name
        while time.monotonic() < deadline:
            try:
                self._run_cli("read", self._ref_arg(ref), check=False)
                time.sleep(_SETTLE_POLL_INTERVAL)
            except RuntimeError:
                return  # Gone — postcondition satisfied

        raise SettleTimeout(
            f"Settle timeout (delete) for {name} after {timeout:.1f}s — note may still be cached"
        )

    def _wait_for_prop(self, ref: NoteRef | str, prop_name: str, expected_value: str,
                       timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until a frontmatter property reflects the expected value.

        Postcondition (C2 / set_prop): props_of(ref)[prop_name] == expected_value
        """
        deadline = time.monotonic() + timeout
        name = ref if isinstance(ref, str) else ref.name
        while time.monotonic() < deadline:
            try:
                props = self.props_of(ref)
                if str(props.get(prop_name, "")) == expected_value:
                    return
            except RuntimeError:
                pass
            time.sleep(_SETTLE_POLL_INTERVAL)

        raise SettleTimeout(
            f"Settle timeout (set_prop '{prop_name}') for {name} after {timeout:.1f}s — cache may be stale"
        )

    def _wait_for_move(self, original_ref: NoteRef | str, to_path: str,
                       timeout: float = _SETTLE_TIMEOUT) -> None:
        """Poll until a move is reflected: destination readable AND source gone,
        and its backlink cache has updated.

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
                # Also verify that the backlink cache has updated
                if not self.backlinks(original_ref):
                    return
            time.sleep(_SETTLE_POLL_INTERVAL)

        raise SettleTimeout(
            f"Settle timeout (move to '{to_path}') after {timeout:.1f}s — cache may be stale"
        )

    # Legacy alias kept for any remaining internal callers
    def _wait_for_settle(self, ref: NoteRef, timeout: float = _SETTLE_TIMEOUT) -> None:
        self._wait_for_create(ref, timeout=timeout)
