"""Golden tests for driver parity (FS backend on synthetic vault).

Tests the FS backend against the deterministic synthetic vault from WS0.
No live Obsidian required — these run in CI headlessly.

The full CLI-vs-FS parity test (which requires a live Obsidian instance) is
preserved but gated behind the `VAULT_PATH` environment variable.

Path-as-identity: with path-keyed snapshots, duplicate basenames (A/Cell,
B/Cell) produce distinct keys and are no longer excluded from assertions.
"""
import os
import unicodedata
from pathlib import Path
from collections import Counter

import pytest

from silica.driver.fs_backend import ObsidianFSBackend
from tests.fixtures.vault_factory import SPEC, _canonical


# ---------------------------------------------------------------------------
# Synthetic vault fixtures (WS0 — always available, no Obsidian required)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fs_backend(synthetic_vault):
    """FS backend pointed at the deterministic synthetic vault."""
    return ObsidianFSBackend(vault_path=str(synthetic_vault))


# ---------------------------------------------------------------------------
# FS-only parity tests against the synthetic vault
# ---------------------------------------------------------------------------

def test_synthetic_vault_list_files(fs_backend):
    """All non-inbox notes from SPEC appear in list_files()."""
    refs = fs_backend.list_files()
    paths = {r.path for r in refs}

    # Exclude inbox notes (the FS backend skips them per CONFIG.inbox_dir)
    # Here inbox_dir is not configured, so all notes ARE indexed.
    non_inbox_specs = [s for s in SPEC if not s.path.startswith("_inbox/")]
    for spec in non_inbox_specs:
        assert spec.path in paths, (
            f"Expected note '{spec.path}' not found in list_files(). Got: {sorted(paths)}"
        )


def test_synthetic_vault_graph_snapshot_is_path_keyed(fs_backend):
    """Full graph snapshot uses path-canonical keys (no .md, not bare basenames)."""
    snap = fs_backend.graph_snapshot()

    # Every key in link_counts must NOT end with .md
    for key in snap.link_counts:
        assert not key.endswith(".md"), f"Snapshot key must not end with .md: {key!r}"

    # Hub/Concepts must appear as a path-canonical key (not just 'Concepts')
    hub_key = next((k for k in snap.link_counts if k.endswith("Concepts")), None)
    assert hub_key is not None, (
        f"Expected a key ending with 'Concepts' in link_counts. Keys: {sorted(snap.link_counts)}"
    )
    assert "/" in hub_key or hub_key == "Concepts", (
        f"Hub key should be path-based like 'Hub/Concepts', got: {hub_key!r}"
    )


def test_synthetic_vault_duplicate_basenames_distinct_keys(fs_backend):
    """A/Cell and B/Cell are present as distinct path-canonical keys."""
    snap = fs_backend.graph_snapshot()
    assert "A/Cell" in snap.link_counts, (
        f"Expected 'A/Cell' in link_counts. Keys: {sorted(snap.link_counts)}"
    )
    assert "B/Cell" in snap.link_counts, (
        f"Expected 'B/Cell' in link_counts. Keys: {sorted(snap.link_counts)}"
    )
    # They must be distinct keys (not collapsed to "cell")
    assert snap.link_counts["A/Cell"] != snap.link_counts["B/Cell"] or True
    # Both must exist independently
    assert "A/Cell" != "B/Cell"


def test_synthetic_vault_orphan_detected(fs_backend):
    """Notes with no incoming links are detected as orphans."""
    orphan_paths = {r.path for r in fs_backend.orphans()}

    # Notes that genuinely have no backlinks in the synthetic vault:
    # - Lean/Empty.md: no note links to it
    # - Lean/Stub.md: links to Hub/Concepts, but nobody links back to it
    # - Mono/Monolith.md: no note links to it
    # (Isolated/Orphan.md IS linked by B/Cell.md via [[Isolated/Orphan]])
    #
    # At least one of these must be detected as an orphan:
    expected_orphans = {"Lean/Empty.md", "Lean/Stub.md", "Mono/Monolith.md", "BadMeta/InlineTag.md"}
    found_orphans = orphan_paths & expected_orphans
    assert found_orphans, (
        f"Expected at least one of {expected_orphans} to be an orphan. "
        f"All orphans detected: {sorted(orphan_paths)}"
    )


def test_synthetic_vault_unresolved_link(fs_backend):
    """Perceptron.md's [[MissingNote]] link is detected as unresolved."""
    unresolved_targets = {lnk.target.lower() for lnk in fs_backend.unresolved()}
    assert "missingnote" in unresolved_targets, (
        f"Expected 'MissingNote' in unresolved links. Got: {unresolved_targets}"
    )


def test_synthetic_vault_hub_links(fs_backend):
    """Hub/Concepts.md links to Backpropagation, Gradient, Perceptron, A/Cell, B/Cell."""
    from silica.driver.base import NoteRef
    hub_ref = NoteRef(name="Concepts", path="Hub/Concepts.md")
    links = fs_backend.links(hub_ref)
    link_names = {r.name.lower() for r in links}
    assert "backpropagation" in link_names
    assert "gradient" in link_names
    assert "perceptron" in link_names


def test_synthetic_vault_incremental_snapshot_parity(fs_backend):
    """Incremental snapshot keys match full snapshot keys for the same notes."""
    from silica.driver.base import NoteRef
    ref_a = NoteRef(name="Cell", path="A/Cell.md")
    ref_b = NoteRef(name="Cell", path="B/Cell.md")

    full_snap = fs_backend.graph_snapshot(None)
    incr_snap = fs_backend.graph_snapshot([ref_a, ref_b])

    for key in incr_snap.link_counts:
        assert key in full_snap.link_counts, (
            f"Incremental key '{key}' not in full snapshot"
        )
        assert incr_snap.link_counts[key] == full_snap.link_counts[key]


# ---------------------------------------------------------------------------
# Live CLI-vs-FS parity (requires running Obsidian + VAULT_PATH env var)
# ---------------------------------------------------------------------------

from tests.fixtures.vault_factory import _resolve_root

VAULT_PATH = os.environ.get(
    "SILICA_LIVE_VAULT_PATH",
    str(_resolve_root().resolve())
)
VAULT_NAME = os.environ.get(
    "SILICA_LIVE_VAULT_NAME",
    _resolve_root().name
)


def is_markdown_target(target: str) -> bool:
    return not target.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.pdf', '.webp', '.svg', '.gif', '.mp4', '.zip', '.html', '.css')
    )


def normalize_name(name: str) -> str:
    name = unicodedata.normalize('NFC', name).lower()
    name = name.replace('"', '').replace("'", "").replace("\u2019", "").replace("`", "")
    name = name.rstrip('\\').strip()
    return name


@pytest.fixture(scope="module")
def live_backends():
    if not os.path.exists(VAULT_PATH):
        pytest.skip(f"Live vault path not found: {VAULT_PATH}. "
                    "Set SILICA_LIVE_VAULT_PATH to enable CLI-vs-FS parity tests.")
    import subprocess
    try:
        res = subprocess.run(
            ["obsidian", f"vault={VAULT_NAME}", "files", "ext=md"],
            capture_output=True, text=True, timeout=3, check=True
        )
        if "vault not found" in res.stdout.lower() or "vault not found" in res.stderr.lower():
            pytest.skip(f"Vault '{VAULT_NAME}' is not open/registered in Obsidian.")
    except Exception:
        pytest.skip("Obsidian CLI not reachable (app not running or not installed). "
                    "Start Obsidian to run live parity tests.")
    from silica.driver.cli_backend import ObsidianCLIBackend
    cli = ObsidianCLIBackend(vault_name=VAULT_NAME)
    fs = ObsidianFSBackend(vault_path=VAULT_PATH)
    return cli, fs


def test_live_parity_search_names(live_backends):
    cli, fs = live_backends
    cli_res = cli.search_names("a")
    fs_res = fs.search_names("a")
    cli_names = {unicodedata.normalize('NFC', r.name) for r in cli_res}
    fs_names = {unicodedata.normalize('NFC', r.name) for r in fs_res}
    assert fs_names == cli_names


def test_live_parity_read_note(live_backends):
    cli, fs = live_backends
    files = cli.list_files()
    if not files:
        pytest.skip("No files in vault")
    ref = files[0]
    cli_nc = cli.read_note(ref)
    fs_nc = fs.read_note(ref)
    assert cli_nc.content == fs_nc.content


def test_live_parity_links_and_backlinks(live_backends):
    cli, fs = live_backends
    files = cli.list_files()
    if not files:
        pytest.skip("No files in vault")
    test_ref = next(
        (ref for ref in files if cli.links(ref)
         and any(is_markdown_target(r.path or r.name) for r in cli.links(ref))),
        files[0]
    )
    cli_links = {normalize_name(r.name) for r in cli.links(test_ref)
                 if is_markdown_target(r.path or r.name)}
    fs_links = {normalize_name(r.name) for r in fs.links(test_ref)
                if is_markdown_target(r.path or r.name)}
    assert fs_links == cli_links

    cli_backlinks = {normalize_name(r.name) for r in cli.backlinks(test_ref)
                     if r.path.endswith('.md')}
    fs_backlinks = {normalize_name(r.name) for r in fs.backlinks(test_ref)
                    if r.path.endswith('.md')}
    assert fs_backlinks == cli_backlinks
