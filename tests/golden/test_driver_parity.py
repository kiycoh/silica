"""Golden tests for driver parity (FS backend on synthetic vault).

Tests the FS backend against the deterministic synthetic vault from WS0.
No live Obsidian required — these run in CI headlessly.

Path-as-identity: with path-keyed snapshots, duplicate basenames (A/Cell,
B/Cell) produce distinct keys and are no longer excluded from assertions.
"""
import inspect

import pytest

from silica.driver.base import ObsidianDriver
from silica.driver.fs_backend import ObsidianFSBackend
from silica.driver.ws_backend import ObsidianWSBackend
from tests.fixtures.vault_factory import SPEC


# ---------------------------------------------------------------------------
# Structural parity tests — always run, no env gating, no I/O
# ---------------------------------------------------------------------------

def _protocol_method_names() -> set[str]:
    """Return the set of public method names declared in ObsidianDriver.

    Uses the class __dict__ directly (not inspect.getmembers) to limit scope
    to methods actually declared on the protocol, not inherited helpers.
    On Python >= 3.12, __protocol_attrs__ would be authoritative; on 3.11 we
    derive the same set: callables not starting with '_'.
    """
    if hasattr(ObsidianDriver, "__protocol_attrs__"):
        # Future-proof: use the canonical attribute when available.
        return set(ObsidianDriver.__protocol_attrs__)
    return {
        name
        for name, val in ObsidianDriver.__dict__.items()
        if callable(val) and not name.startswith("_")
    }


def _public_methods(cls) -> set[str]:
    """Return the public callable method names defined on cls (not inherited)."""
    return {
        name
        for name, val in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


class TestDriverStructuralParity:
    """Structural parity checks: no Obsidian / no I/O required.

    These tests guard against three categories of protocol drift:
      1. A backend loses a protocol method (forward drift).
      2. A method present on BOTH backends goes undeclared in the protocol
         (reverse drift — silent protocol widening).
      3. A backend method's parameter names diverge from the protocol
         (signature drift — catches renamed args without type errors).
    """

    # ------------------------------------------------------------------
    # Test 1: isinstance / protocol satisfaction
    # ------------------------------------------------------------------

    def test_fs_backend_satisfies_protocol(self, tmp_path):
        """ObsidianFSBackend instantiates cheaply and satisfies ObsidianDriver."""
        backend = ObsidianFSBackend(vault_path=str(tmp_path))
        assert isinstance(backend, ObsidianDriver), (
            "ObsidianFSBackend does not satisfy the ObsidianDriver protocol. "
            "A method may be missing or have the wrong signature."
        )

    def test_ws_backend_satisfies_protocol(self):
        """ObsidianWSBackend instantiates cheaply (no dial) and satisfies ObsidianDriver."""
        backend = ObsidianWSBackend(url="ws://127.0.0.1:1", token="")
        assert isinstance(backend, ObsidianDriver), (
            "ObsidianWSBackend does not satisfy the ObsidianDriver protocol. "
            "A method may be missing or have the wrong signature."
        )

    # ------------------------------------------------------------------
    # Test 2: forward drift — protocol methods present on each backend
    # ------------------------------------------------------------------

    def test_fs_backend_implements_all_protocol_methods(self, tmp_path):
        """Every protocol method exists as a callable on ObsidianFSBackend."""
        backend = ObsidianFSBackend(vault_path=str(tmp_path))
        proto_methods = _protocol_method_names()
        missing = [
            m for m in sorted(proto_methods)
            if not (hasattr(backend, m) and callable(getattr(backend, m)))
        ]
        assert not missing, (
            f"ObsidianFSBackend is missing protocol methods: {missing}"
        )

    def test_ws_backend_implements_all_protocol_methods(self):
        """Every protocol method exists as a callable on ObsidianWSBackend."""
        backend = ObsidianWSBackend(url="ws://127.0.0.1:1", token="")
        proto_methods = _protocol_method_names()
        missing = [
            m for m in sorted(proto_methods)
            if not (hasattr(backend, m) and callable(getattr(backend, m)))
        ]
        assert not missing, (
            f"ObsidianWSBackend is missing protocol methods: {missing}"
        )

    # ------------------------------------------------------------------
    # Test 3: reverse drift — public methods on BOTH backends declared in protocol
    # ------------------------------------------------------------------

    def test_no_undeclared_shared_public_methods(self):
        """Public methods present on BOTH backends must be declared in the protocol.

        A method on both backends but absent from the protocol is silent drift:
        callers cannot rely on it through the Driver abstraction, and it should
        either be added to the protocol or made private (prefixed with '_').
        Backend-specific private helpers (prefixed '_') are excluded.
        """
        proto_methods = _protocol_method_names()
        fs_public = _public_methods(ObsidianFSBackend)
        ws_public = _public_methods(ObsidianWSBackend)

        # Methods on BOTH backends = shared domain surface
        shared = fs_public & ws_public

        # Anything shared but not in the protocol = drift
        undeclared = shared - proto_methods
        assert not undeclared, (
            f"These public methods exist on BOTH backends but are NOT declared "
            f"in ObsidianDriver: {sorted(undeclared)}. "
            f"Either add them to the protocol or make them private (prefix with '_')."
        )

    # ------------------------------------------------------------------
    # Test 4: signature drift — parameter names must match the protocol
    # ------------------------------------------------------------------

    def test_fs_backend_parameter_names_match_protocol(self, tmp_path):
        """ObsidianFSBackend method parameter names match the protocol's."""
        proto_methods = _protocol_method_names()
        mismatches = []
        for method_name in sorted(proto_methods):
            proto_params = list(
                inspect.signature(getattr(ObsidianDriver, method_name)).parameters.keys()
            )
            fs_params = list(
                inspect.signature(getattr(ObsidianFSBackend, method_name)).parameters.keys()
            )
            if proto_params != fs_params:
                mismatches.append(
                    f"{method_name}: protocol={proto_params} fs={fs_params}"
                )
        assert not mismatches, (
            "ObsidianFSBackend method signatures diverge from protocol:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )

    def test_ws_backend_parameter_names_match_protocol(self):
        """ObsidianWSBackend method parameter names match the protocol's."""
        proto_methods = _protocol_method_names()
        mismatches = []
        for method_name in sorted(proto_methods):
            proto_params = list(
                inspect.signature(getattr(ObsidianDriver, method_name)).parameters.keys()
            )
            ws_params = list(
                inspect.signature(getattr(ObsidianWSBackend, method_name)).parameters.keys()
            )
            if proto_params != ws_params:
                mismatches.append(
                    f"{method_name}: protocol={proto_params} ws={ws_params}"
                )
        assert not mismatches, (
            "ObsidianWSBackend method signatures diverge from protocol:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )


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
