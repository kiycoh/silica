import os
import pytest
from pathlib import Path
from silica.driver.fs_backend import ObsidianFSBackend
from silica.driver.base import NoteRef, Link

@pytest.fixture
def temp_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    
    # Create notes
    # A links to B and MissingNote
    (vault / "A.md").write_text("# A\n\n[[B]] and [[MissingNote]]", encoding="utf-8")
    # B links to C
    (vault / "B.md").write_text("# B\n\n[[C]]", encoding="utf-8")
    # C is a leaf note
    (vault / "C.md").write_text("# C\n\nNo links here", encoding="utf-8")
    # D is an orphan (no incoming/outgoing links)
    (vault / "D.md").write_text("# D\n\nOrphan note", encoding="utf-8")
    
    return vault

def test_fs_backend_graph_basic(temp_vault):
    backend = ObsidianFSBackend(str(temp_vault))
    
    # Assert orphans
    # D has no incoming links. A has no incoming links either!
    # A, D are orphans
    orphans = backend.orphans()
    orphan_names = {o.name for o in orphans}
    assert orphan_names == {"A", "D"}
    
    # Assert links (outgoing) from A
    a_links = backend.links("A.md")
    a_link_names = {l.name for l in a_links}
    assert "B" in a_link_names
    assert "MissingNote" in a_link_names
    
    # Assert links (outgoing) from B
    b_links = backend.links("B.md")
    assert [l.name for l in b_links] == ["C"]
    
    # Assert links (outgoing) from C
    c_links = backend.links("C.md")
    assert c_links == []
    
    # Assert backlinks (incoming) to B
    b_backlinks = backend.backlinks("B.md")
    assert [bl.name for bl in b_backlinks] == ["A"]
    
    # Assert backlinks (incoming) to C
    c_backlinks = backend.backlinks("C.md")
    assert [bl.name for bl in c_backlinks] == ["B"]
    
    # Assert unresolved links
    unres = backend.unresolved()
    assert len(unres) == 1
    assert unres[0].source.name == "A"
    assert unres[0].target == "MissingNote"
    
    # Assert full snapshot
    snap = backend.graph_snapshot()
    assert {o.name for o in snap.orphans} == {"A", "D"}
    assert len(snap.unresolved) == 1
    assert snap.unresolved[0].target == "MissingNote"
    assert snap.link_counts["A"] == 2
    assert snap.link_counts["B"] == 1
    assert snap.link_counts["C"] == 0
    assert snap.link_counts["D"] == 0
    assert snap.backlink_counts["A"] == 0
    assert snap.backlink_counts["B"] == 1
    assert snap.backlink_counts["C"] == 1
    assert snap.backlink_counts["D"] == 0

def test_fs_backend_graph_snapshot_incremental(temp_vault):
    backend = ObsidianFSBackend(str(temp_vault))
    backend._ensure_index()
    
    # Incremental snapshot for B.md
    b_ref = backend._notes["B.md"]
    snap = backend.graph_snapshot(refs=[b_ref])
    
    # B.md has links B->C and backlink A->B
    # Neighborhood should cover A, B, C (but not D)
    assert "A" in snap.link_counts
    assert "B" in snap.link_counts
    assert "C" in snap.link_counts
    assert "D" not in snap.link_counts
    
    # check backlink counts
    assert snap.backlink_counts["A"] == 0
    assert snap.backlink_counts["B"] == 1
    assert snap.backlink_counts["C"] == 1
    
    # check orphans in neighborhood
    assert {o.name for o in snap.orphans} == {"A"}


def test_fs_backend_duplicate_basename_snapshot(tmp_path):
    """Two notes with the same basename in different folders must produce separate
    entries in link_counts/backlink_counts, keyed by canonical path (not name).

    Before the Option-A fix, both would collapse onto the same dict key and
    one of them would silently be overwritten — producing wrong counts.
    """
    vault = tmp_path / "vault"
    (vault / "folder_a").mkdir(parents=True)
    (vault / "folder_b").mkdir(parents=True)

    # Two notes named "Note" in different folders
    (vault / "folder_a" / "Note.md").write_text("# Note A\n\n[[Hub]]", encoding="utf-8")
    (vault / "folder_b" / "Note.md").write_text("# Note B\n\nNo links here.", encoding="utf-8")
    # Hub is linked by folder_a/Note.md but NOT by folder_b/Note.md
    (vault / "Hub.md").write_text("# Hub\n\nNo links here.", encoding="utf-8")

    backend = ObsidianFSBackend(str(vault))
    snap = backend.graph_snapshot()

    # With path-keyed snapshot there must be two distinct "Note" entries
    note_keys = [k for k in snap.backlink_counts if k.lower().endswith("/note") or k.lower() == "note"]
    assert len(note_keys) == 2, (
        f"Expected 2 distinct Note entries (one per path), got {note_keys}. "
        "Duplicate basenames are collapsing onto a single key — Option-A fix not applied."
    )

    # Hub should have exactly 1 backlink (only folder_a/Note links to it)
    hub_key = next(k for k in snap.backlink_counts if k.lower().endswith("hub"))
    assert snap.backlink_counts[hub_key] == 1

    # folder_b/Note is an orphan (no incoming links), folder_a/Note is also an orphan
    orphan_paths = {o.path for o in snap.orphans}
    assert "folder_a/Note.md" in orphan_paths
    assert "folder_b/Note.md" in orphan_paths
