import pytest
from silica.driver.base import GraphSnapshot, NoteRef, Link
from silica.kernel.graph_diff import check_graph_regression

def test_graph_diff_happy_path():
    pre = GraphSnapshot(
        orphans=[NoteRef(name="Orphan1", path="notes/Orphan1.md")],
        unresolved=[Link(source=NoteRef(name="A", path="notes/A.md"), target="Missing")]
    )
    post = GraphSnapshot(
        orphans=[NoteRef(name="Orphan1", path="notes/Orphan1.md")],
        unresolved=[Link(source=NoteRef(name="A", path="notes/A.md"), target="Missing")]
    )
    
    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert success
    assert not errors


def test_graph_diff_planned_orphans_allowed():
    # If a newly created note is an orphan, it is allowed
    pre = GraphSnapshot(orphans=[], unresolved=[])
    post = GraphSnapshot(
        orphans=[
            NoteRef(name="NewNote", path="notes/NewNote.md")
        ],
        unresolved=[]
    )
    
    # NewNote was explicitly created
    success, errors = check_graph_regression(pre, post, created_paths=["notes/NewNote.md"])
    assert success
    assert not errors


def test_graph_diff_unplanned_orphans_rejected():
    # A note that WAS observed in the pre-snapshot (link_counts has an entry)
    # becomes a new orphan in the post-snapshot — its last incoming link was
    # removed by the write.  This IS a genuine regression.
    pre = GraphSnapshot(
        orphans=[],
        unresolved=[],
        link_counts={"notes/ExistingNote": 0},      # observed, 0 outgoing links
        backlink_counts={"notes/ExistingNote": 1},  # had 1 incoming link (not orphan)
    )
    post = GraphSnapshot(
        orphans=[
            NoteRef(name="ExistingNote", path="notes/ExistingNote.md")
        ],
        unresolved=[],
        link_counts={"notes/ExistingNote": 0},
        backlink_counts={"notes/ExistingNote": 0},
    )

    # ExistingNote was NOT created by this transaction
    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert not success
    # Rule 1 (orphan) + Rule 3 (broken backlinks) both fire for a note that
    # loses its last incoming link, so there may be more than one error.
    assert any("Unplanned orphans introduced: notes/ExistingNote.md" in e for e in errors)


def test_graph_diff_pre_existing_orphan_outside_domain_not_flagged():
    # A note that was already an orphan in the vault but was NOT in the
    # pre-snapshot domain (link_counts has no entry for it) appears in the
    # post-snapshot because a newly-created note links to it — expanding the
    # 1-hop neighborhood.  The regression gate must NOT fire: we have no
    # pre-write baseline, so we cannot call it a new regression.
    # This is the exact scenario that was triggering false rollbacks on
    # pre-existing notes like Deep Learning/Learning Rate.md.
    pre = GraphSnapshot(
        orphans=[],
        unresolved=[],
        link_counts={},
        backlink_counts={},
    )
    post = GraphSnapshot(
        orphans=[
            NoteRef(name="ExistingOrphan", path="notes/ExistingOrphan.md")
        ],
        unresolved=[],
        link_counts={"notes/ExistingOrphan": 0},
        backlink_counts={"notes/ExistingOrphan": 0},
    )

    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert success, f"Pre-existing orphan outside pre-domain must not be flagged: {errors}"
    assert not errors


def test_graph_diff_new_unresolved_links_rejected():
    # NoteA was in the snapshot domain (being patched): it appears in pre.link_counts.
    # A patch then introduces a new ghost link — genuine regression, must be rejected.
    pre = GraphSnapshot(
        orphans=[],
        unresolved=[],
        link_counts={"notes/NoteA": 0},  # NoteA observed pre-write (0 outgoing links)
        backlink_counts={},
    )
    post = GraphSnapshot(
        orphans=[],
        unresolved=[
            Link(source=NoteRef(name="NoteA", path="notes/NoteA.md"), target="NoteB")
        ],
        link_counts={"notes/NoteA": 1},
        backlink_counts={},
    )

    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert not success
    assert len(errors) == 1
    assert "New unresolved links introduced: [[NoteA]] -> [[NoteB]]" in errors[0]


def test_graph_diff_case_insensitivity_and_path_normalization():
    # Verify that differences in path slash or case do not trigger false regressions
    pre = GraphSnapshot(
        orphans=[NoteRef(name="Orphan1", path="notes\\Orphan1.md")],
        unresolved=[Link(source=NoteRef(name="A", path="notes/A.md"), target="Missing")]
    )
    post = GraphSnapshot(
        orphans=[NoteRef(name="Orphan1", path="notes/orphan1.md")],
        unresolved=[Link(source=NoteRef(name="a", path="notes/a.md"), target="missing")]
    )
    
    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert success
    assert not errors


def test_graph_diff_ghost_links_from_created_notes_allowed():
    """Ghost links (unresolved wikilinks) from newly created notes must NOT
    trigger the regression gate. A created note that references a concept not
    yet in the vault is an intentional forward reference — exactly the pattern
    the injector produces (e.g. [[Stochastic Gradient Descent]] inside a newly
    created Gradient Descent note). Mirrors the planned-orphans exemption in Rule 1."""
    pre = GraphSnapshot(orphans=[], unresolved=[])
    post = GraphSnapshot(
        orphans=[],
        unresolved=[
            Link(
                source=NoteRef(name="Gradient Descent", path="DL/Gradient Descent.md"),
                target="Stochastic Gradient Descent",
            ),
            Link(
                source=NoteRef(name="Learning Rate", path="DL/Learning Rate.md"),
                target="Stochastic Gradient Descent",
            ),
        ],
    )
    # Both source notes were created in this transaction
    success, errors = check_graph_regression(
        pre,
        post,
        created_paths=[
            "DL/Gradient Descent.md",
            "DL/Learning Rate.md",
        ],
    )
    assert success, f"Ghost links from created notes should be allowed. Errors: {errors}"
    assert not errors


def test_graph_diff_ghost_links_from_existing_notes_rejected():
    """A new ghost link whose source is a PRE-EXISTING note that was in the
    snapshot domain (e.g. was being patched) is a genuine regression and must
    be rejected — e.g. a patch op silently nuked a previously-resolved link.
    The note must appear in pre.link_counts to establish a baseline."""
    pre = GraphSnapshot(
        orphans=[],
        unresolved=[],
        link_counts={"notes/ExistingNote": 1},  # was in domain with 1 outgoing link
        backlink_counts={"notes/ExistingNote": 0},
    )
    post = GraphSnapshot(
        orphans=[],
        unresolved=[
            Link(
                source=NoteRef(name="ExistingNote", path="notes/ExistingNote.md"),
                target="NowMissing",
            ),
        ],
        link_counts={"notes/ExistingNote": 2},
        backlink_counts={"notes/ExistingNote": 0},
    )
    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert not success
    assert "New unresolved links introduced" in errors[0]


def test_graph_diff_ghost_links_from_pre_existing_note_outside_domain_not_flagged():
    """Rule 2 domain guard — the false-positive scenario fixed by this commit.

    Chain: write op creates RAM.md with snippet [[GPS]].
    GPS.md pre-exists in the vault with ghost links [[LiDAR]], [[IMU]].
    Pre-snapshot: RAM.md didn't exist → empty domain → GPS.md absent from link_counts.
    Post-snapshot: RAM.md resolves to GPS, so GPS enters the 1-hop neighborhood.
    GPS's pre-existing ghost links surface as new_unres.

    Without the norm_pre_observed guard this triggered ROLLBACK — a false positive.
    With the guard: GPS not in norm_pre_observed → no baseline → EXEMPT.
    """
    gps_ref = NoteRef(name="GPS", path="notes/GPS.md")
    pre = GraphSnapshot(
        orphans=[],
        unresolved=[],
        link_counts={},      # GPS.md was not in the pre-snapshot domain
        backlink_counts={},
    )
    post = GraphSnapshot(
        orphans=[],
        unresolved=[
            Link(source=gps_ref, target="LiDAR"),
            Link(source=gps_ref, target="IMU"),
        ],
        link_counts={"notes/RAM": 1, "notes/GPS": 2},
        backlink_counts={"notes/GPS": 1},
    )
    success, errors = check_graph_regression(
        pre, post, created_paths=["notes/RAM.md"]
    )
    assert success, f"Pre-existing ghost links outside pre-domain must not be flagged: {errors}"
    assert not errors


def test_graph_diff_broken_backlinks_rejected():
    # If a pre-existing note has its backlink count decreased, reject it
    pre = GraphSnapshot(
        orphans=[],
        unresolved=[],
        backlink_counts={"NoteA": 2, "NoteB": 1}
    )
    post = GraphSnapshot(
        orphans=[],
        unresolved=[],
        backlink_counts={"NoteA": 1, "NoteB": 1}
    )
    
    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert not success
    assert len(errors) == 1
    assert "Broken backlinks detected for 'NoteA': decreased from 2 to 1" in errors[0]
