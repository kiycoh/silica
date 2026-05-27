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
    # If an existing note becomes an orphan, it is rejected
    pre = GraphSnapshot(orphans=[], unresolved=[])
    post = GraphSnapshot(
        orphans=[
            NoteRef(name="ExistingNote", path="notes/ExistingNote.md")
        ],
        unresolved=[]
    )
    
    # ExistingNote was NOT created by this transaction
    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert not success
    assert len(errors) == 1
    assert "Unplanned orphans introduced: notes/ExistingNote.md" in errors[0]


def test_graph_diff_new_unresolved_links_rejected():
    pre = GraphSnapshot(orphans=[], unresolved=[])
    post = GraphSnapshot(
        orphans=[],
        unresolved=[
            Link(source=NoteRef(name="NoteA", path="notes/NoteA.md"), target="NoteB")
        ]
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
    """A new ghost link whose source is a PRE-EXISTING note is still a regression
    and must be rejected (e.g. a patch op silently nuked a previously-resolved link)."""
    pre = GraphSnapshot(orphans=[], unresolved=[])
    post = GraphSnapshot(
        orphans=[],
        unresolved=[
            Link(
                source=NoteRef(name="ExistingNote", path="notes/ExistingNote.md"),
                target="NowMissing",
            ),
        ],
    )
    # ExistingNote was NOT created — it was already in the vault
    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert not success
    assert "New unresolved links introduced" in errors[0]


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
