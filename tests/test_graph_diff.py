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
