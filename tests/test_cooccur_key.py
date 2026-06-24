"""CooccurStore keyspace is `.md`-suffix agnostic (single-source-of-truth key).

The landmine this pins: producers store the stripped path ('notes/foo') while
consumers pass graph node ids that carry '.md' ('notes/foo.md'). Without one
canonical key the lookup misses silently — community_labels drops the community
and falls back to 'Cluster N'. The store must own the normalization so no caller
can diverge.
"""
from __future__ import annotations

from silica.kernel.cooccurrence import CooccurStore, build_contribution


def _contrib(name: str, body: str):
    return build_contribution(name, body)


def test_note_nodes_is_md_suffix_agnostic() -> None:
    store = CooccurStore()
    # Producer stores the stripped path, as the refresh hook does.
    store.upsert_note("notes/attention", _contrib("Attention", "Transformers use self-attention over tokens."))
    # A consumer passing the graph node id (with .md) must resolve the same note.
    assert store.note_nodes("notes/attention.md") == store.note_nodes("notes/attention")
    assert store.note_nodes("notes/attention.md"), "expected a hit despite the .md suffix"


def test_community_labels_match_members_that_carry_md() -> None:
    store = CooccurStore()
    store.upsert_note("a", _contrib("Alpha", "neural networks learn deep representations from data"))
    store.upsert_note("b", _contrib("Beta", "sourdough bread needs a long cold fermentation for flavour"))
    # Caller passes graph node ids WITH .md and no manual strip.
    labels = store.community_labels([{"a.md"}, {"b.md"}])
    assert labels, "communities silently dropped when members carry the .md suffix"


def test_delete_is_md_suffix_agnostic() -> None:
    store = CooccurStore()
    store.upsert_note("a.md", _contrib("Alpha", "neural networks learn representations"))
    # Delete with the other suffix form must still remove the note.
    store.delete_note("a")
    assert store.note_nodes("a") == {}
    assert store.note_nodes("a.md") == {}
