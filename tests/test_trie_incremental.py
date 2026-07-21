"""Task 2.1 — incremental mention-trie updates in ObsidianFSBackend._patch_index.

_patch_index used to rebuild the ENTIRE mention trie via build_title_trie() on
every single write (and move() calls _patch_index R+2 times per move, R =
referrer count). The backend now holds ONE trie as state (self._title_trie)
and inserts/removes only the single changed title on each patch.

Covers the four correctness cases from the perf audit (docs/audits/2026-07-21-
perf-audit-hot-paths.md, root fix 2):
  1. create -> mention
  2. delete (unique title) -> mention no longer recorded
  3. delete one of two same-named notes -> title still mentionable
  4. move/rename -> old title gone, new title present
plus a direct unit test of trie_insert/trie_remove and a parity check that
build_title_trie's output is unchanged by the trie_insert refactor.
"""
from __future__ import annotations

from pathlib import Path

from silica.driver.base import (
    _TITLE,
    build_title_trie,
    mentions_in,
    trie_insert,
    trie_remove,
)
from silica.driver.fs_backend import ObsidianFSBackend


def _backend(tmp_path: Path, notes: dict[str, str] | None = None) -> ObsidianFSBackend:
    """Fresh FS backend over an initial vault, indexed once.

    Consumes the startup full rebuild so later ops in a test exercise the
    incremental _patch_index path, not _rebuild_index.
    """
    for name, body in (notes or {}).items():
        p = tmp_path / f"{name}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    b = ObsidianFSBackend(str(tmp_path))
    b._ensure_index()
    return b


# ---------------------------------------------------------------------------
# Unit tests: trie_insert / trie_remove
# ---------------------------------------------------------------------------

def test_trie_insert_then_mentions_in_finds_it():
    trie: dict = {}
    trie_insert(trie, "foo")
    assert mentions_in("see foo here", trie) == {"foo"}


def test_trie_remove_then_mentions_in_no_longer_finds_it():
    trie: dict = {}
    trie_insert(trie, "foo")
    trie_remove(trie, "foo")
    assert mentions_in("see foo here", trie) == set()


def test_trie_remove_missing_title_is_noop():
    trie: dict = {}
    trie_insert(trie, "foo")
    trie_remove(trie, "bar")  # never inserted -- must not raise or corrupt trie
    assert mentions_in("see foo here", trie) == {"foo"}


def test_trie_insert_short_title_skipped():
    trie: dict = {}
    trie_insert(trie, "a")  # < 2 chars, same guard as build_title_trie
    assert trie == {}


def test_build_title_trie_structure_unchanged():
    """Pin build_title_trie's output for a small title set across the
    trie_insert refactor -- behavior/output must stay identical."""
    trie = build_title_trie(["ab", "abc", "xy"])
    expected = {
        "a": {"b": {_TITLE: "ab", "c": {_TITLE: "abc"}}},
        "x": {"y": {_TITLE: "xy"}},
    }
    assert trie == expected


# ---------------------------------------------------------------------------
# Correctness case 1: create -> mention
# ---------------------------------------------------------------------------

def test_create_then_mention_indexed(tmp_path):
    b = _backend(tmp_path, {"Foo": "the Foo note"})
    b.create("Essay.md", "this note has a see Foo reference")
    assert "Essay.md" in b._mention_index.get("foo", set())
    assert "Essay.md" in b.mentions_of("Foo")


# ---------------------------------------------------------------------------
# Correctness case 2: delete (unique) -> no mention
# ---------------------------------------------------------------------------

def test_delete_unique_title_removes_mention(tmp_path):
    b = _backend(tmp_path, {"Foo": "the Foo note", "Other": "unrelated body"})
    b.delete("Foo.md")

    # "foo" must be gone from the trie itself...
    assert mentions_in("see foo here", b._title_trie) == set()

    # ...so a subsequently-patched note mentioning "Foo" records no mention.
    b.create("Essay.md", "this note mentions Foo but Foo is gone")
    assert "Essay.md" not in b._mention_index.get("foo", set())
    assert "Essay.md" not in b.mentions_of("Foo")


# ---------------------------------------------------------------------------
# Correctness case 3: delete one of two same-named -> still mentionable
# ---------------------------------------------------------------------------

def test_delete_one_of_two_same_named_still_mentioned(tmp_path):
    b = _backend(tmp_path, {"a/Foo": "content A", "b/Foo": "content B"})
    b.delete("a/Foo.md")

    assert "a/Foo.md" not in b._notes
    assert "b/Foo.md" in b._notes

    # "foo" must still be in the trie (b/Foo.md still holds the name)...
    assert mentions_in("see foo here", b._title_trie) == {"foo"}

    # ...so a subsequently-patched note mentioning "Foo" IS recorded.
    b.create("Essay.md", "this note references Foo directly")
    assert "Essay.md" in b.mentions_of("Foo")


# ---------------------------------------------------------------------------
# Correctness case 4: move/rename -> old gone, new present
# ---------------------------------------------------------------------------

def test_move_rename_updates_trie(tmp_path):
    b = _backend(tmp_path, {"Foo": "the Foo note"})
    b.move("Foo.md", "Bar.md")

    assert mentions_in("about foo and bar", b._title_trie) == {"bar"}

    b.create("Essay.md", "this note has both foo and bar words")
    assert "Essay.md" not in b.mentions_of("Foo")
    assert "Essay.md" in b.mentions_of("Bar")


# ---------------------------------------------------------------------------
# _patch_index no longer rebuilds the trie from scratch
# ---------------------------------------------------------------------------

def test_patch_index_mutates_trie_in_place(tmp_path):
    """The trie object identity must survive a patch -- proof _patch_index
    inserts/removes in place instead of calling build_title_trie() again."""
    b = _backend(tmp_path, {"Foo": "the Foo note"})
    trie_id_before = id(b._title_trie)

    b.create("Essay.md", "mentions Foo")
    assert id(b._title_trie) == trie_id_before

    b.delete("Essay.md")
    assert id(b._title_trie) == trie_id_before
