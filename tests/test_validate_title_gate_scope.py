# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Task 2.4 — validate: title gate scoped to target dir, near-title list hoisted.

`_target_dir_titles()` used to enumerate the ENTIRE vault via
`DRIVER.search_names("")` and the write branch rebuilt the near-title
candidate list on every op. This covers:

  1. Scoping: `_target_dir_titles()` enumerates via `DRIVER.list_files(norm_dir)`
     (the target-dir subtree), not the whole vault.
  2. Equivalence: a note exactly in the target dir still gates (key-equal
     coercion); a note in a SUBFOLDER of the target dir, or in a sibling dir,
     must NOT gate — proving the `list_files(norm_dir)` swap kept the exact
     `dirname == norm_dir` filter's semantics.
  3. Hoist: the near-title candidate list is the SAME object across every
     write op inside one `validate_operations()` call — built once, not
     rebuilt per op.
"""
from silica.kernel.validate import MIN_WRITE_SNIPPET_CHARS, validate_operations

# Write ops must clear the precision gate; the padding keeps fixtures short.
_PAD = " lorem" * (MIN_WRITE_SNIPPET_CHARS // 6 + 1)


def _write_op(heading: str, path: str) -> dict:
    return {
        "op": "write", "path": path, "heading": heading,
        "source_basename": "lez.md",
        "snippet": f"corpo di {heading}" + _PAD,
    }


def test_target_dir_titles_scopes_enumeration_to_target_dir(tmp_vault, monkeypatch):
    """`_target_dir_titles()` must call DRIVER.list_files(norm_dir) — the
    target-dir subtree — instead of enumerating the whole vault."""
    import silica.driver as driver_mod

    tmp_vault.note("Corso/Machine Learning.md", "# ML\n\ncorpo")
    tmp_vault.note("Corso/Sub/Nested.md", "# Nested\n\ncorpo")
    tmp_vault.note("Altro/Other.md", "# Other\n\ncorpo")

    real_list_files = driver_mod.DRIVER.list_files
    calls: list[str] = []

    def _spy(folder=""):
        calls.append(folder)
        return real_list_files(folder)

    monkeypatch.setattr(driver_mod.DRIVER, "list_files", _spy)

    validate_operations(
        [_write_op("Machine Learning (9 CFU)", "Corso/Machine Learning (9 CFU).md")],
        [], "Corso",
    )

    assert calls == ["Corso"], "title gate must enumerate via list_files(norm_dir), not the whole vault"


def test_equivalence_target_dir_sibling_subfolder(tmp_vault):
    """One note exactly in the target dir (must gate), one in a SUBFOLDER of
    the target dir (must NOT gate the fuzzy-near band), and one in a sibling
    dir (must NOT gate key-equal coercion) — all inside one validate() call."""
    tmp_vault.note("Corso/Machine Learning.md", "# ML\n\ncorpo")        # target dir — must gate (key-equal)
    tmp_vault.note("Corso/Sub/Descriptor.md", "# Descriptor\n\ncorpo")  # subfolder — must NOT gate (near)
    tmp_vault.note("Altro/Alpha.md", "# Alpha\n\ncorpo")                # sibling dir — must NOT gate (key-equal)

    ops = [
        _write_op("Machine Learning (9 CFU)", "Corso/Machine Learning (9 CFU).md"),
        _write_op("Description", "Corso/Description.md"),
        _write_op("Alpha", "Corso/Alpha.md"),
    ]
    validated, rejected = validate_operations(ops, [], "Corso")

    assert rejected == []

    coerced = [o for o in validated if o.heading == "Machine Learning (9 CFU)"]
    assert len(coerced) == 1
    assert coerced[0].op.value == "patch"
    assert coerced[0].path == "Corso/Machine Learning.md"

    unaffected_by_subfolder = [o for o in validated if o.heading == "Description"]
    assert len(unaffected_by_subfolder) == 1
    assert unaffected_by_subfolder[0].op.value == "write"

    unaffected_by_sibling = [o for o in validated if o.heading == "Alpha"]
    assert len(unaffected_by_sibling) == 1
    assert unaffected_by_sibling[0].op.value == "write"
    assert unaffected_by_sibling[0].path == "Corso/Alpha.md"


def test_target_dir_title_list_built_once_per_validate_call(tmp_vault, monkeypatch):
    """The near-title candidate list must be the SAME object across every
    write op inside one validate_operations() call — built once, not per-op."""
    import silica.kernel.title as title_mod

    tmp_vault.note("Corso/Existing.md", "# Existing\n\ncorpo")

    real_near_titles = title_mod.near_titles
    seen_ids: list[int] = []

    def _spy(stem, titles, *a, **kw):
        seen_ids.append(id(titles))
        return real_near_titles(stem, titles, *a, **kw)

    monkeypatch.setattr(title_mod, "near_titles", _spy)

    ops = [
        _write_op("Alpha", "Corso/Alpha.md"),
        _write_op("Beta", "Corso/Beta.md"),
    ]
    validate_operations(ops, [], "Corso")

    assert len(seen_ids) == 2
    assert seen_ids[0] == seen_ids[1], "near-title candidate list must be built once per validate() call"
