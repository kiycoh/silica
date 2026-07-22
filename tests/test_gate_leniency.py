# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate-leniency fixes: heading token-subset remap, YAML pipe unescape,
fence auto-close, and diff-aware patch lint. Each recovers a class of op the
mineru run of 2026-07-22 deferred without a real defect."""

from silica.kernel.templates import _link_name, close_unbalanced_fences


# --- pure normalizers --------------------------------------------------------

def test_link_name_unescapes_table_pipe():
    # `[[Target\|Alias]]` — `\|` is an invalid escape inside the double-quoted
    # YAML scalar we wrap the link in; it must collapse to a bare pipe.
    assert _link_name(r"[[User Defined Function\|UDF]]") == "User Defined Function|UDF"


def test_link_name_still_strips_brackets():
    assert _link_name("[[Spark]]") == "Spark"


def test_close_unbalanced_fences_appends_when_odd():
    out = close_unbalanced_fences("text\n```python\nprint(1)")
    assert out.count("```") % 2 == 0
    assert out.rstrip().endswith("```")


def test_close_unbalanced_fences_noop_when_balanced():
    body = "text\n```python\nprint(1)\n```\n"
    assert close_unbalanced_fences(body) == body


# --- heading -> concept token-subset remap -----------------------------------

def _payloads(concepts, inbox="Inbox/spark.md"):
    return [{"batches": [{"inbox_file": inbox,
                          "concepts": [{"name": c} for c in concepts]}]}]


def _write_op(heading, path, source="spark.md"):
    return {"op": "write", "path": path, "heading": heading,
            "source_basename": source,
            "snippet": f"corpo di {heading} " + "lorem " * 30}


def test_heading_unique_token_subset_remaps(tmp_vault):
    """'DataFrame' is a clean rephrase of the registered 'dataframe spark' —
    a unique token-subset match remaps instead of deferring."""
    from silica.kernel.validate import validate_operations

    validated, rejected = validate_operations(
        [_write_op("DataFrame", "Spark/DataFrame.md")],
        _payloads(["dataframe spark"]), "Spark",
    )
    assert rejected == []
    assert "dataframe spark" in [o.heading for o in validated]
    assert not any(o.heading == "DataFrame" for o in validated)


def test_heading_ambiguous_token_subset_still_rejects(tmp_vault):
    """Two concepts share the token — no unique match, so the anti-hallucination
    gate still refuses rather than guess."""
    from silica.kernel.validate import validate_operations

    validated, rejected = validate_operations(
        [_write_op("DataFrame", "Spark/DataFrame.md")],
        _payloads(["dataframe spark", "dataframe rdd"]), "Spark",
    )
    assert not validated
    assert len(rejected) == 1
    assert "not present in payload concepts" in rejected[0].reason


# --- diff-aware patch lint ---------------------------------------------------

def test_patch_not_reverted_for_preexisting_violation(tmp_vault):
    """A patch to a user note carrying an [!definizione] callout must land —
    the pre-existing violation is not one the patch introduced."""
    from silica.kernel.atomic_write import commit_note_atomic
    from silica.kernel.ops import Op, OpType
    from silica.driver import DRIVER

    path = "Math/Insiemi.md"
    tmp_vault.note(
        path,
        "---\nAI: true\nlast modified: 2026-07-22\nrelated:\n  - \"[[Hub]]\"\n---\n\n"
        "# Insiemi\n\n> [!definizione] Def\n> Un insieme e limitato se ammette un maggiorante.\n",
    )

    op = Op(op=OpType.patch, path=path, heading="LIMIT",
            source_basename="spark.md", hub="Hub",
            snippet="LIMIT restringe il numero di righe restituite da una query.")
    res = commit_note_atomic(op, hub="Hub", lint=True)

    assert res.ok, f"patch should survive a pre-existing violation, got: {res.error}"
    assert not res.reverted
    assert "LIMIT restringe" in DRIVER.read_note(path).content


def test_patch_reverted_for_newly_introduced_violation(tmp_vault):
    """The gate still bites: a snippet that introduces a NEW unknown callout is
    reverted, proving the baseline subtraction is not a blanket bypass."""
    from silica.kernel.atomic_write import commit_note_atomic
    from silica.kernel.ops import Op, OpType

    path = "Math/Clean.md"
    tmp_vault.note(
        path,
        "---\nAI: true\nlast modified: 2026-07-22\nrelated:\n  - \"[[Hub]]\"\n---\n\n"
        "# Clean\n\nCorpo pulito.\n",
    )

    op = Op(op=OpType.patch, path=path, heading="Bad",
            source_basename="spark.md", hub="Hub",
            snippet="> [!inventato] questo callout non esiste nel vocabolario.")
    res = commit_note_atomic(op, hub="Hub", lint=True)

    assert not res.ok
    assert res.reverted
    assert "inventato" in res.error
