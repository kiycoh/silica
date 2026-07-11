# tests/test_report_code_signals.py
"""graph_report code signals — coverage + import autolink candidates (spec §4b, §5)."""
from silica.kernel.codegraph import CodeGraph
from silica.kernel.graph_report.code_signals import _coverage_from, _import_autolinks_from

GRAPH = CodeGraph(head_ref="x", files={
    "pkg/paths.py": {"imports": [], "symbols": []},
    "pkg/embed.py": {"imports": ["pkg/paths.py"], "symbols": []},
    "pkg/cli.py":   {"imports": ["pkg/paths.py"], "symbols": []},
    "pkg/lone.py":  {"imports": [], "symbols": []},
})
DOCMAP = {  # note id (with .md) → documented repo paths
    "notes/embed.md": ["pkg/embed.py"],
    "notes/paths.md": ["pkg/paths.py"],
}


def test_coverage_counts_and_fan_in_order():
    cov = _coverage_from(GRAPH, DOCMAP)
    assert cov.documented == 2 and cov.total == 4          # "2/4 file documentati"
    # undocumented sorted by fan-in desc, then path: cli imports nothing but
    # nothing imports it either; paths.py has fan-in 2 but is documented.
    assert cov.undocumented == [["pkg/cli.py", 0], ["pkg/lone.py", 0]]


def test_import_autolink_candidate_when_wikilink_missing():
    cands = _import_autolinks_from(GRAPH, DOCMAP, wikilinks=set())
    assert len(cands) == 1
    c = cands[0]
    assert {c.source, c.target} == {"notes/embed.md", "notes/paths.md"}
    assert c.provenance == "import"
    assert c.shared == ["pkg/embed.py imports pkg/paths.py"]


def test_no_candidate_when_already_linked():
    linked = {tuple(sorted(("notes/embed.md", "notes/paths.md")))}
    assert _import_autolinks_from(GRAPH, DOCMAP, wikilinks=linked) == []
