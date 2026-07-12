"""kernel/codewiki — partition + digest for the behavioral code wiki."""
from silica.kernel.codegraph import CodeGraph
from silica.kernel.codewiki import Subsystem, partition, source_root


def _graph(paths, entries=None):
    files = {p: {"language": "python", "imports": [], "external": [],
                 "unresolved": [], "symbols": [], "calls": [],
                 "module_doc": "", "module_comments": [], "dunder_all": None,
                 "has_main_guard": False, "parse_error": False}
             for p in paths}
    for p, extra in (entries or {}).items():
        files[p].update(extra)
    return CodeGraph(head_ref="abc123", files=files)


def test_source_root_is_densest_top_dir():
    g = _graph(["silica/cli.py", "silica/kernel/a.py", "silica/kernel/b.py",
                "tools/x.py"])
    assert source_root(g) == "silica"


def test_partition_subdirs_and_core():
    g = _graph(["silica/cli.py", "silica/config.py",
                "silica/kernel/a.py", "silica/kernel/graph/deep.py",
                "silica/router/r.py"])
    subs = {s.key: s for s in partition(g)}
    assert set(subs) == {"core", "kernel", "router"}
    assert subs["core"].members == ["silica/cli.py", "silica/config.py"]
    # deep files roll up to the immediate subdir
    assert "silica/kernel/graph/deep.py" in subs["kernel"].members


def test_flat_repo_root_excludes_tests_and_docs():
    g = _graph(["app.py", "lib/x.py", "tests/test_x.py", "docs/conf.py"])
    subs = {s.key: s for s in partition(g)}
    assert "tests" not in subs and "docs" not in subs
    assert "core" in subs and "lib" in subs


# ---------------------------------------------------------------------------
# Task 6: SubsystemDigest
# ---------------------------------------------------------------------------

from silica.kernel.codewiki import build_digests, cross_edges, edges_ref


def _rich_graph():
    return _graph(
        ["silica/cli.py",
         "silica/kernel/util.py", "silica/kernel/other.py",
         "silica/router/r.py"],
        entries={
            "silica/cli.py": {
                "imports": ["silica/kernel/util.py"],
                "calls": [{"target": "silica/kernel/util.py",
                           "callee": "helper", "caller": "main"}],
                "has_main_guard": True,
                "symbols": [{"kind": "function", "name": "main", "parent": "",
                             "signature": "def main()", "doc": "", "doc_full": "",
                             "decorators": []}],
            },
            "silica/router/r.py": {
                "imports": ["silica/kernel/util.py"],
                "symbols": [{"kind": "function", "name": "dispatch", "parent": "",
                             "signature": "def dispatch()", "doc": "", "doc_full": "",
                             "decorators": ["app.command"]}],
            },
            "silica/kernel/util.py": {
                "dunder_all": ["helper"],
                "external": ["orjson"],
                "symbols": [
                    {"kind": "function", "name": "helper", "parent": "",
                     "signature": "def helper()", "doc": "", "doc_full": "", "decorators": []},
                    {"kind": "function", "name": "not_exported", "parent": "",
                     "signature": "def not_exported()", "doc": "", "doc_full": "", "decorators": []},
                ],
            },
        },
    )


def test_digest_collaborators_two_weights_and_publics(tmp_path):
    g = _rich_graph()
    digests = {d.key: d for d in build_digests(g, partition(g), tmp_path)}
    core = digests["core"]
    assert ("kernel", 1, 1) in core.collaborators_out       # 1 import edge, 1 call edge
    kernel = digests["kernel"]
    assert ("core", 1, 1) in kernel.collaborators_in
    assert ("router", 1, 0) in kernel.collaborators_in      # import-only, zero calls
    names = [s["name"] for s in kernel.public_symbols["silica/kernel/util.py"]]
    assert names == ["helper"]                              # __all__ is the authority
    assert "orjson" in kernel.external_deps


def test_digest_entry_points_labeled(tmp_path):
    g = _rich_graph()
    digests = {d.key: d for d in build_digests(g, partition(g), tmp_path)}
    labels = dict(digests["core"].entry_points)
    assert "__main__ guard" in labels["silica/cli.py"]
    labels_r = dict(digests["router"].entry_points)
    assert "registration decorator" in labels_r["silica/router/r.py"]


def test_struct_sig_changes_on_body_only_call_change(tmp_path):
    g1 = _rich_graph()
    d1 = {d.key: d for d in build_digests(g1, partition(g1), tmp_path)}
    g2 = _rich_graph()
    g2.files["silica/cli.py"]["calls"] = []   # body-only edit: call removed
    d2 = {d.key: d for d in build_digests(g2, partition(g2), tmp_path)}
    assert d1["core"].struct_sig != d2["core"].struct_sig
    assert d1["core"].members == d2["core"].members


def test_cross_edges_and_ref(tmp_path):
    g = _rich_graph()
    edges = cross_edges(g, partition(g))
    assert ("core", "kernel", 1, 1) in edges
    assert ("router", "kernel", 1, 0) in edges
    ref = edges_ref(edges)
    # weight-only change must NOT move the ref
    bumped = [(a, b, iw + 5, cw) for (a, b, iw, cw) in edges]
    assert edges_ref(bumped) == ref
