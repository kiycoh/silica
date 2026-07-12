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
