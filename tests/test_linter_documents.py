from silica.kernel.linter import check_documents_paths


def test_missing_documents_path_warns(tmp_path):
    (tmp_path / "exists.py").write_text("x", encoding="utf-8")
    data = {"documents": ["exists.py", "gone.py"]}
    warns = check_documents_paths(data, repo_root=tmp_path)
    assert len(warns) == 1
    assert "gone.py" in warns[0]


def test_all_paths_present_no_warn(tmp_path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    data = {"documents": ["a.py"]}
    assert check_documents_paths(data, repo_root=tmp_path) == []


def test_no_documents_key_no_warn(tmp_path):
    assert check_documents_paths({"title": "x"}, repo_root=tmp_path) == []
