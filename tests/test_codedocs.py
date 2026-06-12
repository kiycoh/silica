import subprocess
from pathlib import Path

from silica.kernel import codedocs


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def _commit(path: Path, rel: str, text: str, msg: str) -> str:
    f = path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "--", rel], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg, "--", rel], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True
    ).stdout.strip()


def _write_note(vault: Path, rel: str, documents: list[str], code_ref: str | None) -> None:
    fm_lines = ["---", "documents:"]
    for d in documents:
        fm_lines.append(f"  - {d}")
    if code_ref is not None:
        fm_lines.append(f"code_ref: {code_ref}")
    fm_lines += ["---", "", "doc body"]
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(fm_lines) + "\n", encoding="utf-8")


def test_stale_docs_flags_changed_file(tmp_path):
    _init_repo(tmp_path)
    ref0 = _commit(tmp_path, "src/m.py", "v1\n", "c1")
    vault = tmp_path / "docs"
    vault.mkdir()
    _write_note(vault, "m.md", ["src/m.py"], ref0)
    _commit(tmp_path, "src/m.py", "v2\n", "c2")  # code moved past ref0

    stale = codedocs.stale_docs(vault, repo_root=tmp_path)
    assert len(stale) == 1
    sd = stale[0]
    assert sd.note_path.endswith("m.md")
    assert sd.code_path == "src/m.py"
    assert sd.recorded_ref == ref0
    assert [c.subject for c in sd.intervening] == ["c2"]


def test_stale_docs_clean_when_ref_current(tmp_path):
    _init_repo(tmp_path)
    ref0 = _commit(tmp_path, "src/m.py", "v1\n", "c1")
    vault = tmp_path / "docs"
    vault.mkdir()
    _write_note(vault, "m.md", ["src/m.py"], ref0)
    assert codedocs.stale_docs(vault, repo_root=tmp_path) == []


def test_stale_docs_ignores_notes_without_documents(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "src/m.py", "v1\n", "c1")
    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "plain.md").write_text("---\ntitle: x\n---\n\nhi\n", encoding="utf-8")
    assert codedocs.stale_docs(vault, repo_root=tmp_path) == []


def test_stale_docs_unknown_ref_not_stale(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "src/m.py", "v1\n", "c1")
    vault = tmp_path / "docs"
    vault.mkdir()
    _write_note(vault, "m.md", ["src/m.py"], None)  # no code_ref
    assert codedocs.stale_docs(vault, repo_root=tmp_path) == []


def test_rebadge_updates_code_ref(tmp_path):
    _init_repo(tmp_path)
    ref0 = _commit(tmp_path, "src/m.py", "v1\n", "c1")
    vault = tmp_path / "docs"
    vault.mkdir()
    _write_note(vault, "m.md", ["src/m.py"], ref0)
    new_ref = _commit(tmp_path, "src/m.py", "v2\n", "c2")

    returned = codedocs.rebadge(vault, "m.md", repo_root=tmp_path)
    assert returned == new_ref

    from silica.kernel import frontmatter
    data, _, _ = frontmatter.split((vault / "m.md").read_text(encoding="utf-8"))
    assert data["code_ref"] == new_ref
    # after rebadge the note is no longer stale
    assert codedocs.stale_docs(vault, repo_root=tmp_path) == []


def test_stale_count_zero_without_git(tmp_path):
    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "m.md").write_text("---\ndocuments:\n  - x.py\ncode_ref: abc\n---\n\nb\n", encoding="utf-8")
    assert codedocs.stale_count(vault) == 0  # not a repo → soft zero


# ---------------------------------------------------------------------------
# note_is_stale tests
# ---------------------------------------------------------------------------

from silica.kernel.codedocs import note_is_stale


def _write_note_single(vault, name, documents, code_ref):
    (vault / name).write_text(
        "---\ndocuments:\n  - " + documents + f"\ncode_ref: {code_ref}\n---\n\n# n\n",
        encoding="utf-8",
    )


def test_note_is_stale_fresh_and_stale(tmp_path):
    import subprocess
    from silica.kernel import gitstate

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    vault = tmp_path / ".silica"
    vault.mkdir()
    head = gitstate.head_ref(tmp_path)

    _write_note_single(vault, "n.md", "m.py", head)
    assert note_is_stale(vault, "n.md") is False

    (tmp_path / "m.py").write_text("x = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "change"], cwd=tmp_path, check=True)
    assert note_is_stale(vault, "n.md") is True


def test_note_is_stale_unknown_cases(tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "plain.md").write_text("# no frontmatter\n", encoding="utf-8")
    assert note_is_stale(vault, "plain.md") is None      # no git repo
    assert note_is_stale(vault, "missing.md") is None    # unreadable note
