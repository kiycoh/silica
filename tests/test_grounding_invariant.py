import subprocess
from pathlib import Path

from silica.kernel import codedocs


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def _commit(root: Path, rel: str, text: str, msg: str) -> str:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "--", rel], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg, "--", rel], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()


def test_grounded_concept_note_is_stale_tracked(tmp_path):
    _init_repo(tmp_path)
    ref0 = _commit(tmp_path, "silica/auth.py", "v1\n", "c1")
    vault = tmp_path / "docs" / "silica"
    concept = vault / "concepts"
    concept.mkdir(parents=True)
    (concept / "Auth.md").write_text(
        f"---\ndocuments:\n  - silica/auth.py\ncode_ref: {ref0}\n---\n\n# Auth\n",
        encoding="utf-8",
    )
    _commit(tmp_path, "silica/auth.py", "v2\n", "c2")

    stale = codedocs.stale_docs(vault, repo_root=tmp_path)
    assert len(stale) == 1
    assert stale[0].note_path == "concepts/Auth.md"
    assert stale[0].code_path == "silica/auth.py"
