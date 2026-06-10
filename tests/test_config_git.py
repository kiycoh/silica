from silica.config import SilicaConfig


def test_git_commit_default_off(monkeypatch):
    monkeypatch.delenv("SILICA_GIT_COMMIT", raising=False)
    assert SilicaConfig().git_commit == "off"


def test_git_commit_auto(monkeypatch):
    monkeypatch.setenv("SILICA_GIT_COMMIT", "auto")
    assert SilicaConfig().git_commit == "auto"
