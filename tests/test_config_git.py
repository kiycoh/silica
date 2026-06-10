import importlib

import silica.config as config_mod


def _fresh_config(monkeypatch, value=None):
    if value is None:
        monkeypatch.delenv("SILICA_GIT_COMMIT", raising=False)
    else:
        monkeypatch.setenv("SILICA_GIT_COMMIT", value)
    importlib.reload(config_mod)
    return config_mod.SilicaConfig()


def test_git_commit_default_off(monkeypatch):
    cfg = _fresh_config(monkeypatch, None)
    assert cfg.git_commit == "off"


def test_git_commit_auto(monkeypatch):
    cfg = _fresh_config(monkeypatch, "auto")
    assert cfg.git_commit == "auto"
