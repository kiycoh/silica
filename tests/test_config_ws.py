from silica.config import SilicaConfig


def test_ws_port_default_zero_means_ephemeral(monkeypatch):
    monkeypatch.delenv("SILICA_WS_PORT", raising=False)
    assert SilicaConfig().ws_port == 0  # 0 → OS picks a free port


def test_ws_port_from_env(monkeypatch):
    monkeypatch.setenv("SILICA_WS_PORT", "8899")
    assert SilicaConfig().ws_port == 8899


def test_ws_token_default_empty_means_generate(monkeypatch):
    monkeypatch.delenv("SILICA_WS_TOKEN", raising=False)
    assert SilicaConfig().ws_token == ""  # empty → `silica connect` mints one


def test_ws_token_from_env(monkeypatch):
    monkeypatch.setenv("SILICA_WS_TOKEN", "deadbeef")
    assert SilicaConfig().ws_token == "deadbeef"
