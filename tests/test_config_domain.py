from silica.config import SilicaConfig


def test_domain_default_is_none():
    assert SilicaConfig().domain is None


def test_domain_from_env(monkeypatch):
    monkeypatch.setenv("SILICA_DOMAIN", "legal")
    assert SilicaConfig().domain == "legal"
