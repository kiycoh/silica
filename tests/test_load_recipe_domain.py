from __future__ import annotations

import textwrap

from silica.router.recipe_parser import load_recipe


def test_no_domain_is_unchanged():
    plain = load_recipe("injector")
    same = load_recipe("injector", domain=None)
    assert plain == same


def test_missing_domain_file_falls_back_to_base(caplog):
    base = load_recipe("injector")
    out = load_recipe("injector", domain="does-not-exist")
    assert out == base  # warn, do not crash


def test_legal_pack_real_file():
    """Integration: legal.yaml in silica/domains/ applies both overrides."""
    out = load_recipe("injector", domain="legal")
    assert out["gates"]["rejection_rate_max"] == 0.05
    payload = next(p for p in out["phases"] if p["id"] == "payload")
    assert payload["partition_if_over"] == 4


def test_domain_overrides_gate(tmp_path, monkeypatch):
    domains = tmp_path / "domains"
    domains.mkdir()
    (domains / "legal.yaml").write_text(
        textwrap.dedent(
            """
            gates:
              rejection_rate_max: 0.05
            """
        )
    )
    monkeypatch.setenv("SILICA_DOMAINS_DIR", str(domains))
    out = load_recipe("injector", domain="legal")
    assert out["gates"]["rejection_rate_max"] == 0.05
