from __future__ import annotations

import textwrap

from silica.config import CONFIG
from silica.router.orchestrator import InjectorFSM
from silica.router.refiner_fsm import RefinerFSM


def _legal_pack(tmp_path, monkeypatch):
    domains = tmp_path / "domains"
    domains.mkdir()
    (domains / "legal.yaml").write_text(textwrap.dedent("""
        gates:
          rejection_rate_max: 0.05
    """))
    monkeypatch.setenv("SILICA_DOMAINS_DIR", str(domains))


def test_injector_domain_threshold_reaches_fsm(tmp_path, monkeypatch):
    _legal_pack(tmp_path, monkeypatch)
    monkeypatch.setattr(CONFIG, "domain", "legal", raising=False)
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm._get_recipe_gate("rejection_rate_max", 0.10) == 0.05


def test_injector_no_domain_threshold_is_base(monkeypatch):
    monkeypatch.setattr(CONFIG, "domain", None, raising=False)
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm._get_recipe_gate("rejection_rate_max", 0.10) == 0.10


def test_refiner_domain_threshold_reaches_fsm(tmp_path, monkeypatch):
    _legal_pack(tmp_path, monkeypatch)
    monkeypatch.setattr(CONFIG, "domain", "legal", raising=False)
    fsm = RefinerFSM("TargetDir")
    assert fsm._get_recipe_gate("rejection_rate_max", 0.10) == 0.05


def test_refiner_no_domain_threshold_is_base(monkeypatch):
    monkeypatch.setattr(CONFIG, "domain", None, raising=False)
    fsm = RefinerFSM("TargetDir")
    assert fsm._get_recipe_gate("rejection_rate_max", 0.10) == 0.10
