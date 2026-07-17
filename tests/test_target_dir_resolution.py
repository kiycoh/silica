"""resolve_target_dir folds a typed folder path onto existing vault casing.

Real incident (2026-07-17 nucleate run): the user typed
'Informatica/Intelligenza Artificiale/Machine Learning' against a vault
holding 'Informatica/Intelligenza artificiale/Machine learning' — every new
note ENOENT'd through the Obsidian bridge and collision paths mismatched.
"""
from silica.config import CONFIG
from silica.kernel.paths import resolve_target_dir


def _vault(tmp_path, monkeypatch, *dirs):
    for d in dirs:
        (tmp_path / d).mkdir(parents=True)
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))


def test_exact_match_passes_through(tmp_path, monkeypatch):
    _vault(tmp_path, monkeypatch, "Informatica/Intelligenza artificiale")
    assert resolve_target_dir("Informatica/Intelligenza artificiale") == \
        "Informatica/Intelligenza artificiale"


def test_case_mismatch_adopts_existing_casing(tmp_path, monkeypatch):
    _vault(tmp_path, monkeypatch, "Informatica/Intelligenza artificiale/Machine learning")
    assert resolve_target_dir("Informatica/Intelligenza Artificiale/Machine Learning") == \
        "Informatica/Intelligenza artificiale/Machine learning"


def test_new_tail_segment_keeps_typed_casing(tmp_path, monkeypatch):
    _vault(tmp_path, monkeypatch, "Informatica/Intelligenza artificiale")
    assert resolve_target_dir("Informatica/Intelligenza Artificiale/Deep Learning") == \
        "Informatica/Intelligenza artificiale/Deep Learning"


def test_wholly_new_path_unchanged(tmp_path, monkeypatch):
    _vault(tmp_path, monkeypatch)
    assert resolve_target_dir("Nuova/Cartella") == "Nuova/Cartella"


def test_empty_and_unconfigured_pass_through(tmp_path, monkeypatch):
    assert resolve_target_dir("") == ""
    monkeypatch.setattr(CONFIG, "vault_path", "")
    assert resolve_target_dir("A/B") == "A/B"


def test_absolute_path_unchanged(tmp_path, monkeypatch):
    _vault(tmp_path, monkeypatch, "Informatica")
    abs_dir = str(tmp_path / "INFORMATICA")
    assert resolve_target_dir(abs_dir) == abs_dir
