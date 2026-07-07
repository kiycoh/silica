# tests/test_vault_command.py
import silica.driver as driver_pkg
from silica.cli import _handle_direct_shortcut
from silica.config import CONFIG
from silica.ui.commands import command_names


def test_vault_is_registered():
    assert "/vault" in command_names()


def test_vault_no_args_shows_status_without_mutating(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(CONFIG, "backend", "fs")
    (tmp_path / "a.md").write_text("# a")
    (tmp_path / "b.md").write_text("# b")

    handled = _handle_direct_shortcut("/vault", [])

    assert handled is True
    assert CONFIG.vault_path == str(tmp_path)  # unchanged
    out = capsys.readouterr().out
    assert str(tmp_path) in out
    assert "fs" in out
    assert "2" in out  # note count


def test_vault_no_args_shows_detected_language(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(CONFIG, "backend", "fs")
    (tmp_path / "a.md").write_text(
        "Questo è un appunto scritto in italiano con molte parole comuni "
        "come il, la, di, che, per, con, sono, questo, quella.",
        encoding="utf-8",
    )

    handled = _handle_direct_shortcut("/vault", [])

    assert handled is True
    out = capsys.readouterr().out
    assert "Language: italian" in out


def test_vault_no_args_warns_on_frozen_store_mismatch(tmp_path, monkeypatch, capsys):
    import silica.kernel.cooccurrence as cooc_mod
    from silica.kernel.cooccurrence import CooccurStore

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(CONFIG, "backend", "fs")
    (tmp_path / "a.md").write_text(
        "Questo è un appunto scritto in italiano con molte parole comuni "
        "come il, la, di, che, per, con, sono, questo, quella.",
        encoding="utf-8",
    )
    index_path = tmp_path / "cooc.json"
    monkeypatch.setattr(cooc_mod, "_index_path_for", lambda vault: index_path)
    store = CooccurStore(path=index_path, lang="english")
    store.save()

    handled = _handle_direct_shortcut("/vault", [])

    assert handled is True
    out = capsys.readouterr().out
    assert "Language: italian" in out
    assert "store frozen: english" in out
    assert "/cooccur" in out


def test_vault_switch_updates_config_and_resets_driver(tmp_path, monkeypatch):
    target = tmp_path / "other_vault"
    (target / ".obsidian").mkdir(parents=True)  # Obsidian vault → adopted verbatim
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    # Prime the driver singleton with a sentinel so we can observe the reset.
    monkeypatch.setattr(driver_pkg, "_driver", object())

    handled = _handle_direct_shortcut(f"/vault {target}", [])

    assert handled is True
    assert CONFIG.vault_path == str(target.resolve())
    assert driver_pkg._driver is None  # reset so next get_driver() rebuilds


def test_vault_switch_nonexistent_path_creates_repo_mode(tmp_path, monkeypatch):
    # A not-yet-existing path is no longer rejected: no .obsidian → repo mode,
    # docs/silica created on demand.
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(driver_pkg, "_driver", object())
    target = tmp_path / "does_not_exist"

    handled = _handle_direct_shortcut(f"/vault {target}", [])

    assert handled is True
    assert CONFIG.vault_path == str((target / "docs" / "silica").resolve())
    assert (target / "docs" / "silica").is_dir()
    assert driver_pkg._driver is None  # reset


def test_reset_driver_forces_rebuild(monkeypatch):
    fresh = object()
    monkeypatch.setattr(driver_pkg, "_create_driver", lambda: fresh)
    monkeypatch.setattr(driver_pkg, "_driver", object())

    driver_pkg.reset_driver()
    assert driver_pkg._driver is None

    assert driver_pkg.get_driver() is fresh
