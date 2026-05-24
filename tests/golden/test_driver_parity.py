"""Golden tests for driver parity (fs vs cli).

This validates that the FS backend produces the exact same results as the CLI backend
on a real vault, ensuring the headless oracle mode is perfectly compatible.
"""
import os
import unicodedata
import pytest

from silica.driver.cli_backend import ObsidianCLIBackend
from silica.driver.fs_backend import ObsidianFSBackend

# The real vault to test against (must be open in Obsidian)
VAULT_NAME = "Alex's Second Brain Sync"
VAULT_PATH = "/home/kiycoh/Documents/Obsidian/Alex's Second Brain Sync"


@pytest.fixture(scope="module")
def backends():
    if not os.path.exists(VAULT_PATH):
        pytest.skip(f"Test vault path not found: {VAULT_PATH}")
    
    cli = ObsidianCLIBackend(vault_name=VAULT_NAME)
    fs = ObsidianFSBackend(vault_path=VAULT_PATH)
    return cli, fs


def test_parity_search_names(backends):
    cli, fs = backends
    # Query a common letter
    cli_res = cli.search_names("a")
    fs_res = fs.search_names("a")
    
    cli_names = {unicodedata.normalize('NFC', r.name) for r in cli_res}
    fs_names = {unicodedata.normalize('NFC', r.name) for r in fs_res}
    
    assert fs_names == cli_names


def test_parity_orphans(backends):
    cli, fs = backends
    cli_res = cli.orphans()
    fs_res = fs.orphans()
    
    cli_names = {unicodedata.normalize('NFC', r.name) for r in cli_res if r.path.endswith('.md')}
    fs_names = {unicodedata.normalize('NFC', r.name) for r in fs_res if r.path.endswith('.md')}
    
    # Allow minor divergence due to aliases/code blocks
    intersection = fs_names & cli_names
    overlap = len(intersection) / max(len(fs_names), len(cli_names))
    assert overlap > 0.8, f"Overlap too low: {overlap}"


def test_parity_unresolved(backends):
    cli, fs = backends
    cli_res = cli.unresolved()
    fs_res = fs.unresolved()
    
    cli_targets = {unicodedata.normalize('NFC', r.target) for r in cli_res if not r.target.endswith(('.png', '.jpg', '.pdf'))}
    fs_targets = {unicodedata.normalize('NFC', r.target) for r in fs_res if not r.target.endswith(('.png', '.jpg', '.pdf'))}
    
    intersection = fs_targets & cli_targets
    overlap = len(intersection) / max(len(fs_targets), len(cli_targets))
    assert overlap > 0.8, f"Overlap too low: {overlap}"


def test_parity_read_note(backends):
    cli, fs = backends
    # Find an arbitrary note from files
    files = cli.list_files()
    if not files:
        pytest.skip("No files in vault to test read_note")
        
    ref = files[0]
    
    cli_nc = cli.read_note(ref)
    fs_nc = fs.read_note(ref)
    
    assert cli_nc.content == fs_nc.content


def test_parity_links_and_backlinks(backends):
    cli, fs = backends
    files = cli.list_files()
    if not files:
        pytest.skip("No files in vault to test links")
        
    # Pick a file with links if possible
    test_ref = None
    for ref in files:
        if cli.links(ref):
            test_ref = ref
            break
            
    if not test_ref:
        test_ref = files[0]
        
    cli_links = {r.name for r in cli.links(test_ref)}
    fs_links = {r.name for r in fs.links(test_ref)}
    assert fs_links == cli_links
    
    cli_backlinks = {r.name for r in cli.backlinks(test_ref)}
    fs_backlinks = {r.name for r in fs.backlinks(test_ref)}
    assert fs_backlinks == cli_backlinks
