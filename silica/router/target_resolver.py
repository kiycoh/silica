import os
from pathlib import Path

def resolve_ingestion_target(target_str: str, vault_path: str) -> tuple[str, str]:
    """Resolve target string to an ingestion mode and its resolved vault-relative path.

    Modes:
      - FILE_APPEND_MODE: Target is a single note file (for MOC/log append).
      - ATOMIC_FOLDER_MODE: Target is a directory (for atomic notes creation).
    """
    if not target_str:
        return "ATOMIC_FOLDER_MODE", ""

    # Normalize target string
    target_str_clean = target_str.replace("\\", "/").strip("/")
    
    # Path inside the vault
    vault_p = Path(vault_path)
    target_p = vault_p / target_str_clean
    
    # Case 1: ends with .md or exists as a file
    if target_str_clean.endswith(".md") or target_p.is_file():
        # Ensure it has .md extension
        resolved = target_str_clean if target_str_clean.endswith(".md") else f"{target_str_clean}.md"
        return "FILE_APPEND_MODE", resolved
        
    # Check if target_str + ".md" exists as a file
    target_p_md = vault_p / f"{target_str_clean}.md"
    if target_p_md.is_file():
        return "FILE_APPEND_MODE", f"{target_str_clean}.md"
        
    # Case 2: exists as a directory
    if target_p.is_dir():
        return "ATOMIC_FOLDER_MODE", target_str_clean
        
    # Case 3: Default mode when neither exists.
    return "ATOMIC_FOLDER_MODE", target_str_clean
