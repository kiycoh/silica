# Content Search and Note Tools for Obsidian Vault

## Available Tools

When querying content and managing notes in the vault, use the `obsidian-cli` command-line tool (run via terminal/execution primitives) or interact with the markdown files directly on disk.

### 1. `obsidian-cli search-content` (Primary Content Search)

Search inside note contents for terms, phrases, or keywords. This is the primary method to check for semantic concept collisions.

**Command structure:**
```bash
obsidian-cli search-content "<search query>"
```

**Recommended usage for concept collision checking:**
```bash
obsidian-cli search-content "concept keyword"
```
This returns snippets and line numbers of occurrences within vault notes.

### 2. `obsidian-cli search` (Note Name Search)

Search for notes by their file/note names.

**Command structure:**
```bash
obsidian-cli search "<note name query>"
```
Use this when verifying if a note with a specific title/concept name already exists in the vault.

### 3. Direct File Reading & System Tools

Since an Obsidian vault is a normal folder on disk, you can inspect note content directly using standard workspace tools:
- **`view_file`**: Read a specific markdown file to review its full content.
- **Terminal `cat`**: View file content in shell scripts when avoiding tool-call deduplication.
  ```bash
  cat "/path/to/vault/Folder/Note.md"
  ```

---

## Modifying & Managing Notes

For note creation and editing, prefer direct file-writing tools, but use `obsidian-cli` when structural vault operations are needed.

### 1. Direct File Editing (Preferred for Injector)
Write and modify notes directly on the disk using:
- **`write_to_file`**: Create a new Spoke note.
- **`replace_file_content`** / **`multi_replace_file_content`**: Patch/enrich existing notes.

### 2. `obsidian-cli move` (Safe Refactoring)
Use `obsidian-cli move` when renaming or moving files, as it automatically updates all `[[wikilinks]]` pointing to the note across the vault:
```bash
obsidian-cli move "old/path/note" "new/path/note"
```

### 3. `obsidian-cli create` (Alternative Creation)
Create a note and open it in the Obsidian client:
```bash
obsidian-cli create "Folder/New Note" --content "body content" --open
```

---

## Common Search Patterns

### Finding by English Term (with Italian translation)
Run searches for both language variations to catch all matches:
```bash
obsidian-cli search-content "packet"
obsidian-cli search-content "pacchetto"
```

### Case-Insensitive Search
`obsidian-cli` performs case-insensitive content searches automatically:
```bash
obsidian-cli search-content "caching"
```

---

## Pitfalls to Avoid

1. **Avoid guessing file paths**: Always use paths returned by `obsidian-cli search` or `obsidian-cli search-content`.
2. **Handle spaces in paths**: Ensure vault paths or note names containing spaces or quotes are properly escaped or quoted in terminal commands.
3. **No dot-folders**: Do not create or look for notes in hidden dot-folders (like `.obsidian/`) via `obsidian-cli`.
4. **Do not run blind wildcard searches**: Avoid calling `search_files` with wildcard patterns like `*` without a specific path prefix. This will retrieve thousands of Git objects and temporary metadata files, polluting your context window. To discover files inside the inbox or target folders, use terminal tools like `find "/path/to/dir" -maxdepth 2 -not -path '*/.*' -name '*.md'` or write a clean directory scanner in Python.


---

## Best Practice Workflow

1. Extract unique concepts from the source file.
2. For each concept, check if the note already exists by searching note names:
   ```bash
   obsidian-cli search "Concept Name"
   ```
3. Check for mentions inside existing notes using:
   ```bash
   obsidian-cli search-content "Concept Name"
   ```
4. Decide to **CREATE** (new note), **ENRICH** (patch existing), or **SKIP** based on the results.