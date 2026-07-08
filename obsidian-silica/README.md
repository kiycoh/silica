# Silica Bridge (Obsidian plugin)

Connects your vault to a running `silica connect` session over a **loopback-only**
WebSocket (`ws://127.0.0.1`). Silica then reads and edits notes through Obsidian's
own APIs, and you chat with it from a side panel.

## Network use

This plugin opens a WebSocket to `127.0.0.1` (localhost) **only** — never to a
remote host. It talks solely to a `silica connect` process you start yourself on
the same machine. The port and a shared token are exchanged via
`<vault>/.obsidian/silica-bridge.json` (written by `silica connect`, mode `0600`).
No data leaves your machine through this plugin.

## Status

Feature-complete for v1:

- Connection lifecycle: handshake, reconnect-with-backoff, status panel, settings.
- Vault RPC surface — reads (`read`/`list_files`/`props_of`/`outline`/
  `search_context`/`resolved_links`/`mention_index`) and graph-safe writes
  (`create`/`overwrite`/`append`/`set_prop`/`move`/`delete`/`autolink_note`).
- Chat panel: message log + input over the bridge's chat channel; assistant
  replies rendered as markdown (clickable wikilinks).

The wire contract is `PROTOCOL.md` (kept in lockstep with the Python side).

## Use

1. In the vault, run `silica connect` (needs the `[connect]` extra).
2. Open the **Silica bridge** panel (ribbon icon or command palette). It shows
   `connected` once the handshake completes.
3. Type in the panel to chat; Silica reads and edits notes through Obsidian's own
   APIs, so the graph and file explorer update live.

## Develop

```sh
npm install
npm run dev      # esbuild watch → main.js
npm test         # node --test — handshake / reconnect state machine
npm run build    # tsc typecheck (strict) + production bundle
```

Point Obsidian at the build: symlink or copy this folder into
`<throwaway-vault>/.obsidian/plugins/silica-bridge/` (needs `manifest.json` and a
built `main.js`), then enable **Silica Bridge** in *Settings → Community plugins*.
