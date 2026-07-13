// RPC handlers — typed ports of the JS inlined in Silica's cli_backend. Result
// shapes and write postconditions match what ws_backend.py consumes (PROTOCOL.md
// §Reads/§Writes). Depends only on a structural `RpcApp`, so it runs headless
// under `node --test` against a fake vault; main.ts passes the real Obsidian
// `app` plus `normalizePath`.

export interface TFileLike {
  path: string;
  basename: string;
}

interface Pos {
  start: { offset: number };
  end?: { offset: number };
}

export interface FileCacheLike {
  frontmatter?: Record<string, unknown>;
  headings?: Array<{ level: number; heading: string; position: Pos }>;
  links?: Array<{ link?: string; displayText?: string; position?: Pos }>;
  embeds?: Array<{ position?: Pos }>;
  sections?: Array<{ type: string; position?: Pos }>;
  frontmatterPosition?: Pos;
}

export interface RpcApp {
  vault: {
    getMarkdownFiles(): TFileLike[];
    cachedRead(file: TFileLike): Promise<string>;
    getFileByPath(path: string): TFileLike | null;
    getFolderByPath(path: string): unknown;
    create(path: string, content: string): Promise<TFileLike>;
    createFolder(path: string): Promise<unknown>;
    process(file: TFileLike, fn: (data: string) => string): Promise<string>;
  };
  metadataCache: {
    getFileCache(file: TFileLike): FileCacheLike | null;
    getFirstLinkpathDest(linkpath: string, sourcePath: string): TFileLike | null;
    resolvedLinks: Record<string, Record<string, number>>;
    unresolvedLinks: Record<string, Record<string, number>>;
  };
  fileManager: {
    processFrontMatter(file: TFileLike, fn: (fm: Record<string, unknown>) => void): Promise<void>;
    renameFile(file: TFileLike, newPath: string): Promise<void>;
    trashFile(file: TFileLike): Promise<void>;
    generateMarkdownLink(file: TFileLike, sourcePath: string): string;
  };
}

type Params = Record<string, unknown>;
type Normalize = (p: string) => string;
type Handler = (app: RpcApp, p: Params, norm: Normalize) => Promise<unknown>;

interface SearchGroup {
  path: string;
  name: string;
  matches: Array<{ line: number; content: string }>;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : String(v ?? "");
}

function asStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.map(str) : [];
}

function fileOrThrow(app: RpcApp, path: string, norm: Normalize): TFileLike {
  const f = app.vault.getFileByPath(norm(path));
  if (!f) throw new Error(`file not found: ${path}`);
  return f;
}

/** One cachedRead sweep, every query → per-file groups (mirrors search_context_batch JS). */
async function sweepSearch(app: RpcApp, queries: string[]): Promise<Record<string, SearchGroup[]>> {
  const out: Record<string, SearchGroup[]> = {};
  const lower = queries.map((q) => q.toLowerCase());
  for (const q of queries) out[q] = [];
  await Promise.all(
    app.vault.getMarkdownFiles().map(async (file) => {
      let content: string;
      try {
        content = await app.vault.cachedRead(file);
      } catch {
        return;
      }
      const lines = content.split("\n");
      const linesLower = lines.map((l) => l.toLowerCase());
      for (let k = 0; k < queries.length; k++) {
        const matches: Array<{ line: number; content: string }> = [];
        for (let i = 0; i < linesLower.length; i++) {
          if (linesLower[i].includes(lower[k])) matches.push({ line: i + 1, content: lines[i].trim() });
        }
        if (matches.length) out[queries[k]].push({ path: file.path, name: file.basename, matches });
      }
    }),
  );
  return out;
}

/** Title-mention inverted index: {title_lower: [paths]}. Titles occurring in a
 * body as a substring beginning at a word boundary. Direct port of
 * cli_backend._build_mention_index (the JS mirror of base.mentions_in). */
async function buildMentionIndex(app: RpcApp, titles: string[]): Promise<Record<string, string[]>> {
  const TERM = String.fromCharCode(0); // NUL sentinel — cannot appear in a title
  // ponytail: char-trie typed as any — faithful port of the proven cli JS.
  const trie: Record<string, any> = {};
  for (const t of titles) {
    if (t.length < 2) continue;
    let node = trie;
    for (const ch of t) node = node[ch] = node[ch] || {};
    node[TERM] = t;
  }
  const isWord = (c: string) => (c >= "a" && c <= "z") || (c >= "0" && c <= "9");
  const mentions: Record<string, string[]> = {};
  await Promise.all(
    app.vault.getMarkdownFiles().map(async (file) => {
      let s: string;
      try {
        s = (await app.vault.cachedRead(file)).toLowerCase();
      } catch {
        return;
      }
      const n = s.length;
      const seen = new Set<string>();
      for (let i = 0; i < n; i++) {
        if (!isWord(s[i])) continue;
        if (i && isWord(s[i - 1])) continue; // start walks at word boundaries only
        let node = trie;
        for (let j = i; j < n; j++) {
          node = node[s[j]];
          if (node === undefined) break;
          const t: string | undefined = node[TERM];
          if (t !== undefined && !seen.has(t)) {
            seen.add(t);
            (mentions[t] ??= []).push(file.path);
          }
        }
      }
    }),
  );
  return mentions;
}

/** All linkable vault titles, ambiguous basenames dropped (mirrors
 * kernel.autolink.build_title_index). Used when autolink_note gets a null
 * `candidates` — cli_backend fills this in Python-side, but over the WS bridge
 * the plugin receives raw null and must build it here. */
function allVaultTitles(app: RpcApp): string[] {
  const counts = new Map<string, number>();
  for (const f of app.vault.getMarkdownFiles()) counts.set(f.basename, (counts.get(f.basename) ?? 0) + 1);
  return [...counts].filter(([, c]) => c === 1).map(([n]) => n);
}

/** Typed port of cli_backend._AUTOLINK_JS. Skip-mask from getFileCache (code
 * sections, frontmatter, existing links/embeds/headings) plus inline-code/math
 * regexes; match resolvable titles longest-first at word boundaries; wrap via
 * generateMarkdownLink; one atomic vault.process. Returns titles actually linked. */
async function autolinkNote(app: RpcApp, path: string, candidatesRaw: unknown): Promise<string[]> {
  const file = app.vault.getFileByPath(path);
  if (!file) return []; // faithful: missing file → no links (JS returns {added: []})
  const selfLower = file.basename.toLowerCase();

  const candidates = candidatesRaw == null ? allVaultTitles(app) : asStringArray(candidatesRaw);
  const cache = app.metadataCache.getFileCache(file) ?? {};

  // Already-linked set: link target, its basename, and any alias — so an
  // existing [[Path/Title|Alias]] still suppresses re-linking "Title".
  const linked = new Set<string>();
  for (const l of cache.links ?? []) {
    const tgt = (l.link ?? "").split("|")[0].trim();
    if (tgt) {
      linked.add(tgt.toLowerCase());
      linked.add((tgt.split("/").pop() ?? "").replace(/\.md$/, "").toLowerCase());
    }
    if (l.displayText) linked.add(l.displayText.toLowerCase());
  }

  const titles = candidates.filter((t) => t.length >= 2).sort((a, b) => b.length - a.length);
  const added: string[] = [];

  await app.vault.process(file, (cur) => {
    let body = cur;
    const mask = new Array<boolean>(body.length).fill(false);
    const markPos = (p?: Pos) => {
      if (p && p.start && p.end) {
        for (let i = p.start.offset; i < p.end.offset; i++) if (i >= 0 && i < mask.length) mask[i] = true;
      }
    };
    for (const s of cache.sections ?? []) if (s.type === "code") markPos(s.position);
    if (cache.frontmatterPosition) markPos(cache.frontmatterPosition);
    for (const l of cache.links ?? []) markPos(l.position);
    for (const e of cache.embeds ?? []) markPos(e.position);
    for (const h of cache.headings ?? []) markPos(h.position);
    const markRe = (re: RegExp) => {
      let m: RegExpExecArray | null;
      while ((m = re.exec(body)) !== null) for (let i = m.index; i < m.index + m[0].length; i++) mask[i] = true;
    };
    markRe(/\$\$[^]*?\$\$/g); // display math (multi-line)
    markRe(/`[^`\n]+`/g); // inline code
    markRe(/\$[^$\n]+\$/g); // inline math

    added.length = 0; // idempotent if vault.process retries the transformer
    for (const title of titles) {
      const tl = title.toLowerCase();
      if (tl === selfLower || linked.has(tl)) continue;
      const dest = app.metadataCache.getFirstLinkpathDest(title, path);
      if (!dest) continue;
      const reg = new RegExp("(?<![\\w\\[])" + title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?![\\w\\]])", "ig");
      let found = -1;
      let mEnd = -1;
      let m: RegExpExecArray | null;
      while ((m = reg.exec(body)) !== null) {
        let clean = true;
        for (let i = m.index; i < m.index + m[0].length; i++) if (mask[i]) { clean = false; break; }
        if (clean) { found = m.index; mEnd = m.index + m[0].length; break; }
      }
      if (found < 0) continue;
      const md = app.fileManager.generateMarkdownLink(dest, path);
      body = body.slice(0, found) + md + body.slice(mEnd);
      const insert = new Array<boolean>(md.length).fill(true);
      mask.splice(found, mEnd - found, ...insert);
      added.push(title);
      linked.add(tl);
    }
    return body;
  });
  return added;
}

/** Obsidian's vault.create ENOENTs when the parent folder is missing — unlike
 * fs_backend (Path.mkdir parents=True) and cli_backend (_ensure_dest_dir). Create
 * it first, else every note nucleated into a not-yet-existing dir gets deferred. */
async function ensureFolder(app: RpcApp, filePath: string): Promise<void> {
  const slash = filePath.lastIndexOf("/");
  if (slash <= 0) return; // root-level note — no parent to create
  const dir = filePath.slice(0, slash);
  if (app.vault.getFolderByPath(dir)) return;
  try {
    await app.vault.createFolder(dir); // recursive in current Obsidian
  } catch {
    // Already exists (concurrent create / cache lag) — the postcondition holds.
  }
}

const HANDLERS: Record<string, Handler> = {
  async read(app, p, norm) {
    const f = fileOrThrow(app, str(p.path), norm);
    const content = await app.vault.cachedRead(f);
    return { path: f.path, content, size: content.length };
  },
  async list_files(app, p, norm) {
    const folder = str(p.folder ?? "");
    const prefix = folder ? norm(folder).replace(/\/+$/, "") + "/" : "";
    return app.vault
      .getMarkdownFiles()
      .filter((f) => !prefix || f.path.startsWith(prefix))
      .map((f) => ({ name: f.basename, path: f.path }));
  },
  async props_of(app, p, norm) {
    const f = fileOrThrow(app, str(p.path), norm);
    return app.metadataCache.getFileCache(f)?.frontmatter ?? {};
  },
  async outline(app, p, norm) {
    const f = fileOrThrow(app, str(p.path), norm);
    const headings = app.metadataCache.getFileCache(f)?.headings ?? [];
    return headings.map((h) => ({ level: h.level, text: h.heading, position: h.position.start.offset }));
  },
  async search_context(app, p) {
    const query = str(p.query);
    return (await sweepSearch(app, [query]))[query] ?? [];
  },
  async search_context_batch(app, p) {
    return sweepSearch(app, asStringArray(p.queries));
  },
  async resolved_links(app) {
    return { resolved: app.metadataCache.resolvedLinks, unresolved: app.metadataCache.unresolvedLinks };
  },
  async mention_index(app, p) {
    return buildMentionIndex(app, asStringArray(p.titles));
  },

  // --- Writes (graph-safe) — the reply IS the settle (PROTOCOL §2.4) ---------
  async create(app, p, norm) {
    // vault.create throws if the path already exists — that IS the postcondition.
    const path = norm(str(p.path));
    await ensureFolder(app, path); // parent dir must exist first (see helper)
    const f = await app.vault.create(path, str(p.content));
    return { name: f.basename, path: f.path };
  },
  async overwrite(app, p, norm) {
    const f = fileOrThrow(app, str(p.path), norm);
    await app.vault.process(f, () => str(p.content)); // in place — history/block-refs kept
    return { ok: true };
  },
  async append(app, p, norm) {
    const f = fileOrThrow(app, str(p.path), norm);
    await app.vault.process(f, (cur) => cur + str(p.content));
    return { ok: true };
  },
  async set_prop(app, p, norm) {
    const f = fileOrThrow(app, str(p.path), norm);
    const name = str(p.name);
    // `type` governs only the CLI fallback (cli_backend); the eval path takes
    // value as-is (already JSON-typed by ws_backend). Deliberately ignored here.
    await app.fileManager.processFrontMatter(f, (fm) => { fm[name] = p.value; });
    return { ok: true };
  },
  async move(app, p, norm) {
    const f = fileOrThrow(app, str(p.path), norm);
    const to = norm(str(p.to));
    await ensureFolder(app, to); // dest parent must exist first — renameFile ENOENTs otherwise
    await app.fileManager.renameFile(f, to); // Obsidian rewrites incoming wikilinks
    return { ok: true };
  },
  async delete(app, p, norm) {
    const f = fileOrThrow(app, str(p.path), norm);
    await app.fileManager.trashFile(f); // recoverable, not vault.delete
    return { ok: true };
  },
  async autolink_note(app, p, norm) {
    return autolinkNote(app, norm(str(p.path)), p.candidates);
  },
};

/** Allowlist — the plugin dispatches only these methods; anything else is refused. */
export const RPC_METHODS: ReadonlySet<string> = new Set(Object.keys(HANDLERS));

export async function dispatchRpc(
  app: RpcApp,
  method: string,
  params: Params,
  normalize: Normalize,
): Promise<unknown> {
  const h = HANDLERS[method];
  if (!h) throw new Error(`unknown method: ${method}`);
  return h(app, params, normalize);
}
