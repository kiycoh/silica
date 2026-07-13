import assert from "node:assert/strict";
import { test } from "node:test";

import { dispatchRpc, RPC_METHODS, type FileCacheLike, type RpcApp, type TFileLike } from "./handlers.ts";

interface FileSpec {
  content: string;
  frontmatter?: Record<string, unknown>;
  headings?: FileCacheLike["headings"];
  cache?: FileCacheLike;
}

const baseOf = (path: string) => path.replace(/\.md$/, "").split("/").pop() ?? path;

// Mutable in-memory vault so write postconditions are assertable. `files` is the
// live store; handlers mutate it through the structural RpcApp surface.
function makeApp(
  files: Record<string, FileSpec>,
  resolved: Record<string, Record<string, number>> = {},
  unresolved: Record<string, Record<string, number>> = {},
): RpcApp {
  const tfiles: Record<string, TFileLike> = {};
  const folders = new Set<string>(); // folder paths that "exist" (created via createFolder)
  for (const path of Object.keys(files)) tfiles[path] = { path, basename: baseOf(path) };
  return {
    vault: {
      getMarkdownFiles: () => Object.values(tfiles),
      cachedRead: async (f) => files[f.path].content,
      getFileByPath: (p) => tfiles[p] ?? null,
      getFolderByPath: (p) => (folders.has(p) ? { path: p } : null),
      create: async (path, content) => {
        if (files[path]) throw new Error(`already exists: ${path}`);
        const slash = path.lastIndexOf("/");
        const dir = slash > 0 ? path.slice(0, slash) : "";
        // Mirror Obsidian: vault.create ENOENTs when the parent folder is absent.
        if (dir && !folders.has(dir)) throw new Error(`ENOENT: no such file or directory, open '${path}'`);
        files[path] = { content };
        tfiles[path] = { path, basename: baseOf(path) };
        return tfiles[path];
      },
      createFolder: async (p) => {
        if (folders.has(p)) throw new Error(`already exists: ${p}`);
        const parts = p.split("/"); // recursive, like Obsidian's createFolder
        for (let i = 1; i <= parts.length; i++) folders.add(parts.slice(0, i).join("/"));
      },
      process: async (f, fn) => (files[f.path].content = fn(files[f.path].content)),
    },
    metadataCache: {
      getFileCache: (f) =>
        files[f.path].cache ?? { frontmatter: files[f.path].frontmatter, headings: files[f.path].headings },
      getFirstLinkpathDest: (linkpath) =>
        Object.values(tfiles).find((t) => t.basename.toLowerCase() === linkpath.toLowerCase()) ?? null,
      resolvedLinks: resolved,
      unresolvedLinks: unresolved,
    },
    fileManager: {
      processFrontMatter: async (f, fn) => fn((files[f.path].frontmatter ??= {})),
      renameFile: async (f, to) => {
        const slash = to.lastIndexOf("/");
        const dir = slash > 0 ? to.slice(0, slash) : "";
        // Mirror Obsidian: renameFile ENOENTs when the destination folder is absent.
        if (dir && !folders.has(dir)) throw new Error(`ENOENT: no such file or directory, rename to '${to}'`);
        files[to] = files[f.path];
        tfiles[to] = { path: to, basename: baseOf(to) };
        delete files[f.path];
        delete tfiles[f.path];
      },
      trashFile: async (f) => {
        delete files[f.path];
        delete tfiles[f.path];
      },
      generateMarkdownLink: (f) => `[[${f.basename}]]`,
    },
  };
}

const idNorm = (p: string) => p;

test("read returns path/content/size; missing file throws", async () => {
  const app = makeApp({ "A.md": { content: "hello" } });
  assert.deepEqual(await dispatchRpc(app, "read", { path: "A.md" }, idNorm), {
    path: "A.md", content: "hello", size: 5,
  });
  await assert.rejects(() => dispatchRpc(app, "read", { path: "Missing.md" }, idNorm), /file not found/);
});

test("list_files filters by folder prefix", async () => {
  const app = makeApp({ "A.md": { content: "" }, "sub/B.md": { content: "" }, "sub/C.md": { content: "" } });
  const all = (await dispatchRpc(app, "list_files", {}, idNorm)) as Array<{ name: string; path: string }>;
  assert.equal(all.length, 3);
  const sub = (await dispatchRpc(app, "list_files", { folder: "sub" }, idNorm)) as Array<{ name: string; path: string }>;
  assert.deepEqual(sub.map((r) => r.path).sort(), ["sub/B.md", "sub/C.md"]);
  assert.deepEqual(sub.map((r) => r.name).sort(), ["B", "C"]);
});

test("props_of returns frontmatter, {} when none", async () => {
  const app = makeApp({ "A.md": { content: "", frontmatter: { tags: ["x"], n: 3 } }, "B.md": { content: "" } });
  assert.deepEqual(await dispatchRpc(app, "props_of", { path: "A.md" }, idNorm), { tags: ["x"], n: 3 });
  assert.deepEqual(await dispatchRpc(app, "props_of", { path: "B.md" }, idNorm), {});
});

test("outline maps level/text/position", async () => {
  const app = makeApp({
    "A.md": {
      content: "",
      headings: [
        { level: 1, heading: "Title", position: { start: { offset: 0 } } },
        { level: 2, heading: "Sub", position: { start: { offset: 42 } } },
      ],
    },
  });
  assert.deepEqual(await dispatchRpc(app, "outline", { path: "A.md" }, idNorm), [
    { level: 1, text: "Title", position: 0 },
    { level: 2, text: "Sub", position: 42 },
  ]);
});

test("search_context returns per-file groups with 1-based trimmed matches", async () => {
  const app = makeApp({ "A.md": { content: "alpha\n  Beta here\ngamma" }, "B.md": { content: "nothing" } });
  const groups = (await dispatchRpc(app, "search_context", { query: "beta" }, idNorm)) as Array<{
    path: string; matches: Array<{ line: number; content: string }>;
  }>;
  assert.equal(groups.length, 1);
  assert.equal(groups[0].path, "A.md");
  assert.deepEqual(groups[0].matches, [{ line: 2, content: "Beta here" }]);
});

test("search_context_batch returns groups per query", async () => {
  const app = makeApp({ "A.md": { content: "cat\ndog" }, "B.md": { content: "cat" } });
  const out = (await dispatchRpc(app, "search_context_batch", { queries: ["cat", "dog"] }, idNorm)) as Record<
    string, Array<{ path: string; matches: Array<{ line: number; content: string }> }>
  >;
  assert.deepEqual(Object.keys(out).sort(), ["cat", "dog"]);
  assert.deepEqual(out.cat.map((g) => g.path).sort(), ["A.md", "B.md"]);
  assert.deepEqual(out.dog.map((g) => g.path), ["A.md"]);
  assert.deepEqual(out.dog[0].matches, [{ line: 2, content: "dog" }]);
});

test("resolved_links returns the metadataCache maps verbatim", async () => {
  const resolved = { "A.md": { "B.md": 1 } };
  const unresolved = { "A.md": { Ghost: 2 } };
  const app = makeApp({ "A.md": { content: "" }, "B.md": { content: "" } }, resolved, unresolved);
  assert.deepEqual(await dispatchRpc(app, "resolved_links", {}, idNorm), { resolved, unresolved });
});

test("mention_index matches at word boundaries, dedups per file", async () => {
  const app = makeApp({
    "Net.md": { content: "the network is not the internet" }, // "net" via network (boundary); internet's net is mid-word
    "Other.md": { content: "NET works" }, // lowercased → boundary hit
    "Twice.md": { content: "net net" }, // two boundary hits → path listed once
    "None.md": { content: "kitten basket" }, // no boundary "net"
  });
  const out = (await dispatchRpc(app, "mention_index", { titles: ["net"] }, idNorm)) as Record<string, string[]>;
  assert.deepEqual(Object.keys(out), ["net"]);
  assert.deepEqual(out.net.sort(), ["Net.md", "Other.md", "Twice.md"]);
});

test("allowlist covers reads + writes, rejects unknown", async () => {
  const app = makeApp({ "A.md": { content: "" } });
  await assert.rejects(() => dispatchRpc(app, "nope", {}, idNorm), /unknown method/);
  for (const m of ["read", "create", "overwrite", "append", "set_prop", "move", "delete", "autolink_note"]) {
    assert.equal(RPC_METHODS.has(m), true, m);
  }
  assert.equal(RPC_METHODS.has("eval"), false);
});

test("normalize is applied to path params", async () => {
  const app = makeApp({ "A.md": { content: "x" } });
  const norm = (p: string) => p.replace(/^\.\//, "");
  const r = (await dispatchRpc(app, "read", { path: "./A.md" }, norm)) as { content: string };
  assert.equal(r.content, "x");
});

// --- Writes ----------------------------------------------------------------

test("create writes content verbatim, returns {name, path}; errors if exists", async () => {
  const files: Record<string, { content: string }> = {};
  const app = makeApp(files);
  const r = await dispatchRpc(app, "create", { path: "sub/New.md", content: "hi\nthere" }, idNorm);
  assert.deepEqual(r, { name: "New", path: "sub/New.md" });
  assert.equal(files["sub/New.md"].content, "hi\nthere");
  await assert.rejects(() => dispatchRpc(app, "create", { path: "sub/New.md", content: "x" }, idNorm), /already exists/);
});

test("create round-trips a LaTeX/CRLF body verbatim (no escaping mangling)", async () => {
  const files: Record<string, { content: string }> = {};
  const app = makeApp(files);
  const body = "$$\\begin{aligned}\\nabla f = \\sum_i x_i\\end{aligned}$$\r\nline\\neq2\r\n";
  await dispatchRpc(app, "create", { path: "Math.md", content: body }, idNorm);
  const read = (await dispatchRpc(app, "read", { path: "Math.md" }, idNorm)) as { content: string; size: number };
  assert.equal(read.content, body);
  assert.equal(read.size, body.length);
});

test("create makes the parent folder first (Obsidian vault.create ENOENTs otherwise)", async () => {
  const files: Record<string, { content: string }> = {};
  const app = makeApp(files);
  // Exact bridge repro: nucleate into a folder that doesn't exist yet. Without
  // ensureFolder this ENOENTs and every note defers (cli_backend already mkdir -p's).
  const r = await dispatchRpc(app, "create", { path: "Machine Learning/Random Variable.md", content: "body" }, idNorm);
  assert.deepEqual(r, { name: "Random Variable", path: "Machine Learning/Random Variable.md" });
  assert.equal(files["Machine Learning/Random Variable.md"].content, "body");
  // A sibling reuses the now-existing folder (getFolderByPath short-circuits).
  await dispatchRpc(app, "create", { path: "Machine Learning/Sibling.md", content: "y" }, idNorm);
  assert.equal(files["Machine Learning/Sibling.md"].content, "y");
  // Nested dirs get created recursively.
  await dispatchRpc(app, "create", { path: "A/B/C/Deep.md", content: "x" }, idNorm);
  assert.equal(files["A/B/C/Deep.md"].content, "x");
});

test("create round-trips a 1MB body (no 30KB special case)", async () => {
  const files: Record<string, { content: string }> = {};
  const app = makeApp(files);
  const body = "x".repeat(1_000_000);
  await dispatchRpc(app, "create", { path: "Big.md", content: body }, idNorm);
  assert.equal(files["Big.md"].content.length, 1_000_000);
});

test("overwrite replaces in place; missing file throws", async () => {
  const app = makeApp({ "A.md": { content: "old" } });
  assert.deepEqual(await dispatchRpc(app, "overwrite", { path: "A.md", content: "new" }, idNorm), { ok: true });
  assert.equal((await dispatchRpc(app, "read", { path: "A.md" }, idNorm) as { content: string }).content, "new");
  await assert.rejects(() => dispatchRpc(app, "overwrite", { path: "Gone.md", content: "x" }, idNorm), /file not found/);
});

test("append concatenates the fragment at the end", async () => {
  const app = makeApp({ "A.md": { content: "head" } });
  await dispatchRpc(app, "append", { path: "A.md", content: "\ntail" }, idNorm);
  assert.equal((await dispatchRpc(app, "read", { path: "A.md" }, idNorm) as { content: string }).content, "head\ntail");
});

test("set_prop sets frontmatter[name]=value, preserving value type; ignores `type`", async () => {
  const app = makeApp({ "A.md": { content: "", frontmatter: { keep: 1 } } });
  await dispatchRpc(app, "set_prop", { path: "A.md", name: "tags", value: ["x", "y"], type: "list" }, idNorm);
  await dispatchRpc(app, "set_prop", { path: "A.md", name: "n", value: 42, type: "number" }, idNorm);
  assert.deepEqual(await dispatchRpc(app, "props_of", { path: "A.md" }, idNorm), { keep: 1, tags: ["x", "y"], n: 42 });
});

test("move renames: dest readable, source gone", async () => {
  const app = makeApp({ "A.md": { content: "body" } });
  assert.deepEqual(await dispatchRpc(app, "move", { path: "A.md", to: "sub/B.md" }, idNorm), { ok: true });
  assert.equal((await dispatchRpc(app, "read", { path: "sub/B.md" }, idNorm) as { content: string }).content, "body");
  await assert.rejects(() => dispatchRpc(app, "read", { path: "A.md" }, idNorm), /file not found/);
});

test("move creates the destination folder first (renameFile ENOENTs otherwise)", async () => {
  const app = makeApp({ "Note.md": { content: "body" } });
  // Move into a not-yet-existing (nested) folder — same parent-ENOENT class as create.
  assert.deepEqual(await dispatchRpc(app, "move", { path: "Note.md", to: "Archive/2026/Note.md" }, idNorm), { ok: true });
  assert.equal((await dispatchRpc(app, "read", { path: "Archive/2026/Note.md" }, idNorm) as { content: string }).content, "body");
  await assert.rejects(() => dispatchRpc(app, "read", { path: "Note.md" }, idNorm), /file not found/);
});

test("move param is `to` (not dest); missing source throws", async () => {
  const app = makeApp({ "A.md": { content: "" } });
  await assert.rejects(() => dispatchRpc(app, "move", { path: "Gone.md", to: "X.md" }, idNorm), /file not found/);
});

test("delete removes the path; missing source throws", async () => {
  const app = makeApp({ "A.md": { content: "" } });
  assert.deepEqual(await dispatchRpc(app, "delete", { path: "A.md" }, idNorm), { ok: true });
  await assert.rejects(() => dispatchRpc(app, "read", { path: "A.md" }, idNorm), /file not found/);
  await assert.rejects(() => dispatchRpc(app, "delete", { path: "A.md" }, idNorm), /file not found/);
});

test("autolink_note wraps first unlinked mention, longest-first, returns titles", async () => {
  const app = makeApp({
    "Note.md": { content: "I study deep learning and learning theory." },
    "Deep Learning.md": { content: "" },
    "Learning.md": { content: "" },
  });
  const added = (await dispatchRpc(app, "autolink_note", { path: "Note.md", candidates: null }, idNorm)) as string[];
  // "deep learning" matched longest-first, so the standalone "learning" later still links.
  assert.deepEqual(added.sort(), ["Deep Learning", "Learning"]);
  const body = (await dispatchRpc(app, "read", { path: "Note.md" }, idNorm) as { content: string }).content;
  assert.equal(body, "I study [[Deep Learning]] and [[Learning]] theory.");
});

test("autolink_note skips self, code spans, and unresolvable titles", async () => {
  const app = makeApp({
    "Alpha.md": { content: "Alpha talks about Beta but `Beta` in code stays, and Ghost is unknown." },
    "Beta.md": { content: "" },
  });
  const added = (await dispatchRpc(app, "autolink_note", { path: "Alpha.md", candidates: ["Alpha", "Beta", "Ghost"] }, idNorm)) as string[];
  assert.deepEqual(added, ["Beta"]); // Alpha=self, Ghost=unresolvable, `Beta` in backticks skipped
  const body = (await dispatchRpc(app, "read", { path: "Alpha.md" }, idNorm) as { content: string }).content;
  assert.equal(body, "Alpha talks about [[Beta]] but `Beta` in code stays, and Ghost is unknown.");
});

test("autolink_note skips a title already linked, and is idempotent", async () => {
  const app = makeApp({
    "N.md": {
      content: "See [[Beta]] and Beta again.",
      cache: { links: [{ link: "Beta", position: { start: { offset: 4 }, end: { offset: 12 } } }] },
    },
    "Beta.md": { content: "" },
  });
  const added = (await dispatchRpc(app, "autolink_note", { path: "N.md", candidates: ["Beta"] }, idNorm)) as string[];
  assert.deepEqual(added, []); // already linked → nothing to add
  const body = (await dispatchRpc(app, "read", { path: "N.md" }, idNorm) as { content: string }).content;
  assert.equal(body, "See [[Beta]] and Beta again.");
});

test("autolink_note on a missing file returns [] (not an error)", async () => {
  const app = makeApp({ "A.md": { content: "" } });
  const added = await dispatchRpc(app, "autolink_note", { path: "Gone.md", candidates: null }, idNorm);
  assert.deepEqual(added, []);
});

test("autolink_note empty candidates → no links", async () => {
  const app = makeApp({ "N.md": { content: "mentions Beta" }, "Beta.md": { content: "" } });
  const added = await dispatchRpc(app, "autolink_note", { path: "N.md", candidates: [] }, idNorm);
  assert.deepEqual(added, []);
});
