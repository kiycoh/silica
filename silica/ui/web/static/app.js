// Vanilla client: POST /chat returns text/event-stream, read incrementally via
// the body's ReadableStream (not EventSource — that only does GET).
const $ = (s) => document.querySelector(s);
const log = $("#log");
const input = $("#input");
const stopBtn = $("#stop");

let streaming = false;
let activeTab = "chat";

function bubble(role) {
  const el = document.createElement("div");
  el.className = "msg " + (role === "user" ? "user" : "silica");
  el.innerHTML = `<div class="role">${role === "user" ? "you" : "silica"}</div><div class="body"></div>`;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el.querySelector(".body");
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// Hover-revealed "copy" button in a message body's corner. getText() is called
// at click time so live turns can hand back their accumulated raw markdown.
function addCopyBtn(bodyEl, getText) {
  const b = document.createElement("button");
  b.className = "copy-btn";
  b.type = "button";
  b.textContent = "copy";
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(getText()); b.textContent = "copied"; }
    catch { b.textContent = "failed"; }
    setTimeout(() => (b.textContent = "copy"), 1200);
  });
  bodyEl.appendChild(b);
}

// ponytail: lazy live markdown for the streaming turn — headings, bold, italic,
// inline + fenced code, bullet/ordered lists, links. Re-parses the whole segment
// on every delta (O(n²) over the turn, fine at KB scale; parse from the last
// block boundary if very long turns ever stutter). The server re-renders
// the canonical answer (wikilinks, callouts, mermaid) on `done` for uninterrupted
// turns; swap in a vendored parser if full CommonMark is ever needed here.
function mdLite(src) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const inline = (t) =>
    esc(t)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*\n]+?)\*/g, "<em>$1</em>")
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2">$1</a>');
  const lines = src.split("\n");
  const out = [];
  let i = 0, list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const isBlock = (l) => /^```|^#{1,6}\s|^\s*[-*]\s|^\s*\d+\.\s/.test(l);
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { closeList(); i++; continue; }
    if (/^```/.test(line)) {
      closeList();
      const buf = []; i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++; // closing fence (or EOF while still streaming)
      out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { closeList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }
    const item = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)$/);
    if (item) {
      const want = /^\s*\d/.test(line) ? "ol" : "ul";
      if (list !== want) { closeList(); out.push(`<${want}>`); list = want; }
      out.push(`<li>${inline(item[1])}</li>`); i++; continue;
    }
    closeList();
    const para = [];
    while (i < lines.length && lines[i].trim() && !isBlock(lines[i])) para.push(lines[i++]);
    out.push(`<p>${para.map(inline).join("<br>")}</p>`);
  }
  closeList();
  return out.join("");
}

function fmtTokens(n) {
  n = Number(n) || 0;
  return n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
}
function setCtxTokens(used, max) {
  max = Number(max) || 0;
  $("#ctx-tokens").textContent = max ? `CTX ${fmtTokens(used)}/${fmtTokens(max)}` : "";
}

async function runTurn(fetchPromise) {
  if (streaming) return;
  streaming = true;
  stopBtn.hidden = false;
  const body = bubble("silica");
  // flow = thinking blocks, tool groups and text segments interleaved in arrival
  // order, so the transcript reads chronologically: think, tools, think, tools,
  // text… (Claude-style). In this agent the connective tissue between tool calls
  // is *thinking*, so it must interleave too or tools pile into one group.
  const flow = document.createElement("div");
  body.appendChild(flow);

  // The live iridescent caret is ONE physical element, re-parented onto
  // whatever is streaming right now (thinking body / tool group / text tail).
  const caret = document.createElement("span");
  caret.className = "caret";
  caret.textContent = "▍";

  const toolEls = {};
  const texts = [];    // every text segment { el, raw }, for the copy button
  const touched = new Set(); // notes referenced by tools this turn → sources footer
  let curText = null;   // open markdown segment { el, raw }
  let curTools = null;  // open group of consecutive tools
  let curThink = null;  // open thinking block { details, body, raw }
  let segments = 0;     // text runs so far; an uninterrupted one upgrades to server html

  // Opening one segment kind closes the other two; a thinking block collapses
  // as it closes (it stays open only while it is the live tail).
  function close(keep) {
    if (keep !== "text") curText = null;
    if (keep !== "tools") curTools = null;
    if (keep !== "think" && curThink) { curThink.details.open = false; curThink = null; }
  }
  function thinkSeg() {
    if (curThink) return curThink;
    close("think");
    const details = document.createElement("details");
    details.className = "thinking";
    details.open = true;
    details.innerHTML = `<summary>thinking</summary><div class="thinking-body"></div>`;
    flow.appendChild(details);
    return (curThink = { details, body: details.querySelector(".thinking-body"), raw: "" });
  }
  function textSeg() {
    if (curText) return curText;
    close("text");
    const el = document.createElement("div");
    el.className = "stream-text";
    flow.appendChild(el);
    curText = { el, raw: "" };
    texts.push(curText);
    segments++;
    return curText;
  }
  function toolsGroup() {
    if (curTools) return curTools;
    close("tools");
    const g = document.createElement("div");
    g.className = "tools";
    flow.appendChild(g);
    return (curTools = g);
  }
  const flowMsg = (s) => { const d = document.createElement("div"); d.className = "stream-text"; d.textContent = s; flow.appendChild(d); };

  try {
    const resp = await fetchPromise;
    if (resp.status === 409) { flowMsg("(a turn is already in progress)"); return; }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        handle(JSON.parse(line.slice(6)));
      }
    }
  } catch (e) {
    flowMsg("error: " + e);
    peekError(String(e));
  } finally {
    streaming = false;
    stopBtn.hidden = true;
    caret.remove(); // no-op if a rerender already detached it
    freezePeek(); // done or aborted — stop mirroring, keep the preview up
    if (curThink) curThink.details.open = false; // aborted mid-thought — still collapse
    if (touched.size) {
      const s = document.createElement("div");
      s.className = "sources";
      s.innerHTML = '<span class="sources-label">sources</span>';
      for (const ref of touched) {
        const c = document.createElement("span");
        c.className = "note-link"; // reuses the delegated click → note drawer
        c.dataset.path = ref;
        c.textContent = ref.split("/").pop().replace(/\.md$/, "");
        s.appendChild(c);
      }
      flow.appendChild(s);
    }
    const answer = texts.map((t) => t.raw).join("\n\n").trim();
    if (answer) addCopyBtn(body, () => answer);
    loadSessions(); // turn saved server-side — refresh titles/order
    loadVaultInfo(); // a turn may have written notes — refresh stats + tree
    graphStale = true; // a turn may have written notes — rebuild next graph view
  }

  function handle(ev) {
    if (ev.type === "delta" && ev.kind === "reasoning") {
      const th = thinkSeg();
      th.raw += ev.text;
      th.body.textContent = th.raw;
      th.body.appendChild(caret);
      th.body.scrollTop = th.body.scrollHeight; // follow the caret in the capped box
    } else if (ev.type === "delta" && ev.kind === "text") {
      const seg = textSeg();
      seg.raw += ev.text;
      seg.el.innerHTML = mdLite(seg.raw);
      (seg.el.lastElementChild || seg.el).appendChild(caret); // inline at the text tail
      peekDelta(ev.text);
    } else if (ev.type === "tool_start") {
      const t = document.createElement("div");
      t.className = "tool";
      t.textContent = "» " + ev.name + " …";
      toolsGroup().appendChild(t);
      curTools.appendChild(caret);
      toolEls[ev.id] = t;
      (ev.notes || []).forEach((n) => touched.add(n));
    } else if (ev.type === "tool_done") {
      const t = toolEls[ev.id];
      if (t) { t.className = "tool done"; t.textContent = "✓ " + ev.name; }
    } else if (ev.type === "tool_error") {
      const t = toolEls[ev.id];
      if (t) { t.className = "tool error"; t.textContent = "✗ " + ev.name + " — " + ev.error; }
    } else if (ev.type === "batch") {
      const t = document.createElement("div");
      t.className = "tool";
      t.textContent = "» " + ev.kind + " · " + ev.label;
      toolsGroup().appendChild(t);
      curTools.appendChild(caret);
    } else if (ev.type === "done") {
      // Uninterrupted answer (no tool split the text) → upgrade the live md to the
      // canonical server render (wikilinks, callouts, mermaid). Interleaved turns
      // keep their live segments; they render canonically on the next reload.
      if (segments === 0 && (ev.html || ev.answer)) {
        const seg = textSeg();
        seg.raw = ev.answer || ""; // keep the copy button fed on no-delta turns
        seg.el.innerHTML = ev.html || escapeHtml(ev.answer || "");
      } else if (segments === 1 && curText && (ev.html || ev.answer)) {
        curText.el.innerHTML = ev.html || escapeHtml(ev.answer || "");
      }
      close(""); // collapse any open thinking, end all segments
      setCtxTokens(ev.context_tokens, ev.max_context_tokens);
      peekDone(ev); // card gets the canonical OFM render
    } else if (ev.type === "error") {
      close("");
      peekError(ev.error);
      const t = document.createElement("div");
      t.className = "tool error";
      t.textContent = "error: " + ev.error;
      flow.appendChild(t);
    }
    log.scrollTop = log.scrollHeight;
  }
}

function send(text) {
  if (!text.trim() || streaming) return;
  bubble("user").textContent = text;
  const find = text.trim().match(/^\/find\s*(.*)$/);
  if (find) { runFind(find[1]); return; }
  runTurn(fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }));
}

// /find bypasses the agent entirely — same "direct tool, no LLM" pattern as
// the /graph and /map tabs, just rendered inline as a result bubble.
async function runFind(rest) {
  const body = bubble("silica");
  // dock-launched /find: mirror the result bubble into the card (no SSE stream
  // here, so the peek would otherwise sit at "thinking" forever)
  const mirror = () => { if (peek) { peek.body.innerHTML = body.innerHTML; freezePeek(); } };
  let k = 5;
  const tokens = [];
  for (const part of rest.trim().split(/\s+/)) {
    const m = part.match(/^--k=(\d+)$/);
    if (m) k = parseInt(m[1], 10);
    else if (part) tokens.push(part);
  }
  const query = tokens.join(" ");
  if (!query) { body.textContent = "usage: /find <query> [--k=N]"; mirror(); return; }
  body.textContent = "searching…";
  try {
    const r = await fetch("/find?q=" + encodeURIComponent(query) + "&k=" + k);
    body.innerHTML = await r.text();
  } catch (e) {
    body.textContent = "error: " + e;
  }
  mirror();
}

// --- composer ---------------------------------------------------------------
function autoGrow(el) {
  el.style.height = "auto";
  const border = el.offsetHeight - el.clientHeight; // box-sizing: border-box
  el.style.height = (el.scrollHeight + border) + "px"; // clamped visually by CSS max-height
}
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const t = input.value;
  input.value = "";
  autoGrow(input);
  if (staged.length) nucleateStaged(t); // files attached: upload + act on them together
  else send(t);
});
let allCommands = [];
let filteredCommands = [];
let cmdSelIdx = -1;

fetch("/commands").then(r => r.json()).then(data => allCommands = data || []).catch(() => {});

function renderCommands(q) {
  const box = $("#commands");
  if (!q.startsWith("/")) {
    box.hidden = true;
    return;
  }
  const search = q.substring(1).toLowerCase();
  
  filteredCommands = allCommands.map(cmd => {
    let score = 0;
    const name = cmd.name.substring(1).toLowerCase();
    if (name === search) score = 10;
    else if (name.startsWith(search)) score = 5;
    else if (name.includes(search)) score = 3;
    else {
      let i = 0;
      let matched = true;
      for (const c of search) {
        i = name.indexOf(c, i);
        if (i === -1) { matched = false; break; }
        i++;
      }
      if (matched && search.length > 0) score = 1;
    }
    return { cmd, score };
  }).filter(x => x.score > 0).sort((a, b) => b.score - a.score || a.cmd.name.localeCompare(b.cmd.name)).map(x => x.cmd);

  if (!filteredCommands.length) {
    box.hidden = true;
    return;
  }
  
  cmdSelIdx = 0;
  box.innerHTML = "";
  filteredCommands.forEach((c, i) => {
    const el = document.createElement("button");
    el.className = "cmd-item" + (i === cmdSelIdx ? " sel" : "");
    el.type = "button";
    el.innerHTML = `<span class="cmd-name">${c.name}</span> <span class="cmd-summary">${escapeHtml(c.usage ? c.usage + " — " + c.summary : c.summary)}</span>`;
    el.addEventListener("click", () => pickCommand(c));
    box.appendChild(el);
  });
  box.hidden = false;
}

function updateCmdSel() {
  const box = $("#commands");
  Array.from(box.children).forEach((el, i) => {
    el.classList.toggle("sel", i === cmdSelIdx);
    if (i === cmdSelIdx) el.scrollIntoView({ block: "nearest" });
  });
}

function pickCommand(c) {
  input.value = c.name + (c.usage ? " " : "");
  input.focus();
  renderCommands(input.value);
}

input.addEventListener("input", () => {
  autoGrow(input);
  renderCommands(input.value);
});

input.addEventListener("keydown", (e) => {
  const box = $("#commands");
  if (!box.hidden && filteredCommands.length > 0) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      cmdSelIdx = (cmdSelIdx + 1) % filteredCommands.length;
      updateCmdSel();
      return;
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      cmdSelIdx = (cmdSelIdx - 1 + filteredCommands.length) % filteredCommands.length;
      updateCmdSel();
      return;
    } else if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (cmdSelIdx >= 0 && cmdSelIdx < filteredCommands.length) {
        pickCommand(filteredCommands[cmdSelIdx]);
      }
      return;
    } else if (e.key === "Escape") {
      e.preventDefault();
      box.hidden = true;
      return;
    }
  }

  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#composer").requestSubmit();
    box.hidden = true;
  }
});

// --- dock composer (graph/map) — same conversation, mirrored into the card ---
// The turn is a real chat turn (user bubble + transcript land in the chat tab);
// the dock card is a lens showing only the latest exchange.
const dockInput = $("#dock-input");
$("#dock-composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const t = dockInput.value;
  if (!t.trim() || streaming) return;
  dockInput.value = "";
  autoGrow(dockInput);
  openPeek(t.trim());
  send(t);
});
dockInput.addEventListener("input", () => autoGrow(dockInput));
dockInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#dock-composer").requestSubmit();
  }
});
stopBtn.addEventListener("click", () => fetch("/stop", { method: "POST" }));
// Optimistic: clear the transcript at once (the reset itself is a cached-seed
// copy server-side, but never make the click wait on the network).
$("#brand-logo").addEventListener("click", async () => {
  if (streaming) return;
  log.innerHTML = "";
  await fetch("/reset", { method: "POST" });
  document.querySelector('.tab[data-tab="chat"]').click(); // surface the loaded chat
  loadVault();
  loadSessions();
});

// --- unified sidebar (stats · search · files · history) ----------------------
if (localStorage.getItem("sidebar-collapsed") === "1")
  document.body.classList.add("sidebar-collapsed");
$("#sidebar-toggle").addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("sidebar-collapsed");
  localStorage.setItem("sidebar-collapsed", collapsed ? "1" : "0");
});

// Vault stats + file tree, from /vault_info. Best-effort: on error the placeholders stay.
async function loadVaultInfo() {
  try {
    const r = await fetch("/vault_info");
    const data = await r.json();
    if (data.error) return;
    $("#stat-notes").textContent = data.notes;
    $("#stat-links").textContent = data.links;
    $("#stat-clusters").textContent = data.clusters;
    $("#stat-unresolved").textContent = data.unresolved;
    $("#tree").innerHTML = data.tree || "";
    renderMapPicker(data.hubs || []); // map landing: best-connected notes
    buildNoteIndex();                 // explore note search reads the fresh tree
    applySidebarFilter();
  } catch (_) {}
}

// Tree click routing follows the active view: in explore's map mode a click
// roots the radial map on the note; otherwise it opens the note drawer (which
// also mirrors focus into the graph iframe via focusGraphNode).
$("#tree").addEventListener("click", (e) => {
  const leaf = e.target.closest(".tree-note");
  if (!leaf) return;
  const path = leaf.dataset.id;
  if (activeTab === "graph" && graphMode === "map") rootMap(path);
  else openNote(path);
});

// One search box filters both the file tree and the chat history.
function applySidebarFilter() {
  const q = $("#side-search").value.trim().toLowerCase();
  // notes: substring on name or full path
  $("#tree").querySelectorAll(".tree-note").forEach((el) => {
    el.hidden = !!q && !el.textContent.toLowerCase().includes(q) &&
                !(el.dataset.id || "").toLowerCase().includes(q);
  });
  // folders: hide if nothing visible remains inside; reveal matches while searching
  $("#tree").querySelectorAll("details").forEach((d) => {
    const any = Array.from(d.querySelectorAll(".tree-note")).some((n) => !n.hidden);
    d.hidden = !!q && !any;
    if (q && any) d.open = true;
  });
  // sessions: substring on title; while searching, the expand cap is lifted
  $("#sessions").querySelectorAll(".session").forEach((el) => {
    el.hidden = (!!q && !el.textContent.toLowerCase().includes(q)) ||
                (!q && !sessionsExpanded && +el.dataset.idx >= SESSION_CAP);
  });
  $("#sessions-more").hidden = !!q || sessionsExpanded || sessionCount <= SESSION_CAP;
}
$("#side-search").addEventListener("input", applySidebarFilter);

// --- history (last sidebar section; capped, "expand" reveals the rest) -------
const SESSION_CAP = 8;
let sessionsExpanded = false;
let sessionCount = 0;

$("#sessions-more").addEventListener("click", () => {
  sessionsExpanded = true;
  applySidebarFilter();
});

async function loadSessions() {
  try {
    const r = await fetch("/sessions");
    const current = r.headers.get("X-Silica-Session") || "";
    const box = $("#sessions");
    box.innerHTML = "";
    const sessions = await r.json();
    sessionCount = sessions.length;
    sessions.forEach((s, i) => {
      const el = document.createElement("div");
      el.className = "session" + (s.id === current ? " active" : "");
      el.dataset.idx = i;
      el.textContent = s.title || "untitled";
      el.title = s.title || "";
      el.addEventListener("click", () => openSession(s.id));
      box.appendChild(el);
    });
    $("#sessions-more").textContent = "+ " + Math.max(0, sessionCount - SESSION_CAP) + " more";
    applySidebarFilter();
  } catch (_) {}
}

async function openSession(id) {
  if (streaming) return;
  const r = await fetch("/session/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!r.ok) return;
  document.querySelector('.tab[data-tab="chat"]').click(); // surface the loaded chat
  await loadVault();
  loadSessions();
}

// --- tabs -------------------------------------------------------------------
// Rebuilding the graph (Louvain + cooccurrence labels) is not free — only do it
// when the vault might actually have changed (graphStale), not on every switch
// back into the tab. A turn that writes notes sets graphStale = true.
let graphStale = true;
$(".tabs").addEventListener("click", (e) => {
  const tab = e.target.dataset.tab;
  if (!tab) return;
  activeTab = tab;
  if (tab === "chat") closePeek(); // stream visible → card redundant
  $("#dock").hidden = tab === "chat"; // ask-from-here strip lives on graph + map
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $("#view-chat").classList.toggle("active", tab === "chat");
  $("#view-graph").classList.toggle("active", tab === "graph");
  if (tab === "graph") setGraphMode(graphMode); // load the active mode's content
});

// --- explore tab: network graph | concept heatmap | radial map ---------------
// Three modes in one view, one toolbar. "graph" is one build (wikilink structure
// + semantic k-NN overlay, layers toggled in the frame's HUD); "heat" is a
// server-rendered co-occurrence matrix (/heatmap) — not a network graph, hence
// its own label; "map" is a radial map rooted on one note (/map), which needs a
// root, so it opens on a hub-picker landing. network + heatmap share one iframe
// (src-swapped); map has its own so switching back doesn't rebuild the graph.
let graphMode = "graph";
let loadedMode = null;    // which page (/graph or /heatmap) #graph-frame holds
let mapRootedPath = null; // note the radial map is rooted on, or null → picker

function graphURL() {
  return graphMode === "heat" ? "/heatmap?t=" + Date.now() : "/graph?t=" + Date.now();
}

// Show one mode: toggle its controls + which frame/picker is visible, and load
// the graph/heatmap page only when it isn't the one already sitting in the frame
// (or the vault changed under us). Also the entry point when switching INTO the
// explore tab, so it must be idempotent.
function setGraphMode(m) {
  graphMode = m;
  document.querySelectorAll(".gmode-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.gmode === m));
  const isMap = m === "map", isHeat = m === "heat";
  $("#heat-controls").hidden = !isHeat;
  $("#node-search-wrap").hidden = isHeat;       // note search: network + map only
  $("#graph-frame").hidden = isMap;
  $("#map-frame").hidden = !isMap || !mapRootedPath;
  $("#map-picker").hidden = !isMap || !!mapRootedPath;
  closeNodeResults();
  if (isMap) {
    $("#graph-loading").hidden = true;
    if (mapRootedPath) $("#map-loading").hidden = true;
    $("#node-search").focus();
  } else {
    $("#map-loading").hidden = true;
    if (graphStale || loadedMode !== m) {
      $("#graph-loading").hidden = false;
      $("#graph-frame").src = graphURL();
      loadedMode = m;
      graphStale = false;
    }
  }
}

$("#graph-bar").addEventListener("click", (e) => {
  const m = e.target.dataset.gmode; // only the mode buttons carry it; inputs don't
  if (!m || m === graphMode) return;
  setGraphMode(m);
});

// #graph-frame finishes loading only once the server is done building — drop the
// loader then and re-sync the focus dim state after a (re)load.
$("#graph-frame").addEventListener("load", () => {
  $("#graph-loading").hidden = true;
  if (lastNotePath) focusGraphNode(lastNotePath);
});
$("#map-frame").addEventListener("load", () => { $("#map-loading").hidden = true; });

// heatmap mode: the concept focus fields drive /heatmap's own query params (its
// in-page HUD hides when embedded, so these are the live controls).
$("#heat-controls").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#heat-q").value.trim();
  const n = $("#heat-n").value || 40;
  const p = $("#heat-p").value || 0;
  $("#graph-loading").hidden = false;
  loadedMode = "heat";
  $("#graph-frame").src = "/heatmap?q=" + encodeURIComponent(q) +
    "&n=" + encodeURIComponent(n) + "&p=" + encodeURIComponent(p) + "&t=" + Date.now();
});

// --- map landing: root the radial map on a note; hub-picker until one is set --
function rootMap(path) {
  mapRootedPath = path;
  if (graphMode !== "map") setGraphMode("map");
  $("#map-picker").hidden = true;
  $("#map-frame").hidden = false;
  $("#map-loading").hidden = false;
  $("#map-frame").src = "/map?note=" + encodeURIComponent(path) + "&t=" + Date.now();
  closeNodeResults();
}

function renderMapPicker(hubs) {
  const box = $("#map-picker-list");
  box.innerHTML = "";
  for (const h of hubs) {
    const row = document.createElement("div");
    row.className = "hub-row";
    row.dataset.path = h.path;
    row.innerHTML = '<span class="hub-name"></span><span class="hub-deg"></span>';
    row.querySelector(".hub-name").textContent = h.name;
    row.querySelector(".hub-deg").textContent = h.degree;
    box.appendChild(row);
  }
}
$("#map-picker-list").addEventListener("click", (e) => {
  const row = e.target.closest(".hub-row");
  if (row) rootMap(row.dataset.path);
});

// --- explore note search (network: fly the camera · map: root the map) --------
// A fuzzy ranked picker over the vault's notes, indexed from the sidebar tree —
// same title→prefix→substring→path ranking the graph viewer's own search uses.
let noteIdx = [];      // [{name, path, ln, lp}]
let nodeResults = [];  // current ranked matches
let nodeSel = -1;

function buildNoteIndex() {
  noteIdx = Array.from($("#tree").querySelectorAll(".tree-note")).map((el) => {
    const name = el.textContent, path = el.dataset.id || "";
    return { name, path, ln: name.toLowerCase(), lp: path.toLowerCase() };
  });
}

function scoreNote(n, q) {
  if (n.ln === q) return 5;
  if (n.ln.startsWith(q)) return 4;
  if (n.ln.includes(q)) return 3;
  if (n.lp.includes(q)) return 2;
  return 0;
}

function renderNodeResults(raw) {
  const q = raw.trim().toLowerCase();
  const box = $("#node-results");
  if (!q) { closeNodeResults(); return; }
  nodeResults = noteIdx
    .map((n) => [scoreNote(n, q), n])
    .filter((p) => p[0] > 0)
    .sort((a, b) => b[0] - a[0] || a[1].name.localeCompare(b[1].name))
    .slice(0, 12)
    .map((p) => p[1]);
  nodeSel = nodeResults.length ? 0 : -1;
  box.innerHTML = "";
  nodeResults.forEach((n, i) => {
    const el = document.createElement("div");
    el.className = "node-result" + (i === nodeSel ? " sel" : "");
    el.innerHTML = '<span class="nr-name"></span><span class="nr-path"></span>';
    el.querySelector(".nr-name").textContent = n.name;
    el.querySelector(".nr-path").textContent = n.path;
    el.addEventListener("click", () => pickNote(n.path));
    box.appendChild(el);
  });
  box.hidden = nodeResults.length === 0;
}

function closeNodeResults() {
  $("#node-results").hidden = true;
  nodeResults = [];
  nodeSel = -1;
}

function moveNodeSel(d) {
  nodeSel = (nodeSel + d + nodeResults.length) % nodeResults.length;
  document.querySelectorAll("#node-results .node-result").forEach((el, i) => el.classList.toggle("sel", i === nodeSel));
}

function pickNote(path) {
  if (graphMode === "map") {
    rootMap(path);
  } else { // network: locate the note and fly the graph camera to it
    const f = $("#graph-frame");
    if (f.contentWindow) f.contentWindow.postMessage({ type: "silica-goto-path", path }, "*");
  }
  $("#node-search").value = "";
  closeNodeResults();
}

$("#node-search").addEventListener("input", (e) => renderNodeResults(e.target.value));
$("#node-search").addEventListener("keydown", (e) => {
  if (!nodeResults.length) return;
  if (e.key === "ArrowDown") { e.preventDefault(); moveNodeSel(1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); moveNodeSel(-1); }
  else if (e.key === "Enter") { e.preventDefault(); if (nodeSel >= 0) pickNote(nodeResults[nodeSel].path); }
  else if (e.key === "Escape") { $("#node-search").value = ""; closeNodeResults(); }
});


// --- attachments: drop / "+" accumulate files as chips above the input; they
// are NOT nucleated on drop. The next composer submit uploads them together with
// the typed message, so the agent acts on the files per the user's instruction.
let staged = []; // File objects awaiting the next submit
const attachEls = $("#attachments");

function renderAttachments() {
  attachEls.innerHTML = "";
  attachEls.hidden = staged.length === 0;
  staged.forEach((f, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span class="chip-name"></span><button type="button" class="chip-x" title="remove">✕</button>`;
    chip.querySelector(".chip-name").textContent = f.name;
    chip.querySelector(".chip-x").addEventListener("click", () => { staged.splice(i, 1); renderAttachments(); });
    attachEls.appendChild(chip);
  });
}
function addFiles(fileList) {
  for (const f of fileList) staged.push(f);
  renderAttachments();
}

// Upload every staged file + the typed text as one turn (server stages them —
// converts PDFs, stubs code — then the agent works on them per `text`).
function nucleateStaged(text) {
  if (streaming || !staged.length) return;
  const names = staged.map((f) => f.name);
  bubble("user").textContent = (text.trim() ? text.trim() + "\n" : "") + "⇪ " + names.join(", ");
  const fd = new FormData();
  for (const f of staged) fd.append("files", f);
  fd.append("text", text);
  staged = [];
  renderAttachments();
  runTurn(fetch("/nucleate", { method: "POST", body: fd }));
}

let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", (e) => { e.preventDefault(); if (--dragDepth <= 0) document.body.classList.remove("dragging"); });
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});

// "+" opens the native picker, constrained to what the nucleate lanes accept.
const nucleateInput = $("#nucleate-file");
fetch("/supported_types")
  .then((r) => r.json())
  .then((d) => { nucleateInput.accept = (d.extensions || []).join(","); })
  .catch(() => {}); // accept="" just means the picker shows all files
$("#attach").addEventListener("click", () => nucleateInput.click());
nucleateInput.addEventListener("change", () => {
  addFiles(nucleateInput.files);
  nucleateInput.value = ""; // reset so re-picking the same file fires change again
});

// --- note panel (right overlay drawer; opens from .note-link, the graph, and the map) -
const notePanel = $("#note-panel");
let lastNotePath = null;   // note currently open in the drawer
let lastViewedPath = null; // survives close — feeds the header reopen button

// The dock inset and the drawer width must agree; CSS reads it as --note-w.
function setNoteW(w) {
  document.documentElement.style.setProperty("--note-w", w + "px");
}

// Mirror the open note onto the graph + map iframes: the matching node + its
// 1-hop neighbours go full-opacity, everything else dims. No-op harmlessly if
// a tab was never opened (contentWindow still exists, message just has no
// listener yet).
function focusGraphNode(path) {
  for (const id of ["#graph-frame", "#map-frame"]) {
    const frame = $(id);
    if (frame.contentWindow) frame.contentWindow.postMessage({ type: "silica-focus-path", path }, "*");
  }
}

// Mermaid is a 3.5MB vendored bundle, so it loads on demand — only the first
// time an opened note actually contains a ```mermaid fence. Render failures
// leave the fence as plain text (suppressErrorRendering).
let mermaidLoad = null;
function renderMermaid(root) {
  const blocks = root.querySelectorAll("pre.mermaid");
  if (!blocks.length) return;
  mermaidLoad ||= new Promise((resolve) => {
    const s = document.createElement("script");
    s.src = "/static/mermaid.min.js";
    s.onload = () => {
      mermaid.initialize({
        startOnLoad: false, theme: "dark", suppressErrorRendering: true,
        fontFamily: "Martian Mono, ui-monospace, monospace",
        themeVariables: {
          darkMode: true, background: "#0A0D14",
          primaryColor: "#161B27", primaryTextColor: "#E8ECF5",
          primaryBorderColor: "#38425A", lineColor: "#8B95AC",
        },
      });
      resolve();
    };
    document.head.appendChild(s);
  });
  mermaidLoad.then(() => mermaid.run({ nodes: blocks }).catch(() => {}));
}

async function openNote(path) {
  if (!path) return;
  lastNotePath = path;
  lastViewedPath = path;
  focusGraphNode(path);
  $("#note-mini-map").open = false; // reset: reload lazily if reopened for the new note
  $("#note-mini-map-frame").src = "";
  $("#note-heatmap").open = false;
  $("#note-heatmap-frame").src = "";
  try {
    const r = await fetch("/note?path=" + encodeURIComponent(path));
    const data = await r.json();
    $("#note-title").textContent = data.title || "";
    $("#note-body").innerHTML = data.html || "";
    renderMermaid($("#note-body"));
    $("#note-body").scrollTop = 0;
    notePanel.classList.add("open");
    notePanel.setAttribute("aria-hidden", "false");
    document.body.classList.add("note-open"); // dock insets to the drawer's edge
    const btn = $("#note-last");
    btn.querySelector("span").textContent = data.title || path;
  } catch (_) {}
}
function closeNote() {
  notePanel.classList.remove("open");
  notePanel.setAttribute("aria-hidden", "true");
  document.body.classList.remove("note-open");
  lastNotePath = null; // lastViewedPath survives — the header button can reopen
  focusGraphNode(null);
}
$("#note-last").addEventListener("click", () => {
  if (lastViewedPath) openNote(lastViewedPath);
});

// Mini-map: load only when expanded (native <details>), so a plain note read
// never pays for a /map render.
$("#note-mini-map").addEventListener("toggle", function () {
  if (this.open && lastNotePath) {
    $("#note-mini-map-frame").src = "/map?note=" + encodeURIComponent(lastNotePath);
  }
});

// Concept heatmap: same lazy idiom — this note's concepts plus their
// strongest out-of-note neighbors, rendered only when expanded.
$("#note-heatmap").addEventListener("toggle", function () {
  if (this.open && lastNotePath) {
    $("#note-heatmap-frame").src = "/heatmap?note=" + encodeURIComponent(lastNotePath);
  }
});

// "map" button in the drawer header — jump to explore's map mode, rooted here.
// Capture the path FIRST: the programmatic tab .click() bubbles to the document
// outside-click handler, which closes the drawer and nulls lastNotePath
// synchronously before rootMap runs (else note=null). Pre-set graphMode so the
// tab-enter goes straight to map instead of loading the graph first.
$("#note-map").addEventListener("click", () => {
  const note = lastNotePath;
  if (!note) return;
  graphMode = "map";
  document.querySelector('.tab[data-tab="graph"]').click();
  rootMap(note);
});

// summarize / explain / quiz — dispatch the reader slash-command for the open
// note as a chat turn. The drawer stays open (the peek dock tucks under it and
// mirrors the turn), so the note you launched from is never lost.
const shellQuote = (s) => '"' + String(s).replace(/"/g, '\\"') + '"';
function drawerReader(makeCmd) {
  if (!lastNotePath || streaming) return; // streaming: send() would no-op — no peek either
  const cmd = makeCmd(lastNotePath, $("#note-title").textContent.trim());
  if (activeTab !== "chat") openPeek(cmd); // on chat the stream is already visible
  send(cmd);
}
$("#note-summarize").addEventListener("click", () => drawerReader((p) => "/summarize " + shellQuote(p)));
$("#note-explain").addEventListener("click", () => drawerReader((p, t) => "/explain " + shellQuote(t || p)));
$("#note-quiz").addEventListener("click", () => drawerReader((p) => "/quiz " + shellQuote(p)));
$("#note-relate").addEventListener("click", () => drawerReader((p) => "/relate " + shellQuote(p)));

// --- dock card (rendered answer for a dock- or drawer-launched turn) ---------
// Not a re-implementation of the chat flow: no tools, no thinking text. Title =
// the dispatched prompt; body = pulsing "thinking", then the answer as live
// markdown (mdLite), upgraded to the canonical OFM render on `done` — so
// wikilinks in the card open the note drawer and focus the graph. One exchange
// only; the next one replaces it. "open in chat" → the full transcript.
const peekEl = $("#peek");
let peek = null; // { body, caret, raw } while a turn is being mirrored
function openPeek(title) {
  const body = $("#peek-body");
  body.className = "";
  body.textContent = "thinking";
  const caret = document.createElement("span"); // own instance: the chat caret is a
  caret.className = "caret";                    // single element, re-parented live
  caret.textContent = "▍";
  body.appendChild(caret);
  $("#peek-title").textContent = title;
  peekEl.hidden = false;
  peek = { body, caret, raw: "" };
}
function closePeek() {
  peekEl.hidden = true;
  peek = null;
}
// Freeze: stop mirroring, drop the caret, leave the card up until dismissed.
function freezePeek() {
  if (!peek) return;
  peek.caret.remove();
  peek = null;
}
function peekDelta(text) {
  if (!peek) return;
  peek.raw += text;
  peek.body.innerHTML = mdLite(peek.raw);
  (peek.body.lastElementChild || peek.body).appendChild(peek.caret);
  peek.body.scrollTop = peek.body.scrollHeight;
}
// `done` upgrade: the server's canonical OFM render (wikilinks, callouts, math),
// same swap the chat pane does. Also covers no-delta turns (raw still empty).
function peekDone(ev) {
  if (!peek) return;
  if (ev.html || ev.answer) peek.body.innerHTML = ev.html || escapeHtml(ev.answer);
  freezePeek();
}
function peekError(msg) {
  if (!peek) return;
  peek.body.classList.add("error");
  peek.body.textContent = "error: " + msg;
  peek = null; // frozen; card stays until dismissed
}
$("#peek-open-chat").addEventListener("click", () => {
  document.querySelector('.tab[data-tab="chat"]').click(); // tab handler closes the peek
});
$("#peek-close").addEventListener("click", closePeek);

// --- note panel resize (drag left edge, clamped) ----------------------------
const NOTE_MIN_W = 280, NOTE_MAX_W = 800;
const savedNoteWidth = parseInt(localStorage.getItem("note-width"), 10);
if (savedNoteWidth) notePanel.style.width = Math.min(NOTE_MAX_W, Math.max(NOTE_MIN_W, savedNoteWidth)) + "px";
setNoteW(parseInt(notePanel.style.width, 10) || 420);
let resizingNote = false; // guards the outside-click-closes handler below: a drag
                           // that ends outside #note-panel fires a "click" there too
$("#note-resize").addEventListener("mousedown", (e) => {
  e.preventDefault();
  resizingNote = true;
  const startX = e.clientX, startWidth = notePanel.getBoundingClientRect().width;
  const onMove = (e2) => {
    const w = Math.min(NOTE_MAX_W, Math.max(NOTE_MIN_W, startWidth + (startX - e2.clientX)));
    notePanel.style.width = w + "px";
    setNoteW(w); // keep the dock inset glued to the drawer edge while dragging
  };
  const onUp = () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    localStorage.setItem("note-width", parseInt(notePanel.style.width, 10));
    setTimeout(() => { resizingNote = false; }, 0); // clear after this click event finishes
  };
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
});
// One delegated handler: .note-link (chat OR in-panel → in-place nav) opens the
// drawer; a click outside an open drawer closes it. The sidebar and the dock
// are persistent instruments — picking a note, toggling a folder, or typing a
// question about the open note must not close the drawer or reset the graph
// focus, so they never count as "outside". Neither does the reopen button
// (its own listener would immediately fight the close).
document.addEventListener("click", (e) => {
  if (resizingNote) return;
  // dismiss the explore note-search dropdown on any click outside it (a result
  // click runs its own handler first, so pickNote still fires)
  if (!e.target.closest("#node-search-wrap")) closeNodeResults();
  const link = e.target.closest(".note-link");
  if (link) { e.preventDefault(); openNote(link.dataset.path); return; }
  if (notePanel.classList.contains("open") &&
      !e.target.closest("#note-panel") && !e.target.closest("#sidebar") &&
      !e.target.closest("#dock") && !e.target.closest("#note-last")) closeNote();
});
$("#note-close").addEventListener("click", closeNote);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeNote(); });
// Graph node clicks (in the iframe) post a message up when embedded.
window.addEventListener("message", (e) => {
  if (e.data && e.data.type === "silica-open-note") openNote(e.data.path);
});

// --- session bootstrap (re-render server-side history; never resets on load) -
async function loadVault() {
  try {
    const r = await fetch("/messages");
    $("#vault").textContent = r.headers.get("X-Silica-Vault") || "";
    setCtxTokens(r.headers.get("X-Silica-Context-Tokens"), r.headers.get("X-Silica-Max-Context-Tokens"));
    const msgs = await r.json();
    log.innerHTML = "";
    for (const m of msgs) {
      const b = bubble(m.role === "user" ? "user" : "silica");
      if (m.role === "user") b.textContent = m.content;
      else { b.innerHTML = m.html || escapeHtml(m.content); addCopyBtn(b, () => m.content); }
    }
  } catch (_) {}
}
loadVault();
loadSessions();
loadVaultInfo();
// Land on chat — it's the primary surface. The tab handler does the rest.
document.querySelector('.tab[data-tab="chat"]').click();
