// Vanilla client: POST /chat returns text/event-stream, read incrementally via
// the body's ReadableStream (not EventSource — that only does GET).
const $ = (s) => document.querySelector(s);
const log = $("#log");
const input = $("#input");
const stopBtn = $("#stop");

let streaming = false;

function bubble(role) {
  const el = document.createElement("div");
  el.className = "msg " + (role === "user" ? "user" : "silica");
  el.innerHTML = `<div class="role">${role === "user" ? "you" : "⏺ silica"}</div><div class="body"></div>`;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el.querySelector(".body");
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function fmtTokens(n) {
  n = Number(n) || 0;
  return n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
}
function setCtxTokens(used, max) {
  max = Number(max) || 0;
  $("#ctx-tokens").textContent = max ? `${fmtTokens(used)}/${fmtTokens(max)} tok` : "";
}

async function runTurn(fetchPromise) {
  if (streaming) return;
  streaming = true;
  stopBtn.hidden = false;
  const body = bubble("silica");
  const thinking = document.createElement("details");
  thinking.className = "thinking";
  thinking.hidden = true;
  thinking.innerHTML = `<summary>✦ thinking</summary><div class="thinking-body"></div>`;
  body.appendChild(thinking);
  const thinkBody = thinking.querySelector(".thinking-body");
  const tools = document.createElement("div");
  tools.className = "tools";
  body.appendChild(tools);
  const text = document.createElement("div");
  text.className = "stream-text streaming";
  body.appendChild(text);
  const toolEls = {};
  let raw = "";
  let thinkRaw = "";

  try {
    const resp = await fetchPromise;
    if (resp.status === 409) { text.textContent = "(a turn is already in progress)"; return; }
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
    text.textContent = "error: " + e;
  } finally {
    streaming = false;
    stopBtn.hidden = true;
    text.classList.remove("streaming");
    loadSessions(); // turn saved server-side — refresh titles/order
    graphStale = true; // a turn may have written notes — rebuild next graph view
  }

  function handle(ev) {
    if (ev.type === "delta" && ev.kind === "reasoning") {
      thinkRaw += ev.text;
      thinkBody.textContent = thinkRaw;
      thinking.hidden = false;
      thinking.open = true;
    } else if (ev.type === "delta" && ev.kind === "text") {
      thinking.open = false; // real answer started — collapse the thinking block
      raw += ev.text;
      text.textContent = raw;
    } else if (ev.type === "tool_start") {
      const t = document.createElement("div");
      t.className = "tool";
      t.textContent = "⏺ " + ev.name + " …";
      tools.appendChild(t);
      toolEls[ev.id] = t;
    } else if (ev.type === "tool_done") {
      const t = toolEls[ev.id];
      if (t) { t.className = "tool done"; t.textContent = "✓ " + ev.name; }
    } else if (ev.type === "tool_error") {
      const t = toolEls[ev.id];
      if (t) { t.className = "tool error"; t.textContent = "✗ " + ev.name + " — " + ev.error; }
    } else if (ev.type === "batch") {
      const t = document.createElement("div");
      t.className = "tool";
      t.textContent = "⏺ " + ev.kind + " · " + ev.label;
      tools.appendChild(t);
    } else if (ev.type === "done") {
      thinking.open = false;
      text.innerHTML = ev.html || escapeHtml(ev.answer || "");
      setCtxTokens(ev.context_tokens, ev.max_context_tokens);
    } else if (ev.type === "error") {
      text.className = "tool error";
      text.textContent = "error: " + ev.error;
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
  let k = 5;
  const tokens = [];
  for (const part of rest.trim().split(/\s+/)) {
    const m = part.match(/^--k=(\d+)$/);
    if (m) k = parseInt(m[1], 10);
    else if (part) tokens.push(part);
  }
  const query = tokens.join(" ");
  if (!query) { body.textContent = "usage: /find <query> [--k=N]"; return; }
  body.textContent = "searching…";
  try {
    const r = await fetch("/find?q=" + encodeURIComponent(query) + "&k=" + k);
    body.innerHTML = await r.text();
  } catch (e) {
    body.textContent = "error: " + e;
  }
}

// --- composer ---------------------------------------------------------------
function autoGrow() {
  input.style.height = "auto";
  const border = input.offsetHeight - input.clientHeight; // box-sizing: border-box
  input.style.height = (input.scrollHeight + border) + "px"; // clamped visually by CSS max-height
}
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const t = input.value;
  input.value = "";
  autoGrow();
  send(t);
});
input.addEventListener("input", autoGrow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#composer").requestSubmit();
  }
});
$("#commands").addEventListener("click", (e) => {
  const cmd = e.target.dataset.cmd;
  if (!cmd) return;
  if (cmd === "/find") { input.value = "/find "; input.focus(); return; } // needs a query, don't send bare
  send(cmd);
});
stopBtn.addEventListener("click", () => fetch("/stop", { method: "POST" }));
$("#new-chat").addEventListener("click", async () => {
  await fetch("/reset", { method: "POST" });
  log.innerHTML = "";
  loadVault();
  loadSessions();
});

// --- history sidebar --------------------------------------------------------
if (localStorage.getItem("sidebar-collapsed") === "1")
  document.body.classList.add("sidebar-collapsed");
$("#sidebar-toggle").addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("sidebar-collapsed");
  localStorage.setItem("sidebar-collapsed", collapsed ? "1" : "0");
});

function applySessionFilter() {
  const q = $("#session-filter").value.trim().toLowerCase();
  $("#sessions").querySelectorAll(".session").forEach((el) => {
    el.hidden = !!q && !el.textContent.toLowerCase().includes(q);
  });
}
$("#session-filter").addEventListener("input", applySessionFilter);

async function loadSessions() {
  try {
    const r = await fetch("/sessions");
    const current = r.headers.get("X-Silica-Session") || "";
    const box = $("#sessions");
    box.innerHTML = "";
    for (const s of await r.json()) {
      const el = document.createElement("div");
      el.className = "session" + (s.id === current ? " active" : "");
      el.textContent = s.title || "untitled";
      el.title = s.title || "";
      el.addEventListener("click", () => openSession(s.id));
      box.appendChild(el);
    }
    applySessionFilter();
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
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $("#view-chat").classList.toggle("active", tab === "chat");
  $("#view-graph").classList.toggle("active", tab === "graph");
  $("#view-map").classList.toggle("active", tab === "map");
  if (tab === "graph" && graphStale) {
    $("#graph-loading").hidden = false;
    $("#graph-frame").src = "/graph?" + Date.now();
    graphStale = false;
  }
  if (tab === "map") $("#map-note").focus();
});
// iframe finishes loading only once the server is done building the graph — drop the loader then
$("#graph-frame").addEventListener("load", () => {
  $("#graph-loading").hidden = true;
  if (lastNotePath) focusGraphNode(lastNotePath); // re-sync dim state after a (re)load
});

// --- mindmap: root on a named note, render its precomputed positions ---------
$("#map-bar").addEventListener("submit", (e) => {
  e.preventDefault();
  const note = $("#map-note").value.trim();
  if (note) { $("#map-loading").hidden = false; $("#map-frame").src = "/map?note=" + encodeURIComponent(note) + "&t=" + Date.now(); }
});
$("#map-frame").addEventListener("load", () => { $("#map-loading").hidden = true; });

// --- drop-zone (whole window) ----------------------------------------------
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", (e) => { e.preventDefault(); if (--dragDepth <= 0) document.body.classList.remove("dragging"); });
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  bubble("user").textContent = "⇪ ingest: " + file.name;
  const fd = new FormData();
  fd.append("file", file);
  runTurn(fetch("/ingest", { method: "POST", body: fd }));
});

// --- note panel (right overlay drawer; opens from .note-link, the graph, and the map) -
const notePanel = $("#note-panel");
let lastNotePath = null;

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

async function openNote(path) {
  if (!path) return;
  lastNotePath = path;
  focusGraphNode(path);
  $("#note-mini-map").open = false; // reset: reload lazily if reopened for the new note
  $("#note-mini-map-frame").src = "";
  try {
    const r = await fetch("/note?path=" + encodeURIComponent(path));
    const data = await r.json();
    $("#note-title").textContent = data.title || "";
    $("#note-body").innerHTML = data.html || "";
    $("#note-body").scrollTop = 0;
    notePanel.classList.add("open");
    notePanel.setAttribute("aria-hidden", "false");
  } catch (_) {}
}
function closeNote() {
  notePanel.classList.remove("open");
  notePanel.setAttribute("aria-hidden", "true");
  lastNotePath = null;
  focusGraphNode(null);
}

// Mini-map: load only when expanded (native <details>), so a plain note read
// never pays for a /map render.
$("#note-mini-map").addEventListener("toggle", function () {
  if (this.open && lastNotePath) {
    $("#note-mini-map-frame").src = "/map?note=" + encodeURIComponent(lastNotePath);
  }
});

// "map" button in the drawer header — jump to the full map tab, rooted here.
$("#note-map").addEventListener("click", () => {
  if (!lastNotePath) return;
  document.querySelector('.tab[data-tab="map"]').click();
  $("#map-note").value = lastNotePath;
  $("#map-loading").hidden = false;
  $("#map-frame").src = "/map?note=" + encodeURIComponent(lastNotePath) + "&t=" + Date.now();
});

// --- note panel resize (drag left edge, clamped) ----------------------------
const NOTE_MIN_W = 280, NOTE_MAX_W = 800;
const savedNoteWidth = parseInt(localStorage.getItem("note-width"), 10);
if (savedNoteWidth) notePanel.style.width = Math.min(NOTE_MAX_W, Math.max(NOTE_MIN_W, savedNoteWidth)) + "px";
let resizingNote = false; // guards the outside-click-closes handler below: a drag
                           // that ends outside #note-panel fires a "click" there too
$("#note-resize").addEventListener("mousedown", (e) => {
  e.preventDefault();
  resizingNote = true;
  const startX = e.clientX, startWidth = notePanel.getBoundingClientRect().width;
  const onMove = (e2) => {
    const w = Math.min(NOTE_MAX_W, Math.max(NOTE_MIN_W, startWidth + (startX - e2.clientX)));
    notePanel.style.width = w + "px";
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
// drawer; a click outside an open drawer closes it.
document.addEventListener("click", (e) => {
  if (resizingNote) return;
  const link = e.target.closest(".note-link");
  if (link) { e.preventDefault(); openNote(link.dataset.path); return; }
  if (notePanel.classList.contains("open") && !e.target.closest("#note-panel")) closeNote();
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
      else b.innerHTML = m.html || escapeHtml(m.content);
    }
  } catch (_) {}
}
loadVault();
loadSessions();
