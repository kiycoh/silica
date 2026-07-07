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
  runTurn(fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }));
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
  if (cmd) send(cmd);
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
$(".tabs").addEventListener("click", (e) => {
  const tab = e.target.dataset.tab;
  if (!tab) return;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $("#view-chat").classList.toggle("active", tab === "chat");
  $("#view-graph").classList.toggle("active", tab === "graph");
  $("#view-map").classList.toggle("active", tab === "map");
  if (tab === "graph") { $("#graph-loading").hidden = false; $("#graph-frame").src = "/graph?" + Date.now(); }
  if (tab === "map") $("#map-note").focus();
});
// iframe finishes loading only once the server is done building the graph — drop the loader then
$("#graph-frame").addEventListener("load", () => { $("#graph-loading").hidden = true; });

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

// --- note panel (right overlay drawer; opens from .note-link and the graph) -
const notePanel = $("#note-panel");
async function openNote(path) {
  if (!path) return;
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
}
// One delegated handler: .note-link (chat OR in-panel → in-place nav) opens the
// drawer; a click outside an open drawer closes it.
document.addEventListener("click", (e) => {
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
