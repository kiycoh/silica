// Chat turn reducer — the pure event→view-model fold, testable headless under
// `node --test`. The DOM lives in main.ts (BridgeView). The wire event map is
// shared verbatim with the web GUI (PROTOCOL §Chat / callback.py::event_to_json),
// so this mirrors the proven app.js handler.

import type { Frame } from "./bridge.ts";

export interface ToolLine {
  id: string;
  label: string;
  status: "run" | "done" | "error";
  error?: string;
}

export interface TurnState {
  text: string; // streamed answer text (delta kind=text)
  tools: ToolLine[]; // tool_start/done/error + batch activity lines
  answer: string | null; // final markdown (chat_done)
  error: string | null; // chat_error, or a socket drop mid-turn
  done: boolean;
}

export function emptyTurn(): TurnState {
  return { text: "", tools: [], answer: null, error: null, done: false };
}

/** Fold one server frame into the in-flight turn (mutates in place). Reasoning
 * deltas never arrive — event_to_json drops them in v1 — so only text deltas
 * render, exactly as the web app.js ignores everything but kind==="text". */
export function applyChatFrame(s: TurnState, frame: Frame): void {
  if (frame.type === "chat_done") {
    s.answer = typeof frame.answer === "string" ? frame.answer : "";
    s.done = true;
    return;
  }
  if (frame.type === "chat_error") {
    s.error = typeof frame.error === "string" ? frame.error : "unknown error";
    s.done = true;
    return;
  }
  if (frame.type !== "chat_event") return;

  const ev = (frame.event ?? {}) as Record<string, unknown>;
  switch (String(ev.type)) {
    case "delta":
      if (ev.kind === "text") s.text += String(ev.text ?? "");
      return; // reasoning/reset ignored (same as app.js)
    case "tool_start":
      s.tools.push({ id: String(ev.id), label: String(ev.name), status: "run" });
      return;
    case "tool_done": {
      const t = s.tools.find((x) => x.id === String(ev.id));
      if (t) t.status = "done";
      return;
    }
    case "tool_error": {
      const t = s.tools.find((x) => x.id === String(ev.id));
      if (t) { t.status = "error"; t.error = String(ev.error ?? ""); }
      return;
    }
    case "batch":
      // A batch run has no id and no terminal — a standalone activity line.
      s.tools.push({ id: `batch:${s.tools.length}`, label: `${String(ev.kind)} · ${String(ev.label)}`, status: "run" });
      return;
  }
}
