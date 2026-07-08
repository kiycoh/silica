import assert from "node:assert/strict";
import { test } from "node:test";

import { applyChatFrame, emptyTurn } from "./chat.ts";
import type { Frame } from "./bridge.ts";

const evt = (event: Record<string, unknown>): Frame => ({ type: "chat_event", turnId: "t", event });

test("folds tool lifecycle + text deltas, finalizes on chat_done", () => {
  const s = emptyTurn();
  applyChatFrame(s, evt({ type: "tool_start", name: "search", id: "1" }));
  applyChatFrame(s, evt({ type: "delta", kind: "text", text: "Hello " }));
  applyChatFrame(s, evt({ type: "delta", kind: "text", text: "world" }));
  applyChatFrame(s, evt({ type: "tool_done", name: "search", id: "1" }));
  applyChatFrame(s, { type: "chat_done", turnId: "t", answer: "**Hi**", html: "<b>Hi</b>" });

  assert.equal(s.text, "Hello world");
  assert.equal(s.answer, "**Hi**");
  assert.equal(s.done, true);
  assert.deepEqual(s.tools, [{ id: "1", label: "search", status: "done" }]);
});

test("reasoning/reset deltas are ignored (never on the wire, tolerated anyway)", () => {
  const s = emptyTurn();
  applyChatFrame(s, evt({ type: "delta", kind: "reasoning", text: "hmm" }));
  applyChatFrame(s, evt({ type: "delta", kind: "reset", text: "x" }));
  applyChatFrame(s, evt({ type: "delta", kind: "text", text: "real" }));
  assert.equal(s.text, "real");
});

test("tool_error records the message; batch is a standalone activity line", () => {
  const s = emptyTurn();
  applyChatFrame(s, evt({ type: "tool_start", name: "write", id: "9" }));
  applyChatFrame(s, evt({ type: "tool_error", name: "write", id: "9", error: "boom" }));
  applyChatFrame(s, evt({ type: "batch", kind: "dedup", label: "3 notes" }));
  assert.deepEqual(s.tools, [
    { id: "9", label: "write", status: "error", error: "boom" },
    { id: "batch:1", label: "dedup · 3 notes", status: "run" },
  ]);
});

test("chat_error sets error + done", () => {
  const s = emptyTurn();
  applyChatFrame(s, { type: "chat_error", turnId: "t", error: "a turn is already in progress" });
  assert.equal(s.error, "a turn is already in progress");
  assert.equal(s.done, true);
});
