// node:test, zero deps (Node 24 strips types natively): run with `npm test`.
// Drives the handshake/reconnect state machine with a fake socket + fake timer.
import assert from "node:assert/strict";
import { test } from "node:test";

import { BridgeClient, PROTOCOL_VERSION, type BridgeInfo, type Frame, type SocketLike } from "./bridge.ts";

function fakeSocket() {
  const sent: string[] = [];
  const sock: SocketLike & { sent: string[]; closed: boolean } = {
    sent,
    closed: false,
    send(d) { sent.push(d); },
    close() { this.closed = true; },
    onOpen: null, onMessage: null, onClose: null, onError: null,
  };
  return sock;
}

function harness(info: BridgeInfo | null) {
  const sock = fakeSocket();
  const statuses: string[] = [];
  const scheduled: Array<{ fn: () => void; ms: number }> = [];
  const frames: Frame[] = [];
  const client = new BridgeClient({
    readBridgeInfo: async () => info,
    connect: () => sock,
    onStatus: (s) => statuses.push(s),
    onFrame: (f) => frames.push(f),
    schedule: (fn, ms) => { scheduled.push({ fn, ms }); return scheduled.length; },
    cancel: () => {},
  });
  return { sock, statuses, scheduled, frames, client };
}

test("dials and sends a spec-correct hello, welcome → connected", async () => {
  const h = harness({ port: 12345, token: "deadbeef", protocolVersion: 1 });
  await h.client.start();
  assert.equal(h.statuses.at(-1), "connecting");

  h.sock.onOpen!();
  assert.equal(h.sock.sent.length, 1);
  assert.deepEqual(JSON.parse(h.sock.sent[0]), {
    type: "hello", token: "deadbeef", protocolVersion: PROTOCOL_VERSION, role: "plugin",
  });

  h.sock.onMessage!(JSON.stringify({ type: "welcome", vault: "MyVault", protocolVersion: 1 }));
  assert.equal(h.statuses.at(-1), "connected");
});

test("frames after welcome reach onFrame; not before", async () => {
  const h = harness({ port: 1, token: "t", protocolVersion: 1 });
  await h.client.start();
  h.sock.onOpen!();
  h.sock.onMessage!(JSON.stringify({ type: "rpc", id: 1, method: "read" })); // pre-welcome: dropped
  assert.equal(h.frames.length, 0);
  h.sock.onMessage!(JSON.stringify({ type: "welcome", protocolVersion: 1 }));
  h.sock.onMessage!(JSON.stringify({ type: "rpc", id: 2, method: "read" }));
  assert.equal(h.frames.length, 1);
  assert.equal(h.frames[0].id, 2);
});

test("verifyWelcome rejection refuses the session (no connect), closes, reconnects", async () => {
  const sock = fakeSocket();
  const statuses: string[] = [];
  const frames: Frame[] = [];
  const scheduled: Array<{ fn: () => void; ms: number }> = [];
  const client = new BridgeClient({
    readBridgeInfo: async () => ({ port: 1, token: "t", protocolVersion: 1 }),
    connect: () => sock,
    onStatus: (s) => statuses.push(s),
    onFrame: (f) => frames.push(f),
    verifyWelcome: (f) => (f.vault === "test" ? null : `wrong vault: ${String(f.vault)}`),
    schedule: (fn, ms) => { scheduled.push({ fn, ms }); return scheduled.length; },
    cancel: () => {},
  });
  await client.start();
  sock.onOpen!();
  sock.onMessage!(JSON.stringify({ type: "welcome", vault: "alex_second_brain", protocolVersion: 1 }));
  assert.equal(statuses.at(-1), "disconnected"); // never "connected"
  assert.ok(sock.closed);
  assert.equal(scheduled.length, 1);
  // A post-welcome frame must NOT be dispatched — handshake was refused.
  sock.onMessage!(JSON.stringify({ type: "rpc", id: 1, method: "read" }));
  assert.equal(frames.length, 0);
});

test("bye disconnects, closes the socket, schedules a reconnect", async () => {
  const h = harness({ port: 1, token: "t", protocolVersion: 1 });
  await h.client.start();
  h.sock.onOpen!();
  h.sock.onMessage!(JSON.stringify({ type: "bye", reason: "bad token" }));
  assert.equal(h.statuses.at(-1), "disconnected");
  assert.ok(h.sock.closed);
  assert.equal(h.scheduled.length, 1);
});

test("missing bridge file schedules a reconnect without dialing", async () => {
  let connectCalls = 0;
  const scheduled: Array<{ fn: () => void; ms: number }> = [];
  const client = new BridgeClient({
    readBridgeInfo: async () => null,
    connect: () => { connectCalls++; return fakeSocket(); },
    onStatus: () => {},
    onFrame: () => {},
    schedule: (fn, ms) => { scheduled.push({ fn, ms }); return scheduled.length; },
    cancel: () => {},
  });
  await client.start();
  assert.equal(connectCalls, 0);
  assert.equal(scheduled.length, 1);
  assert.equal(scheduled[0].ms, 1000);
});

test("reconnect backoff doubles and caps at 30s", async () => {
  const h = harness(null);
  await h.client.start(); // schedules #1 @ 1000
  for (let i = 0; i < 5; i++) {
    h.scheduled[h.scheduled.length - 1].fn();     // fire the pending timer → re-dials
    await new Promise((r) => setImmediate(r));     // let the async dial reschedule
  }
  assert.deepEqual(h.scheduled.map((s) => s.ms), [1000, 2000, 4000, 8000, 16000, 30000]);
});
