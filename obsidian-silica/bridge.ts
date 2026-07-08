// Transport-neutral bridge client: the handshake + reconnect state machine.
// Imports nothing from `obsidian` so it runs under `node --test` with a fake
// socket. main.ts injects the real WebSocket and the vault file reader.
// Contract: PROTOCOL.md (frozen v1).

export const PROTOCOL_VERSION = 1;

const BACKOFF_START_MS = 1000;
const BACKOFF_MAX_MS = 30000;

export interface BridgeInfo {
  port: number;
  token: string;
  pid?: number;
  protocolVersion: number;
}

export type Frame = { type: string; [k: string]: unknown };

/** The subset of a browser/Electron `WebSocket` the client drives. */
export interface SocketLike {
  send(data: string): void;
  close(): void;
  onOpen: (() => void) | null;
  onMessage: ((data: string) => void) | null;
  onClose: (() => void) | null;
  onError: ((err: unknown) => void) | null;
}

export type Status = "disconnected" | "connecting" | "connected";

export interface BridgeDeps {
  readBridgeInfo: () => Promise<BridgeInfo | null>;
  connect: (url: string) => SocketLike;
  onStatus: (status: Status, detail: string) => void;
  onFrame: (frame: Frame, send: (f: Frame) => void) => void;
  /** Vet the `welcome` before trusting the session. Return a rejection reason
   * (e.g. the bridge serves a different vault) to refuse, or null to accept.
   * The token gates *access*; this gates *identity* — a defense-in-depth check
   * so a cross-vault bridge file can't drive writes into the wrong vault. */
  verifyWelcome?: (frame: Frame) => string | null;
  schedule?: (fn: () => void, ms: number) => unknown;
  cancel?: (handle: unknown) => void;
}

export function buildHello(token: string): Frame {
  return { type: "hello", token, protocolVersion: PROTOCOL_VERSION, role: "plugin" };
}

export class BridgeClient {
  private deps: BridgeDeps;
  private socket: SocketLike | null = null;
  private status: Status = "disconnected";
  private backoff = BACKOFF_START_MS;
  private reconnectHandle: unknown = null;
  private stopped = false;
  private handshakeDone = false;

  constructor(deps: BridgeDeps) {
    this.deps = deps;
  }

  async start(): Promise<void> {
    this.stopped = false;
    await this.dial();
  }

  stop(): void {
    this.stopped = true;
    this.cancelReconnect();
    this.closeSocket();
    this.setStatus("disconnected", "stopped");
  }

  /** Send a frame (chat / rpc_result). No-op unless the handshake completed. */
  send(frame: Frame): void {
    if (this.socket && this.status === "connected") this.socket.send(JSON.stringify(frame));
  }

  private setStatus(s: Status, detail: string): void {
    this.status = s;
    this.deps.onStatus(s, detail);
  }

  private async dial(): Promise<void> {
    if (this.stopped) return;
    this.closeSocket();
    const info = await this.readInfoSafe();
    if (!info) {
      this.setStatus("disconnected", "run `silica connect` (no bridge file)");
      this.scheduleReconnect();
      return;
    }
    this.setStatus("connecting", `ws://127.0.0.1:${info.port}`);
    this.handshakeDone = false;
    let sock: SocketLike;
    try {
      sock = this.deps.connect(`ws://127.0.0.1:${info.port}`);
    } catch (e) {
      this.setStatus("disconnected", `dial failed: ${String(e)}`);
      this.scheduleReconnect();
      return;
    }
    this.socket = sock;
    sock.onOpen = () => sock.send(JSON.stringify(buildHello(info.token)));
    sock.onMessage = (data) => this.onMessage(data);
    sock.onClose = () => this.onClose();
    sock.onError = () => { /* a close event always follows; reconnect handled there */ };
  }

  private async readInfoSafe(): Promise<BridgeInfo | null> {
    try {
      return await this.deps.readBridgeInfo();
    } catch {
      return null;
    }
  }

  private onMessage(data: string): void {
    let frame: Frame;
    try {
      frame = JSON.parse(data) as Frame;
    } catch {
      return; // non-JSON frame ignored (mirrors the server)
    }
    if (!this.handshakeDone) {
      if (frame.type === "welcome") {
        const reject = this.deps.verifyWelcome?.(frame);
        if (reject) {
          // Refuse like a `bye`: close + back off. Reconnect keeps watching the
          // bridge file, so starting `silica connect` in the right vault heals it.
          this.setStatus("disconnected", reject);
          this.closeSocket();
          this.scheduleReconnect();
          return;
        }
        this.handshakeDone = true;
        this.backoff = BACKOFF_START_MS;
        this.setStatus("connected", `vault: ${String(frame.vault ?? "")}`);
      } else if (frame.type === "bye") {
        this.setStatus("disconnected", `refused: ${String(frame.reason ?? "")}`);
        this.closeSocket();
        this.scheduleReconnect();
      }
      return;
    }
    this.deps.onFrame(frame, (f) => this.send(f));
  }

  private onClose(): void {
    if (this.stopped) return;
    this.socket = null;
    if (this.status !== "disconnected") this.setStatus("disconnected", "connection closed");
    this.scheduleReconnect();
  }

  private closeSocket(): void {
    if (this.socket) {
      try {
        this.socket.close();
      } catch {
        /* already closed */
      }
      this.socket = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectHandle !== null) return;
    const ms = this.backoff;
    this.backoff = Math.min(this.backoff * 2, BACKOFF_MAX_MS);
    const schedule = this.deps.schedule ?? ((fn, d) => setTimeout(fn, d));
    this.reconnectHandle = schedule(() => {
      this.reconnectHandle = null;
      void this.dial();
    }, ms);
  }

  private cancelReconnect(): void {
    if (this.reconnectHandle === null) return;
    const cancel = this.deps.cancel ?? ((h) => clearTimeout(h as ReturnType<typeof setTimeout>));
    cancel(this.reconnectHandle);
    this.reconnectHandle = null;
  }
}
