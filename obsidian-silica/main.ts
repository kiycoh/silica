import { App, ItemView, MarkdownRenderer, Plugin, PluginSettingTab, WorkspaceLeaf, normalizePath, type SettingDefinitionItem } from "obsidian";

import { BridgeClient, type Frame, type SocketLike, type Status } from "./bridge.ts";
import { applyChatFrame, emptyTurn, type TurnState } from "./chat.ts";
import { dispatchRpc, RPC_METHODS, type RpcApp } from "./handlers.ts";

const VIEW_TYPE = "silica-bridge-view";
const BRIDGE_FILE = ".obsidian/silica-bridge.json";

interface SilicaSettings {
  portOverride: string;
}
const DEFAULT_SETTINGS: SilicaSettings = { portOverride: "" };

export default class SilicaBridgePlugin extends Plugin {
  settings: SilicaSettings = DEFAULT_SETTINGS;
  client: BridgeClient | null = null;
  status: Status = "disconnected";
  statusDetail = "";

  async onload(): Promise<void> {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    this.registerView(VIEW_TYPE, (leaf) => new BridgeView(leaf, this));
    this.addRibbonIcon("link", "Silica bridge", () => void this.activateView());
    this.addCommand({
      id: "open-silica-bridge",
      name: "Open Silica bridge panel",
      callback: () => void this.activateView(),
    });
    this.addSettingTab(new SilicaSettingTab(this.app, this));
    // Heavy setup after layout is ready (avoids the startup `create` event storm).
    this.app.workspace.onLayoutReady(() => this.connect());
  }

  onunload(): void {
    this.client?.stop();
    this.client = null;
  }

  connect(): void {
    if (this.client) return;
    this.client = new BridgeClient({
      readBridgeInfo: async () => {
        try {
          const info = JSON.parse(await this.app.vault.adapter.read(BRIDGE_FILE));
          const override = this.settings.portOverride.trim();
          if (override) info.port = Number(override);
          return info;
        } catch {
          return null; // no session running yet
        }
      },
      connect: (url) => wrapSocket(new WebSocket(url)),
      // Defense-in-depth: refuse a bridge whose vault isn't this one, so a stray
      // silica-bridge.json can't make Silica reason over vault A and write to B.
      verifyWelcome: (frame) => {
        const served = String(frame.vault ?? "");
        const mine = this.app.vault.getName();
        return served && served !== mine
          ? `bridge serves vault "${served}", not "${mine}" — run silica connect in this vault`
          : null;
      },
      onStatus: (s, detail) => {
        this.status = s;
        this.statusDetail = detail;
        this.refreshViews();
      },
      onFrame: (frame, send) => this.onFrame(frame, send),
    });
    void this.client.start();
  }

  onFrame(frame: Frame, send: (f: Frame) => void): void {
    if (frame.type === "rpc") return this.onRpc(frame, send);
    // Chat replies (chat_event/chat_done/chat_error) → the panel that owns the turn.
    if (typeof frame.type === "string" && frame.type.startsWith("chat_")) {
      for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE)) {
        if (leaf.view instanceof BridgeView) leaf.view.handleChatFrame(frame);
      }
    }
  }

  // RPC dispatch (phases 3–4): allowlist → typed read/write handlers. An unknown
  // method is refused, never executed (PROTOCOL §Security: fixed allowlist).
  onRpc(frame: Frame, send: (f: Frame) => void): void {
    const id = frame.id as number;
    const method = String(frame.method);
    const params = (frame.params ?? {}) as Record<string, unknown>;
    if (!RPC_METHODS.has(method)) {
      send({ type: "rpc_error", id, error: `method not implemented: ${method}` });
      return;
    }
    // `as unknown as`: RpcApp's file params (TFileLike) are intentionally
    // narrower than Obsidian's TAbstractFile, so a direct cast can't prove the
    // contravariant param match. Runtime App satisfies RpcApp — the mock proves
    // the shape headlessly in handlers.test.ts.
    dispatchRpc(this.app as unknown as RpcApp, method, params, normalizePath)
      .then((result) => send({ type: "rpc_result", id, result }))
      .catch((e: unknown) => send({ type: "rpc_error", id, error: e instanceof Error ? e.message : String(e) }));
  }

  async activateView(): Promise<void> {
    const { workspace } = this.app;
    const existing = workspace.getLeavesOfType(VIEW_TYPE);
    const leaf: WorkspaceLeaf | null = existing.length ? existing[0] : workspace.getRightLeaf(false);
    if (!leaf) return;
    if (!existing.length) await leaf.setViewState({ type: VIEW_TYPE, active: true });
    workspace.revealLeaf(leaf);
  }

  refreshViews(): void {
    for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE)) {
      if (leaf.view instanceof BridgeView) leaf.view.renderStatus();
    }
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }
}

function wrapSocket(ws: WebSocket): SocketLike {
  const s: SocketLike = {
    send: (d) => ws.send(d),
    close: () => ws.close(),
    onOpen: null, onMessage: null, onClose: null, onError: null,
  };
  ws.onopen = () => s.onOpen?.();
  ws.onmessage = (ev) => s.onMessage?.(String(ev.data));
  ws.onclose = () => s.onClose?.();
  ws.onerror = (e) => s.onError?.(e);
  return s;
}

// Chat panel: a message log + input over the bridge's chat channel. The pure
// event→view-model fold lives in chat.ts; this class owns only the DOM and the
// in-flight turn. One turn at a time (the server refuses a concurrent chat).
class BridgeView extends ItemView {
  plugin: SilicaBridgePlugin;
  private statusEl: HTMLElement | null = null;
  private logEl!: HTMLElement;
  private inputEl!: HTMLTextAreaElement;
  private sendBtn!: HTMLButtonElement;
  private stopBtn!: HTMLButtonElement;
  private turnId: string | null = null;
  private turn: TurnState | null = null;
  private bodyEl!: HTMLElement;
  private toolsEl!: HTMLElement;

  constructor(leaf: WorkspaceLeaf, plugin: SilicaBridgePlugin) {
    super(leaf);
    this.plugin = plugin;
  }

  getViewType(): string { return VIEW_TYPE; }
  getDisplayText(): string { return "Silica bridge"; }
  getIcon(): string { return "link"; }
  async onOpen(): Promise<void> { this.build(); }

  private build(): void {
    const el = this.contentEl;
    el.empty();
    el.addClass("silica-bridge");
    this.statusEl = el.createEl("p", { cls: "silica-status" });
    this.logEl = el.createDiv({ cls: "silica-log" });
    const row = el.createDiv({ cls: "silica-input-row" });
    this.inputEl = row.createEl("textarea", { attr: { rows: "2", placeholder: "Message Silica…" } });
    this.sendBtn = row.createEl("button", { text: "Send" });
    this.stopBtn = row.createEl("button", { text: "Stop" });
    this.stopBtn.hide();
    this.sendBtn.onclick = () => this.sendChat();
    this.stopBtn.onclick = () => {
      if (this.turnId) this.plugin.client?.send({ type: "chat_cancel", turnId: this.turnId });
    };
    this.inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); this.sendChat(); }
    });
    this.renderStatus();
  }

  renderStatus(): void {
    if (!this.statusEl) return; // status can fire before onOpen builds the DOM
    const s = this.plugin.status;
    if (s !== "connected" && this.turnId) this.abortTurn(s); // dropped mid-turn
    const detail = this.plugin.statusDetail ? ` — ${this.plugin.statusDetail}` : "";
    this.statusEl.setText(`${s}${detail}`);
    const blocked = s !== "connected" || this.turnId !== null;
    this.inputEl.disabled = blocked;
    this.sendBtn.disabled = blocked;
  }

  private bubble(role: "user" | "silica"): HTMLElement {
    const b = this.logEl.createDiv({ cls: `silica-msg silica-${role}` });
    this.logEl.scrollTop = this.logEl.scrollHeight;
    return b;
  }

  private sendChat(): void {
    if (this.plugin.status !== "connected" || this.turnId !== null) return;
    const text = this.inputEl.value.trim();
    if (!text) return;
    this.inputEl.value = "";
    this.bubble("user").setText(text);
    const asst = this.bubble("silica");
    this.toolsEl = asst.createDiv({ cls: "silica-tools" });
    this.bodyEl = asst.createDiv({ cls: "silica-body" });
    this.bodyEl.setText("…");
    this.turnId = crypto.randomUUID();
    this.turn = emptyTurn();
    this.plugin.client?.send({ type: "chat", turnId: this.turnId, text });
    this.stopBtn.show();
    this.renderStatus();
  }

  handleChatFrame(frame: Frame): void {
    if (!this.turn || frame.turnId !== this.turnId) return; // not our turn
    applyChatFrame(this.turn, frame);
    this.renderTurn();
    if (this.turn.done) this.finishTurn();
  }

  private renderTurn(): void {
    const t = this.turn;
    if (!t) return;
    this.toolsEl.empty();
    for (const tool of t.tools) {
      const glyph = tool.status === "done" ? "✓" : tool.status === "error" ? "✗" : "⏺";
      this.toolsEl
        .createDiv({ cls: `silica-tool silica-tool-${tool.status}` })
        .setText(`${glyph} ${tool.label}${tool.error ? ` — ${tool.error}` : ""}`);
    }
    if (!t.done) this.bodyEl.setText(t.text || "…");
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  private finishTurn(): void {
    const t = this.turn;
    this.bodyEl.empty();
    if (t?.error) {
      this.bodyEl.addClass("silica-error");
      this.bodyEl.setText(`error: ${t.error}`);
    } else {
      // Render markdown (not the server's html) → clickable wikilinks, no innerHTML.
      void MarkdownRenderer.render(this.app, t?.answer || t?.text || "", this.bodyEl, "", this);
    }
    this.turnId = null;
    this.turn = null;
    this.stopBtn.hide();
    this.renderStatus();
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  private abortTurn(reason: string): void {
    if (this.turn && this.bodyEl) {
      this.bodyEl.empty();
      this.bodyEl.addClass("silica-error");
      this.bodyEl.setText(`turn aborted: ${reason}`);
    }
    this.turnId = null;
    this.turn = null;
    this.stopBtn.hide();
  }
}

// Declarative settings (Obsidian 1.13+): getSettingDefinitions replaces the
// deprecated display(); getControlValue/setControlValue bind keys to our store.
class SilicaSettingTab extends PluginSettingTab {
  plugin: SilicaBridgePlugin;

  constructor(app: App, plugin: SilicaBridgePlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  getSettingDefinitions(): SettingDefinitionItem[] {
    const detail = this.plugin.statusDetail ? ` — ${this.plugin.statusDetail}` : "";
    return [
      { name: "Connection status", desc: `${this.plugin.status}${detail}` },
      {
        name: "Port override",
        desc: "Leave empty to use the port from silica-bridge.json.",
        control: { type: "text", key: "portOverride", placeholder: "Auto" },
      },
    ];
  }

  getControlValue(key: string): unknown {
    return key === "portOverride" ? this.plugin.settings.portOverride : undefined;
  }

  async setControlValue(key: string, value: unknown): Promise<void> {
    if (key === "portOverride") {
      this.plugin.settings.portOverride = String(value ?? "");
      await this.plugin.saveSettings();
    }
  }
}
