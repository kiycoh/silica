import { App, ItemView, Plugin, PluginSettingTab, WorkspaceLeaf, normalizePath, type SettingDefinitionItem } from "obsidian";

import { BridgeClient, type Frame, type SocketLike, type Status } from "./bridge.ts";
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
      onStatus: (s, detail) => {
        this.status = s;
        this.statusDetail = detail;
        this.refreshViews();
      },
      onFrame: (frame, send) => this.onFrame(frame, send),
    });
    void this.client.start();
  }

  // RPC dispatch (phases 3–4): allowlist → typed read/write handlers. An unknown
  // method is refused, never executed (PROTOCOL §Security: fixed allowlist).
  onFrame(frame: Frame, send: (f: Frame) => void): void {
    if (frame.type !== "rpc") return;
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
      if (leaf.view instanceof BridgeView) leaf.view.render();
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

class BridgeView extends ItemView {
  plugin: SilicaBridgePlugin;

  constructor(leaf: WorkspaceLeaf, plugin: SilicaBridgePlugin) {
    super(leaf);
    this.plugin = plugin;
  }

  getViewType(): string { return VIEW_TYPE; }
  getDisplayText(): string { return "Silica bridge"; }
  getIcon(): string { return "link"; }
  async onOpen(): Promise<void> { this.render(); }

  render(): void {
    const el = this.contentEl;
    el.empty();
    el.createEl("h4", { text: "Silica bridge" });
    el.createEl("p", { text: `Status: ${this.plugin.status}` });
    if (this.plugin.statusDetail) el.createEl("p", { text: this.plugin.statusDetail });
    if (this.plugin.status !== "connected") {
      el.createEl("p", { text: "Run `silica connect` in this vault to start the bridge." });
    }
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
