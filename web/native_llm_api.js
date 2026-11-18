import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { mountChat } from "./native_llm_chat.js";

const FIELD_GROUPS = [
  ["Sampling", [
    ["temperature", "Temperature", "number", "0.01", "2", "0.01"],
    ["top_p", "Top P", "number", "0", "1", "0.01"],
    ["top_k", "Top K", "number", "0", "1000", "1"],
    ["min_p", "Min P", "number", "0", "1", "0.01"],
    ["repetition_penalty", "Repetition penalty", "number", "0", "5", "0.01"],
    ["presence_penalty", "Presence penalty", "number", "0", "5", "0.01"],
    ["max_tokens", "Maximum output tokens", "number", "1", "32768", "1"],
  ]],
  ["Native execution", [
    ["sampling", "Sampling enabled", "checkbox"],
    ["thinking", "Thinking / reasoning", "checkbox"],
    ["use_default_template", "Use model chat template", "checkbox"],
  ]],
  ["Server", [
    ["max_pending", "Max queued API requests", "number", "1", "128", "1"],
    ["timeout_seconds", "Request timeout (seconds)", "number", "15", "3600", "1"],
  ]],
];
const MTP_OPTIONS = ["auto", "off", "2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"];
let activePanel = "settings";

const STYLES = `
.native-api-root { height:100%; min-height:0; overflow:hidden; display:flex; flex-direction:column; box-sizing:border-box; padding:0; font:13px/1.45 system-ui, sans-serif; color:var(--fg-color, #ddd); }
.native-api-root .native-internal-tabs { display:flex; gap:5px; padding:8px 12px 0; flex-shrink:0; border-bottom:1px solid var(--border-color, #555); }
.native-api-root .native-internal-tab { flex:1; border:0; border-bottom:2px solid transparent; background:transparent; color:inherit; font:inherit; font-weight:600; padding:8px 12px; cursor:pointer; opacity:.7; }
.native-api-root .native-internal-tab[aria-selected="true"] { opacity:1; border-bottom-color:var(--primary-color, #5595d0); }
.native-api-root .native-internal-tab:focus-visible { outline:2px solid var(--primary-color, #5595d0); outline-offset:-2px; }
.native-api-root .native-internal-panel { flex:1; min-height:0; min-width:0; }
.native-api-root .native-internal-panel[hidden] { display:none; }
.native-api-root .native-settings-panel { overflow-y:auto; padding:14px; }
.native-api-root .native-chat-panel { overflow:hidden; }
.native-api-root .native-settings-panel h2 { font-size:16px; font-weight:650; margin:0 0 3px; }
.native-api-root .native-settings-panel h3 { font-size:13px; font-weight:650; margin:16px 0 8px; border-bottom:1px solid var(--border-color, #555); padding-bottom:6px; }
.native-api-root .native-settings-panel p { margin:5px 0 12px; opacity:.8; }
.native-api-root .native-settings-panel label { display:block; font-size:12px; margin:10px 0 4px; opacity:.92; }
.native-api-root .native-settings-panel input:not([type=checkbox]),.native-api-root .native-settings-panel select { box-sizing:border-box; width:100%; min-width:0; padding:7px; border:1px solid var(--border-color,#555); border-radius:5px; background:var(--comfy-input-bg,#272727); color:inherit; font:inherit; }
.native-api-root .native-settings-panel input[type=checkbox] { margin-right:8px; }
.native-api-root .native-settings-panel .check { display:flex; align-items:center; }
.native-api-root .native-settings-panel .actions { display:flex; flex-wrap:wrap; gap:6px; margin:12px 0; }
.native-api-root .native-settings-panel button { border:1px solid var(--border-color,#666); background:var(--comfy-input-bg,#303030); color:inherit; padding:7px 10px; border-radius:5px; cursor:pointer; }
.native-api-root .native-settings-panel button.primary { background:#3b6794; color:white; border-color:#3b6794; }
.native-api-root .native-settings-panel button:disabled { opacity:.5; cursor:wait; }
.native-api-root .native-settings-panel .hint { font-size:11px; opacity:.7; word-break:break-word; margin-top:5px; }
.native-api-root .native-settings-panel .notice { min-height:20px; font-size:12px; padding-top:5px; }
.native-api-root .native-settings-panel .bad { color:#ff8b8b; }
.native-api-root .native-settings-panel .good { color:#8bdfa4; }
.native-api-root .native-settings-panel .secret { display:flex; gap:6px; }
.native-api-root .native-settings-panel .secret input { flex:1; min-width:0; }
.native-api-root .native-settings-panel details { margin-top:12px; }
.native-api-root .native-settings-panel summary { cursor:pointer; }
.native-api-root .native-settings-panel code { overflow-wrap:anywhere; }
`;

function element(tag, parent, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  parent.append(node);
  return node;
}
function field(parent, key, name, kind, min, max, step, value) {
  const label = element("label", parent, undefined, kind === "checkbox" ? "check" : "");
  const input = document.createElement("input");
  input.type = kind;
  input.dataset.field = key;
  if (kind === "checkbox") {
    input.checked = !!value;
    label.append(input, document.createTextNode(name));
  } else {
    label.textContent = name;
    if (min != null) input.min = min;
    if (max != null) input.max = max;
    if (step != null) input.step = step;
    input.value = value ?? "";
    parent.append(input);
  }
  return input;
}
function select(parent, key, name, options, value) {
  const label = element("label", parent, name);
  const input = element("select", parent);
  input.dataset.field = key;
  for (const opt of options) {
    const option = element("option", input, opt);
    option.value = opt;
  }
  input.value = value || options[0];
  return input;
}

app.registerExtension({
  name: "LatentTalent.NativeLLMAPI",
  async setup(appInstance) {
    appInstance.extensionManager.registerSidebarTab({
      id: "latent-talent-native-llm-api",
      icon: "pi pi-android",
      title: "Native LLM API",
      tooltip: "Native LLM OpenAI-compatible API",
      type: "custom",
      render: (host) => {
        host.textContent = "";
        const root = element("div", host, undefined, "native-api-root");
        const style = element("style", root);
        style.textContent = STYLES;
        const tabs = element("div", root, undefined, "native-internal-tabs");
        tabs.setAttribute("role", "tablist");
        tabs.setAttribute("aria-label", "Native LLM API");
        const settingsTab = element("button", tabs, "Settings", "native-internal-tab");
        const chatTab = element("button", tabs, "Chat", "native-internal-tab");
        settingsTab.type = chatTab.type = "button";
        settingsTab.id = "native-llm-api-settings-tab";
        chatTab.id = "native-llm-api-chat-tab";
        const settingsPanel = element("div", root, undefined, "native-internal-panel native-settings-panel");
        const chatPanel = element("div", root, undefined, "native-internal-panel native-chat-panel");
        settingsPanel.id = "native-llm-api-settings-panel";
        chatPanel.id = "native-llm-api-chat-panel";
        settingsTab.setAttribute("role", "tab");
        chatTab.setAttribute("role", "tab");
        settingsTab.setAttribute("aria-controls", settingsPanel.id);
        chatTab.setAttribute("aria-controls", chatPanel.id);
        settingsPanel.setAttribute("role", "tabpanel");
        chatPanel.setAttribute("role", "tabpanel");
        settingsPanel.setAttribute("aria-labelledby", settingsTab.id);
        chatPanel.setAttribute("aria-labelledby", chatTab.id);

        let chatMounted = false;
        function showPanel(panel) {
          activePanel = panel;
          const isChat = panel === "chat";
          settingsPanel.hidden = isChat;
          chatPanel.hidden = !isChat;
          settingsTab.setAttribute("aria-selected", String(!isChat));
          chatTab.setAttribute("aria-selected", String(isChat));
          settingsTab.tabIndex = isChat ? -1 : 0;
          chatTab.tabIndex = isChat ? 0 : -1;
          if (isChat && !chatMounted) {
            chatMounted = true;
            mountChat(chatPanel, api);
          }
        }
        settingsTab.onclick = () => showPanel("settings");
        chatTab.onclick = () => showPanel("chat");
        showPanel(activePanel);
        element("h2", settingsPanel, "Native LLM API");
        element("p", settingsPanel, "Native ComfyUI inference · OpenAI-compatible endpoints");
        const notice = element("div", settingsPanel, "Loading settings…", "notice");
        const form = element("div", settingsPanel);
        const setNotice = (message, bad = false) => {
          notice.textContent = message;
          notice.className = "notice " + (bad ? "bad" : "good");
        };
        const request = async (path, opts = {}) => {
          const response = await api.fetchApi(path, opts);
          const body = await response.json();
          if (!response.ok) throw new Error(body.error?.message || body.error || "HTTP " + response.status);
          return body;
        };
        const load = async () => {
          try {
            const data = await request("/native_llm_api/settings");
            form.textContent = "";
            const settings = data.settings;
            const modelOptions = ["", ...data.models];
            select(form, "model", "Native text-encoder model", modelOptions, settings.model);
            element("div", form, "Models come from models/text_encoders. Use native ComfyUI-compatible checkpoints.", "hint");
            for (const [title, fields] of FIELD_GROUPS) {
              element("h3", form, title);
              for (const [key, label, type, min, max, step] of fields)
                field(form, key, label, type, min, max, step, settings[key]);
              if (title === "Native execution") {
                select(form, "mtp", "MTP speculative decoding", MTP_OPTIONS, settings.mtp);
                element("div", form, "Auto uses the model's native MTP implementation when MTP weights are present.", "hint");
              }
            }
            element("h3", form, "Connection");
            const base = element("input", form);
            base.value = data.base_url;
            base.readOnly = true;
            element("div", form, "API base URL (SillyTavern Custom OpenAI-compatible)", "hint");
            const secretRow = element("div", form, undefined, "secret");
            const secret = element("input", secretRow);
            secret.type = "password";
            secret.value = settings.api_key;
            secret.readOnly = true;
            secret.setAttribute("aria-label", "API key");
            const show = element("button", secretRow, "Show");
            show.onclick = () => {
              secret.type = secret.type === "password" ? "text" : "password";
              show.textContent = secret.type === "password" ? "Show" : "Hide";
            };
            const copy = element("button", secretRow, "Copy");
            copy.onclick = async () => {
              try { await navigator.clipboard.writeText(secret.value); setNotice("API key copied"); }
              catch (err) { setNotice("Clipboard unavailable: " + err.message, true); }
            };
            const actions = element("div", form, undefined, "actions");
            const save = element("button", actions, "Save settings", "primary");
            const refresh = element("button", actions, "Refresh");
            const rotate = element("button", actions, "Rotate API key");
            const free = element("button", actions, "Unload models");
            const status = element("div", form, "", "hint");
            element("div", form, "Requests use ComfyUI's queue and native memory management. Accepted native tokens stream incrementally via SSE; MTP commits may arrive in batches. Function calls are buffered for JSON validation.", "hint");
            element("div", form, "Security: keep ComfyUI behind a trusted LAN or VPN. The API key does not secure ComfyUI's other endpoints.", "hint");

            save.onclick = async () => {
              const values = {};
              for (const input of form.querySelectorAll("[data-field]")) {
                const key = input.dataset.field;
                values[key] = input.type === "checkbox" ? input.checked : input.type === "number" ? Number(input.value) : input.value;
              }
              save.disabled = true;
              try {
                await request("/native_llm_api/settings", {
                  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values),
                });
                setNotice("Settings saved");
              } catch (err) { setNotice(err.message, true); }
              finally { save.disabled = false; }
            };
            refresh.onclick = () => load();
            rotate.onclick = async () => {
              if (!window.confirm("Rotate API key? Existing clients will lose access until updated.")) return;
              try { await request("/native_llm_api/settings", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({rotate_api_key:true})}); await load(); setNotice("API key rotated"); }
              catch (err) { setNotice(err.message, true); }
            };
            free.onclick = async () => {
              try { await request("/native_llm_api/free", {method:"POST"}); setNotice("Queued ComfyUI native model unload"); }
              catch (err) { setNotice(err.message, true); }
            };
            const check = async () => {
              try {
                const s = await request("/native_llm_api/status");
                status.textContent = `API active: ${s.active_api_requests} · ComfyUI running: ${s.comfy_running} · Queued: ${s.comfy_pending}`;
              } catch { status.textContent = "Status unavailable"; }
            };
            check();
            setNotice("Ready");
          } catch (err) { setNotice("Unable to load: " + err.message, true); }
        };
        load();
      },
    });
  },
});
