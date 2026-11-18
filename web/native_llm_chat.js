import { consumeChatSSE, buildChatMessages } from "./native_llm_sse.mjs";

const CHAT_STYLE = [
  ".native-chat { height:100%; min-height:0; box-sizing:border-box; display:flex; flex-direction:column; gap:9px; padding:12px; color:var(--fg-color, #ddd); font:13px/1.4 system-ui, sans-serif; }",
  ".native-chat *, .native-chat *::before, .native-chat *::after { box-sizing:border-box; }",
  ".native-chat .chat-title { display:flex; align-items:center; justify-content:space-between; gap:8px; }",
  ".native-chat h2 { margin:0; font-size:16px; font-weight:650; }",
  ".native-chat button { border:1px solid var(--border-color, #555); border-radius:6px; background:var(--comfy-input-bg, #303030); padding:7px 10px; color:inherit; cursor:pointer; font:inherit; }",
  ".native-chat button:disabled { cursor:default; opacity:.45; }",
  ".native-chat button.primary { background:#34669b; border-color:#34669b; color:white; }",
  ".native-chat select, .native-chat textarea { background:var(--comfy-input-bg, #262626); color:inherit; border:1px solid var(--border-color, #555); border-radius:6px; font:inherit; }",
  ".native-chat .chat-model-row { display:flex; align-items:center; gap:5px; min-width:0; }",
  ".native-chat .chat-model-row label { flex-shrink:0; font-size:12px; }",
  ".native-chat .chat-model-row select { padding:7px 6px; width:100%; min-width:0; flex:1; }",
  ".native-chat .chat-model-row button { flex-shrink:0; }",
  ".native-chat .chat-scroll { flex:1; min-height:0; overflow-y:auto; overscroll-behavior:contain; padding:10px 5px; border:1px solid var(--border-color, #444); border-radius:7px; display:flex; flex-direction:column; gap:11px; }",
  ".native-chat .chat-empty { margin:auto; text-align:center; color:var(--fg-color, #aaa); opacity:.7; max-width:240px; }",
  ".native-chat .chat-message { max-width:94%; min-width:90px; border-radius:9px; padding:9px 10px; border:1px solid var(--border-color, #444); background:var(--comfy-input-bg, #272727); overflow-wrap:anywhere; }",
  ".native-chat .chat-message.user { align-self:flex-end; background:rgba(63,125,186,.22); }",
  ".native-chat .chat-message.assistant { align-self:flex-start; }",
  ".native-chat .chat-message.error { border-color:var(--error-color, #aa5555); }",
  ".native-chat .chat-role { font-size:11px; opacity:.7; font-weight:600; margin-bottom:5px; }",
  ".native-chat .chat-content, .native-chat .chat-reasoning { white-space:pre-wrap; overflow-wrap:anywhere; font-family:inherit; margin:0; }",
  ".native-chat .chat-reasoning-panel { margin:7px 0; border-top:1px solid var(--border-color, #555); padding-top:5px; }",
  ".native-chat .chat-reasoning-panel summary { cursor:pointer; opacity:.8; font-size:11px; }",
  ".native-chat .chat-reasoning { opacity:.75; font-size:12px; padding-top:5px; }",
  ".native-chat .chat-status { font-size:11px; min-height:15px; overflow-wrap:anywhere; opacity:.8; }",
  ".native-chat .chat-status.failure { color:var(--error-color, #f08f8f); opacity:1; }",
  ".native-chat .chat-composer { display:flex; flex-direction:column; gap:7px; flex-shrink:0; }",
  ".native-chat .chat-composer textarea { width:100%; min-height:68px; max-height:160px; padding:9px; resize:vertical; }",
  ".native-chat .chat-buttons { display:flex; gap:7px; align-items:center; }",
  ".native-chat .chat-buttons button { flex:1; }",
  ".native-chat .chat-hint { font-size:11px; opacity:.6; }",
].join("\n");

function el(tag, parent, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  if (parent) parent.append(node);
  return node;
}

// The conversation remains in memory while ComfyUI runs. We intentionally
// never write messages or API credentials to localStorage or ComfyUI workflows.
const session = {
  messages: [],
  sequence: 0,
  model: "",
  modelUserSelected: false,
  models: [],
  busy: false,
  abort: null,
  status: "Loading models…",
  statusIsError: false,
  clearOnFinish: false,
  active: null,
  view: null,
};

function sync() {
  session.view?.update();
}

function isAbort(error, controller) {
  return controller.signal.aborted || error?.name === "AbortError";
}

async function loadModels(api, force = false) {
  const response = await api.fetchApi("/native_llm_api/settings");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || "Unable to retrieve native LLM settings");
  const names = Array.isArray(data.models) ? data.models : [];
  session.models = names;
  if (!session.modelUserSelected || force || !names.includes(session.model)) {
    session.model = names.includes(data.settings?.model) ? data.settings.model : (names[0] || "");
    session.modelUserSelected = false;
  }
  if (session.status === "Loading models…" || session.statusIsError) {
    session.status = names.length ? "Ready" : "No models found in ComfyUI text_encoders";
    session.statusIsError = !names.length;
  }
  sync();
  return data;
}

async function sendMessage(api, draft) {
  const prompt = draft.trim();
  if (!prompt || session.busy || !session.model || !session.models.includes(session.model)) return false;

  const user = { id: ++session.sequence, role: "user", content: prompt, status: "complete" };
  const reply = { id: ++session.sequence, role: "assistant", content: "", reasoning: "", status: "pending" };
  session.messages.push(user, reply);
  const conversation = buildChatMessages(session.messages);
  const controller = new AbortController();
  const active = { controller, reply, user };
  session.active = active;
  session.abort = controller;
  session.busy = true;
  session.status = "Generating…";
  session.statusIsError = false;
  sync();

  try {
    // Obtain the current API key just for this request (works after key rotation).
    const settings = await loadModels(api);
    const model = session.model;
    if (!settings.models.includes(model)) throw new Error("Selected model is no longer available");
    if (controller.signal.aborted) {
      const error = new Error("Request cancelled");
      error.name = "AbortError";
      throw error;
    }
    const response = await api.fetchApi("/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + settings.settings.api_key,
      },
      body: JSON.stringify({
        model, messages: conversation, stream: true,
        // Sampling, MTP and max tokens continue to use the saved server defaults.
      }),
      signal: controller.signal,
    });
    await consumeChatSSE(response, {
      signal: controller.signal,
      onDelta: ({ content, reasoning }) => {
        if (session.active !== active || session.clearOnFinish) return;
        reply.content += content;
        reply.reasoning += reasoning;
        session.status = "Generating…";
        sync();
      },
    });
    if (session.active === active) {
      reply.status = "complete";
      session.status = "Ready";
      session.statusIsError = false;
    }
  } catch (error) {
    if (session.active === active) {
      if (isAbort(error, controller)) {
        reply.status = "stopped";
        session.status = "Stopped";
      } else {
        reply.status = "error";
        user.status = "error"; // Exclude failed requests from later model context.
        session.status = error?.message || "Chat request failed";
        session.statusIsError = true;
      }
    }
  } finally {
    if (session.active === active) {
      session.active = null;
      session.abort = null;
      session.busy = false;
      if (session.clearOnFinish) {
        session.messages = [];
        session.status = "Conversation cleared";
        session.statusIsError = false;
        session.clearOnFinish = false;
      }
      sync();
    }
  }
  return true;
}

function stopMessage() {
  if (session.abort && !session.abort.signal.aborted) {
    session.status = "Stopping…";
    session.abort.abort();
    sync();
  }
}

function clearMessages() {
  if (session.busy) {
    session.clearOnFinish = true;
    stopMessage();
  } else {
    session.messages = [];
    session.status = "Conversation cleared";
    session.statusIsError = false;
    sync();
  }
}

function mountChat(host, api) {
  host.textContent = "";
  const root = el("div", host, undefined, "native-chat");
  el("style", root).textContent = CHAT_STYLE;

  const heading = el("div", root, undefined, "chat-title");
  el("h2", heading, "Native LLM Chat");
  const clear = el("button", heading, "Clear");
  clear.title = "Clear this in-memory conversation";

  const modelRow = el("div", root, undefined, "chat-model-row");
  const modelLabel = el("label", modelRow, "Model");
  const modelSelect = el("select", modelRow);
  modelLabel.htmlFor = "native-chat-model";
  modelSelect.id = "native-chat-model";
  const refresh = el("button", modelRow, "Refresh");
  refresh.title = "Refresh installed models from ComfyUI";

  const scroll = el("div", root, undefined, "chat-scroll");
  scroll.setAttribute("role", "log");
  scroll.setAttribute("aria-label", "Chat messages");
  const status = el("div", root, session.status, "chat-status");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");

  const composer = el("div", root, undefined, "chat-composer");
  const input = el("textarea", composer);
  input.placeholder = "Type a message…";
  input.setAttribute("aria-label", "Chat message");
  input.rows = 3;
  const controls = el("div", composer, undefined, "chat-buttons");
  const send = el("button", controls, "Send", "primary");
  const stop = el("button", controls, "Stop");
  el("div", composer, "Enter to send · Shift+Enter for a new line · Conversations stay in browser memory only", "chat-hint");

  let modelOptions = null;
  let lastCount = -1;
  const bubbles = new Map();

  function nearBottom() {
    return scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 95;
  }

  function paintMessages() {
    const stick = nearBottom();
    if (lastCount !== session.messages.length) {
      scroll.textContent = "";
      bubbles.clear();
      lastCount = session.messages.length;
      if (!lastCount) {
        el("div", scroll, "Send a message to start a conversation.", "chat-empty");
      }
      for (const message of session.messages) {
        const row = el("article", scroll, undefined, "chat-message " + message.role);
        el("div", row, message.role === "user" ? "You" : "Assistant", "chat-role");
        const text = el("div", row, "", "chat-content");
        const details = el("details", row, undefined, "chat-reasoning-panel");
        el("summary", details, "Reasoning");
        const reasoning = el("div", details, "", "chat-reasoning");
        details.hidden = true;
        bubbles.set(message.id, { row, text, details, reasoning });
      }
    }
    for (const message of session.messages) {
      const bubble = bubbles.get(message.id);
      if (!bubble) continue;
      const content = message.content || (message.status === "pending" ? "…" :
        message.status === "stopped" ? "Stopped" :
        message.status === "error" ? "Failed to generate" : "");
      if (bubble.text.textContent !== content) bubble.text.textContent = content;
      const reason = message.reasoning || "";
      bubble.details.hidden = !reason;
      if (reason && bubble.reasoning.textContent !== reason) bubble.reasoning.textContent = reason;
      bubble.row.className = "chat-message " + message.role
        + (message.status === "error" ? " error" : "");
    }
    if (stick) scroll.scrollTop = scroll.scrollHeight;
  }

  function paintModels() {
    const signature = session.models.join("\0");
    if (signature !== modelOptions) {
      modelOptions = signature;
      modelSelect.textContent = "";
      if (!session.models.length) {
        const unavailable = el("option", modelSelect, "No installed models");
        unavailable.value = "";
      } else {
        for (const name of session.models) {
          const option = el("option", modelSelect, name);
          option.value = name;
        }
      }
    }
    if (modelSelect.value !== session.model) modelSelect.value = session.model;
  }

  const view = {
    update() {
      // A later mount replaces this view; no detached DOM is updated.
      if (session.view !== view) return;
      paintModels();
      paintMessages();
      send.disabled = session.busy || !session.model || !input.value.trim();
      stop.disabled = !session.busy || !!session.abort?.signal.aborted;
      clear.disabled = (!session.messages.length && !session.busy);
      refresh.disabled = session.busy;
      modelSelect.disabled = session.busy || !session.models.length;
      status.textContent = session.status;
      status.className = "chat-status" + (session.statusIsError ? " failure" : "");
    },
  };
  session.view = view;

  modelSelect.onchange = () => {
    session.model = modelSelect.value;
    session.modelUserSelected = true;
    sync();
  };
  input.oninput = () => sync();
  input.onkeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send.click();
    }
  };
  send.onclick = () => {
    const draft = input.value;
    if (!draft.trim() || session.busy || !session.model) return;
    input.value = "";
    sync();
    sendMessage(api, draft);
  };
  stop.onclick = () => stopMessage();
  clear.onclick = () => clearMessages();
  refresh.onclick = async () => {
    refresh.disabled = true;
    session.status = "Refreshing models…";
    session.statusIsError = false;
    sync();
    try { await loadModels(api, true); session.status = "Ready"; }
    catch (error) {
      session.status = error?.message || "Unable to refresh models";
      session.statusIsError = true;
    }
    sync();
  };

  sync();
  loadModels(api).catch((error) => {
    session.status = error?.message || "Unable to load settings";
    session.statusIsError = true;
    sync();
  });
}

// Mounted by the existing Native LLM API sidebar's internal Chat tab.
// Do not create an additional ComfyUI sidebar item.
export { mountChat };
