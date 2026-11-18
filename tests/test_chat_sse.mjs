import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createSSEParser, consumeChatSSE, buildChatMessages } from "../web/native_llm_sse.mjs";

const encoder = new TextEncoder();

function streamResponse(bytes, { ok = true, status = 200 } = {}) {
  const data = bytes.map((part) => part instanceof Uint8Array ? part : encoder.encode(part));
  return {
    ok, status,
    body: new ReadableStream({
      start(controller) {
        for (const part of data) controller.enqueue(part);
        controller.close();
      },
    }),
  };
}

function chunk(text, reasoning) {
  return "data: " + JSON.stringify({
    object: "chat.completion.chunk",
    choices: [{ index: 0, delta: { content: text, reasoning_content: reasoning }, finish_reason: null }],
  }) + "\n\n";
}

test("SSE parser reassembles fragmented lines, multiline data, CRLF and comments", () => {
  const events = [];
  const parser = createSSEParser((e) => events.push(e));
  parser.feed(": heartbeat\r\n");
  parser.feed("event: message\r\ndata: first\n");
  parser.feed("data: second\r\n\r\n");
  parser.feed("event: error\ndata: {");
  parser.feed('"error":{"message":"bad"}}\n\n');
  parser.finish();
  assert.deepEqual(events.map((e) => e.data), [
    "first\nsecond",
    '{"error":{"message":"bad"}}',
  ]);
  assert.equal(events[1].event, "error");
});

test("OpenAI chat stream decodes fragmented UTF-8 and reasoning/usage", async () => {
  const text = chunk("héllo 🌍", "") + chunk("", "thinking text") +
    'data: {"object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n' +
    "data: [DONE]\n\n";
  const bytes = encoder.encode(text);
  const split = [bytes.slice(0, 11), bytes.slice(11, 45), bytes.slice(45, 70),
    bytes.slice(70, 72), bytes.slice(72, 92), bytes.slice(92)];
  const deltas = [];
  const usage = [];
  await consumeChatSSE(streamResponse(split), {
    onDelta: (d) => deltas.push(d),
    onUsage: (u) => usage.push(u),
  });
  assert.equal(deltas.map((d) => d.content).join(""), "héllo 🌍");
  assert.equal(deltas.map((d) => d.reasoning).join(""), "thinking text");
  assert.deepEqual(usage, [{ prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 }]);
});

test("SSE error event propagates model failure and does not mark completion", async () => {
  const response = streamResponse([
    chunk("partial", ""),
    'event: error\ndata: {"error":{"message":"GPU kernel failed","type":"server_error"}}\n\n',
  ]);
  await assert.rejects(
    consumeChatSSE(response),
    /GPU kernel failed/,
  );
});

test("Unexpected early stream closure is an error, not a successful turn", async () => {
  await assert.rejects(
    consumeChatSSE(streamResponse([chunk("unfinished", "")])),
    /connection closed before generation completed/,
  );
});

test("Non-200 OpenAI JSON error is displayed", async () => {
  const response = {
    ok: false, status: 429,
    json: async () => ({ error: { message: "Native API queue full" } }),
  };
  await assert.rejects(consumeChatSSE(response), /Native API queue full/);
});

test("Conversation context includes complete turns only and is bounded", () => {
  const history = [
    { role: "user", content: "hi", status: "complete" },
    { role: "assistant", content: "hello", status: "complete" },
    { role: "user", content: "this failed", status: "error" },
    { role: "assistant", content: "partial", status: "stopped" },
    { role: "user", content: "try again", status: "complete" },
    { role: "assistant", content: "", status: "pending" },
  ];
  assert.deepEqual(buildChatMessages(history), [
    { role: "user", content: "hi" },
    { role: "assistant", content: "hello" },
    { role: "user", content: "try again" },
  ]);
  assert.deepEqual(buildChatMessages(history, 2), [
    { role: "assistant", content: "hello" },
    { role: "user", content: "try again" },
  ]);
});

test("Diagnostic chat is an internal tab, not a separate ComfyUI sidebar icon", () => {
  const sidebar = readFileSync(new URL("../web/native_llm_api.js", import.meta.url), "utf8");
  const chat = readFileSync(new URL("../web/native_llm_chat.js", import.meta.url), "utf8");
  const sidebarRegistrations = sidebar.match(/registerSidebarTab\s*\(/g) || [];
  assert.equal(sidebarRegistrations.length, 1);
  assert.match(sidebar, /id:\s*"latent-talent-native-llm-api"/);
  assert.match(sidebar, /element\("button", tabs, "Settings",/);
  assert.match(sidebar, /element\("button", tabs, "Chat",/);
  assert.match(sidebar, /mountChat\(chatPanel, api\)/);
  assert.doesNotMatch(sidebar, /registerChatSidebar/);
  assert.doesNotMatch(chat, /registerSidebarTab\s*\(/);
});
