/**
 * Small dependency-free Server-Sent Events reader for OpenAI-compatible streams.
 * Deliberately does not persist, log, or inspect user prompts.
 */

export function createSSEParser(onEvent) {
  let buffer = "";
  let data = [];
  let eventType = "message";
  let eventId = "";

  function dispatch() {
    if (data.length > 0) {
      onEvent({ event: eventType, data: data.join("\n"), id: eventId });
    }
    data = [];
    eventType = "message";
  }

  function handleLine(line) {
    if (line.endsWith("\r")) line = line.slice(0, -1);
    if (!line) {
      dispatch();
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") data.push(value);
    else if (field === "event") eventType = value;
    else if (field === "id" && !value.includes("\0")) eventId = value;
  }

  return {
    feed(text) {
      buffer += text;
      let index;
      while ((index = buffer.indexOf("\n")) !== -1) {
        handleLine(buffer.slice(0, index));
        buffer = buffer.slice(index + 1);
      }
    },
    finish() {
      if (buffer) handleLine(buffer);
      buffer = "";
      dispatch();
    },
  };
}

function responseError(body, status) {
  const error = body && typeof body === "object" ? (body.error || body) : null;
  if (typeof error === "string") return error;
  if (error?.message && typeof error.message === "string") return error.message;
  return "Native LLM API returned HTTP " + status;
}

/**
 * Read OpenAI-compatible streaming chunks, including SSE heartbeats, fragmented
 * UTF-8, reasoning_content, streaming usage, and explicit server error events.
 * The returned Promise resolves only after [DONE].
 */
export async function consumeChatSSE(response, { onDelta = () => {}, onUsage = () => {}, signal } = {}) {
  if (!response.ok) {
    let payload;
    try { payload = await response.json(); } catch { payload = null; }
    throw new Error(responseError(payload, response.status));
  }
  if (!response.body || typeof response.body.getReader !== "function") {
    throw new Error("Browser does not support streamed responses");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let done = false;
  const parser = createSSEParser(({ event, data }) => {
    if (event === "error") {
      let parsed;
      try { parsed = JSON.parse(data); } catch { parsed = null; }
      throw new Error(responseError(parsed, response.status));
    }
    if (data === "[DONE]") {
      done = true;
      return;
    }
    let chunk;
    try { chunk = JSON.parse(data); }
    catch { throw new Error("Malformed streaming response from Native LLM API"); }
    if (chunk?.error) throw new Error(responseError(chunk, response.status));
    if (chunk?.usage) onUsage(chunk.usage);
    const choice = chunk?.choices?.[0];
    if (choice?.delta) {
      const delta = choice.delta;
      if (typeof delta.content === "string" || typeof delta.reasoning_content === "string") {
        onDelta({
          content: typeof delta.content === "string" ? delta.content : "",
          reasoning: typeof delta.reasoning_content === "string" ? delta.reasoning_content : "",
        });
      }
    }
  });
  try {
    while (!done) {
      if (signal?.aborted) {
        const error = new Error("Chat request cancelled");
        error.name = "AbortError";
        throw error;
      }
      const { value, done: closed } = await reader.read();
      if (closed) break;
      parser.feed(decoder.decode(value, { stream: true }));
    }
    if (!done) {
      parser.feed(decoder.decode());
      parser.finish();
    }
    if (!done) throw new Error("The connection closed before generation completed");
  } finally {
    if (!done) {
      try { await reader.cancel(); } catch { /* best effort */ }
    }
    reader.releaseLock();
  }
}

/** Convert completed chat turns to OpenAI messages; exclude aborted/error output. */
export function buildChatMessages(records, maxMessages = 80) {
  return records
    .filter((entry) =>
      (entry.role === "user" && entry.status !== "error")
      || (entry.role === "assistant" && entry.status === "complete"))
    .map(({ role, content }) => ({ role, content }))
    .slice(-maxMessages);
}
