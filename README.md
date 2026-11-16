# ComfyUI Native LLM API

An OpenAI-compatible API gateway implemented as a ComfyUI custom node. Inference is submitted as normal ComfyUI workflows using the native `CLIPLoader` and `TextGenerateV2` nodes, retaining ComfyUI's model loading, native quantization, memory management, and MTP where supported.

**Status:** native incremental streaming and SillyTavern compatibility implementation committed. GitHub Actions static/unit tests pass; a live ComfyUI GPU integration test is still required.

## Install

Clone into `ComfyUI/custom_nodes/ComfyUI-Native-LLM-API` and restart ComfyUI. Requires a version of ComfyUI with the native `TextGenerateV2` node (v0.45.0 recommended).

Open **Native LLM API** from the left sidebar. Choose a text-encoder checkpoint from `ComfyUI/models/text_encoders`, adjust sampling and MTP options, and save. The API key is generated automatically and shown in the settings panel.

SillyTavern: **Chat Completion** > **Custom (OpenAI-compatible)**, URL `http://127.0.0.1:8188/v1` (or `http://COMFY-LAN-IP:8188/v1`), API key from sidebar, model from `/v1/models`.

## Endpoints

- `GET /v1/models` and `GET /v1/models/{model_id}`
- `POST /v1/chat/completions` and `POST /v1/completions`
- `GET /native_llm_api/settings`, `POST /native_llm_api/settings`
- `GET /native_llm_api/status`, `POST /native_llm_api/free`

Requests to `/v1/*` require `Authorization: Bearer <API key>`. Inference is queued in ComfyUI's normal execution queue. API and UI share the ComfyUI listening host/port.

## Known limitations (not hidden)

- **True incremental SSE:** the gateway observes accepted token IDs after native ComfyUI progress updates, including Qwen3.5 MTP commit batches. Tokens are decoded and streamed during inference. A small trailing text buffer protects against tokenizer boundary changes, and very short responses may appear only at completion. This relies on internal ComfyUI progress implementation and must be retested when ComfyUI changes.
- Qwen3.5/Qwen3.8 chat-family checkpoints use native role markers for multi-turn conversations. Other native text encoders receive a role-labelled transcript wrapped in their own template; exact multi-turn semantics depend on tokenizer.
- **Best-effort function calling:** OpenAI function/tool descriptions are included in the native prompt. The model must generate function-call JSON that is parsed into OpenAI tool_calls; function-call responses are buffered until complete, even with stream=true. No guarantee of schema adherence, and functions are never executed by this server.
- **JSON object mode:** response_format={"type":"json_object"} uses prompting and validates the complete result. This is not constrained decoding and does not guarantee a valid object on the first attempt.
- No logprobs, strict JSON Schema constrained decoding, image input, embeddings or reranking. Unsupported requested features return errors.
- ComfyUI itself is commonly unauthenticated. The API key protects the `/v1` routes but does not secure the ComfyUI web UI or custom-node admin settings. Do not expose ComfyUI directly to the public Internet.
- `stop` strings are detected after committed native tokens are decoded; when encountered, native generation is interrupted at the next progress update (with native cleanup). Stops are also trimmed from the final text.
- Prompt and completion usage counts are derived from the native tokenizer input and accepted generated token IDs. For text-only requests they are included in regular replies and optionally in the final SSE usage chunk.
- Native VRAM management is automatic; launch-time features such as `--use-ck-attention` must be set when ComfyUI starts.

This project does not use llama.cpp, vLLM, or transformers generation engines.


## Streaming and advanced compatibility

OpenAI-compatible request example (replace KEY and model name):

```bash
curl -N http://127.0.0.1:8188/v1/chat/completions \
  -H "Authorization: Bearer KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"model.safetensors","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Hi"}],"max_tokens":128,"temperature":0.7,"min_p":0.05,"top_k":64,"mtp":"auto"}'
```

Supported options include max_tokens, max_completion_tokens, temperature, top_p, top_k, min_p, repetition_penalty (or rep_pen), presence_penalty, seed, stop, thinking, mtp and stream. Unsupported sampling features such as nonzero frequency_penalty are rejected rather than silently ignored.

When reasoning is enabled, thought tokens are emitted as reasoning_content and ordinary reply tokens as content. Native stop sequences are processed against assistant content, not reasoning.

When OpenAI tools are supplied, the model is instructed to respond with a JSON tool-call object. This is prompt-based tool calling, not a native constrained tool decoder. Client tool execution and tool-result messages remain client responsibilities. In that mode streaming output is held until a complete function call can be parsed.

### Tests

GitHub Actions runs Python syntax checks, JavaScript syntax checks, settings tests, accepted-token/MTP-stream capture tests, and JSON/tool compatibility tests. GPU streaming behavior and SillyTavern end-to-end compatibility need a ComfyUI installation with a supported native model for final verification.

## Native GPU compatibility and chat privacy

- For a known W4A8 ConvRot CUDA shared-memory launch error (`convrot fused kernel shared memory request`), API inference retries through Comfy Kitchen’s existing two-pass rotation/INT8 quantization and chunked W4A8 GEMM backend. Other error types are not swallowed. The fallback applies only during API inference, preserves quantized weights and native ComfyUI memory management, and may be slower than a working fused kernel. **It requires a Kitchen version with `w4a8_codebook_gemm_chunked`; upgrade ComfyUI and Kitchen if your version lacks it.**
- API user messages, system instructions and assistant completions are not written into ComfyUI queue input or history. They are held in a request-scoped Python memory session and released when the request ends. ComfyUI history receives only a request identifier and a `completed` status. The model still processes the message in RAM/VRAM; OpenAI API clients receive the answer. This is not a general-purpose disabling of ComfyUI or system-wide logging, nor does it retroactively delete earlier history entries.
- API client output is available only during the HTTP request. If the client disconnects or generation times out, the private session is released and no chat text persists in the native ComfyUI prompt history.
