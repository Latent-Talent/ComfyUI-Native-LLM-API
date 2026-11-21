# ComfyUI Native LLM API

An OpenAI-compatible API gateway implemented as a ComfyUI custom node. Inference is submitted as normal ComfyUI workflows using the native `CLIPLoader` and `TextGenerateV2` nodes, retaining ComfyUI's model loading, native quantization, memory management, and MTP where supported.

**Status:** native incremental streaming and SillyTavern compatibility implementation committed. GitHub Actions static/unit tests pass; a live ComfyUI GPU integration test is still required.

## Install

Clone into `ComfyUI/custom_nodes/ComfyUI-Native-LLM-API` and restart ComfyUI. Requires a version of ComfyUI with the native `TextGenerateV2` node (v0.45.0 recommended).

Open **Native LLM API** from the left sidebar. This single sidebar item contains **Settings** and **Chat** tabs. Settings selects a text-encoder checkpoint from `ComfyUI/models/text_encoders`, sampling, and MTP options, and displays the generated API key. Chat is a lightweight diagnostic view that uses the same `/v1/chat/completions` API (streaming output, conversation history, Stop and Clear). It does not register a separate ComfyUI sidebar icon. Chat messages remain in browser memory only and are not written to ComfyUI workflow history.

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
- No logprobs, strict JSON Schema constrained decoding, embeddings or reranking. Images require a supported native vision checkpoint; unsupported models are explicitly rejected.
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

## SillyTavern video-generator: first-frame image context

The **Prepare Video Prompt → Include first frame with LLM request** setting
uses SillyTavern's `generateQuietPrompt({ quietPrompt, quietImage })` mechanism.
SillyTavern attaches `quietImage` to the **system-role** quiet prompt as an
OpenAI `image_url` content part:

```json
{"type":"image_url","image_url":{"url":"data:image/png;base64,...","detail":"high"}}
```

This API accepts that shape, including system-role images. It also accepts
standard user-role image parts, JPEG/PNG/WebP data URLs, and public HTTPS URLs.
SillyTavern typically converts locally hosted images into embedded data URLs
before sending the request.

- **Checked:** The prompt text, RP conversation, and first-frame image are
  passed to the native vision tokenizer. The first image is associated with
  `<Picture 1>` and an actual image embedding token. Subsequent images are
  `<Picture 2>`, etc., in original content order.
- **Unchecked:** No image part reaches the LLM. The first-frame image
  remains available to the separate ComfyUI video-generation workflow.
- The API preserves, without alteration, the video generator's instructions
  defining `<Picture 1>` as the exact FIRST FRAME. It leaves image/RP
  precedence, scene continuity, and subsequent action to those instructions.
- Native vision-capable Qwen3.5/Qwen3.8 and Gemma4 checkpoints are supported.
  Text-only models must not be presented as vision-capable.
- The endpoint returns **text descriptions/prompts**, not rendered video.

Media data remains in request-scoped RAM and never enters ComfyUI prompt
graph inputs, outputs, or `/history`. Per request limits: eight still frames,
32 MiB total compressed media and 16 megapixels total. HTTPS image URLs
must resolve exclusively to public IPs and return an image directly without
redirects; local/private URLs and animated images are refused.

**SillyTavern setting:** Enable inline-image sending for the Custom
OpenAI-compatible Chat Completion model. The video generator's own checkbox
still independently controls whether quiet LLM generation includes the image.

### Verify first-frame images actually reach the native model

Do not infer image support from a Qwen/Gemma model filename, or from an LLM
response. Some quantized checkpoints omit the vision encoder. The API now
inspects safetensors tensor names for a vision tower and fails with HTTP 422
when the selected checkpoint has no detected visual-encoder weights.
At runtime it compares the number of incoming `image_url` parts with the
number of **actual native image embedding objects** produced by the tokenizer.
A mismatch fails explicitly rather than returning a successful text-only answer.

The existing **Native LLM API → Settings → Check image status** button
displays only sanitized counts and an image-processing stage. The same data
is available from `GET /native_llm_api/status` under `last_vision_request`.
Neither endpoint includes prompts, image pixels, URLs or response text.

| Stage | Meaning |
|---|---|
| `text_only` | No image parts were attached to this API request |
| `image_received` | `image_url` part found in the incoming OpenAI messages |
| `images_decoded` | Image data decoded/validated successfully |
| `checkpoint_missing_vision` | Checkpoint has no detected visual-encoder tensors; rejected |
| `tokenizer_missing_images` | Image supplied but native tokenizer did not bind its image token; rejected |
| `vision_embeddings_verified` | Image tokens bound successfully; generation underway |
| `completed_with_vision` | Generation finished with matching image embeddings |

If a SillyTavern video prompt shows `text_only`, verify its
**Include first frame with LLM request** checkbox and SillyTavern's
**Send inline images** setting under the Custom OpenAI-compatible provider.
The `prompt: undefined` field in a chat-completion debug object is normal:
chat-completions use `messages` instead of `prompt`.

Note: a text-only prompt-token count is not proof that vision was processed.
For prompts containing image embedding objects, native token usage accounting
may be unavailable because image embeddings are not ordinary text token IDs.
