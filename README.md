# ComfyUI Native LLM API

An OpenAI-compatible API gateway implemented as a ComfyUI custom node. Inference is submitted as normal ComfyUI workflows using the native `CLIPLoader` and `TextGenerateV2` nodes, retaining ComfyUI's model loading, native quantization, memory management, and MTP where supported.

**Status:** initial development release; a live ComfyUI integration test is still required.

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

- **Streaming is completion-buffered SSE, not live token streaming.** Heartbeats are sent while native generation runs and then the completed answer is sent in OpenAI-compatible chunks. Native ComfyUI presently does not expose a stable per-token streaming hook.
- Qwen3.5/Qwen3.8 chat-family checkpoints use native role markers for multi-turn conversations. Other native text encoders receive a role-labelled transcript wrapped in their own template; exact multi-turn semantics depend on tokenizer.
- No true tool/function calling, logprobs, constrained JSON schema generation, image input, embeddings or reranking yet. Unsupported requested features return clear errors rather than claiming success.
- ComfyUI itself is commonly unauthenticated. The API key protects the `/v1` routes but does not secure the ComfyUI web UI or custom-node admin settings. Do not expose ComfyUI directly to the public Internet.
- `stop` strings are applied to completed text after generation; they do not save inference tokens.
- Usage token counts are omitted because the native node does not expose authoritative prompt/completion token counts.
- Native VRAM management is automatic; launch-time features such as `--use-ck-attention` must be set when ComfyUI starts.

This project does not use llama.cpp, vLLM, or transformers generation engines.
