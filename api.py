"""HTTP gateway. Work is executed by ComfyUI's native prompt queue, never a second model engine."""
import asyncio
import hmac
import json
import logging
import secrets
import time
import uuid

from aiohttp import web
import folder_paths
from server import PromptServer
import execution
from . import native_llm_config as config

LOG = logging.getLogger("native_llm_api")
PREFIX = "/native_llm_api"
_ROUTES_REGISTERED = False
_PENDING = set()
_PENDING_LOCK = asyncio.Lock()

class APIError(Exception):
    def __init__(self, status, message, kind="invalid_request_error", code=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind
        self.code = code

def json_error(status, message, kind="invalid_request_error", code=None):
    return web.json_response({"error": {"message": message, "type": kind, "param": None, "code": code}}, status=status)

@web.middleware
async def _unused_middleware(request, handler):
    return await handler(request)

def authed(request, settings):
    value = request.headers.get("Authorization", "")
    supplied = value[7:] if value.lower().startswith("bearer ") else ""
    if not supplied or not hmac.compare_digest(supplied, settings["api_key"]):
        raise APIError(401, "Invalid or missing Bearer API key", "authentication_error", "invalid_api_key")

def models():
    return sorted(name for name in folder_paths.get_filename_list("text_encoders")
                  if name.lower().endswith((".safetensors", ".sft", ".ckpt", ".pt", ".pth")))

async def get_settings(request):
    return web.json_response({"settings": config.read(), "models": models(), "base_url": f"{request.scheme}://{request.host}/v1"})

async def save_settings(request):
    try:
        body = await request.json()
        if "model" in body and body["model"] and body["model"] not in models():
            raise APIError(400, "Model must be a known text encoder")
        return web.json_response({"settings": config.update(body), "models": models()})
    except (ValueError, TypeError) as ex:
        return json_error(400, str(ex))

async def status(request):
    instance = PromptServer.instance
    running, queued = instance.prompt_queue.get_current_queue_volatile()
    return web.json_response({"active_api_requests": len(_PENDING), "comfy_running": len(running),
                              "comfy_pending": len(queued), "model": config.read()["model"]})

async def free(request):
    # Asynchronous flag observed by ComfyUI's worker; avoid unloading an active model directly.
    PromptServer.instance.prompt_queue.set_flag("unload_models", True)
    return web.json_response({"queued": True, "message": "ComfyUI will release loaded models when its worker handles the flag"})

def _validate_body(body, settings, chat):
    if not isinstance(body, dict):
        raise APIError(400, "Request body must be an object")
    model = body.get("model") or settings["model"]
    if not model or model not in models():
        raise APIError(404, "Select an installed text encoder model in Native LLM API settings", code="model_not_found")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
        raise APIError(400, "Only n=1 is supported", code="unsupported_parameter")
    if body.get("response_format") not in (None, {"type": "text"}):
        raise APIError(400, "Only response_format type=text is supported", code="unsupported_parameter")
    for field in ("tools", "functions", "logprobs", "top_logprobs", "logit_bias", "tool_choice"):
        if field in body and body[field] not in (None, [], {}):
            raise APIError(400, f"{field} is not supported by this native backend", code="unsupported_parameter")
    stream_opts = body.get("stream_options")
    if stream_opts is not None and not isinstance(stream_opts, dict):
        raise APIError(400, "stream_options must be an object")
    if stream_opts and stream_opts.get("include_usage"):
        raise APIError(400, "Streaming token usage is not available", code="unsupported_parameter")
    effective = dict(settings)
    mapping = {"max_tokens": "max_tokens", "max_completion_tokens": "max_tokens",
               "temperature": "temperature", "top_p": "top_p", "top_k": "top_k",
               "min_p": "min_p", "repetition_penalty": "repetition_penalty",
               "presence_penalty": "presence_penalty", "seed": "seed",
               "thinking": "thinking", "mtp": "mtp"}
    for source, target in mapping.items():
        if source in body and body[source] is not None:
            effective[target] = body[source]
    for key, low, high in (
        ("temperature", 0.0, 2), ("top_p", 0, 1), ("top_k", 0, 1000),
        ("min_p", 0, 1), ("repetition_penalty", 0, 5), ("presence_penalty", 0, 5),
        ("max_tokens", 1, 32768),
    ):
        v = effective[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not low <= v <= high:
            raise APIError(400, f"Invalid {key}")
    for k in ("top_k", "max_tokens"):
        if int(effective[k]) != effective[k]:
            raise APIError(400, f"{k} must be an integer")
    if effective["mtp"] not in config.MTP:
        raise APIError(400, "Invalid mtp")
    if "seed" in effective and (type(effective["seed"]) is not int or not 0 <= effective["seed"] <= 0xffffffffffffffff):
        raise APIError(400, "Seed must be a nonnegative integer")
    if "stream" in body and not isinstance(body["stream"], bool):
        raise APIError(400, "stream must be boolean")
    if not isinstance(effective["thinking"], bool):
        raise APIError(400, "thinking must be boolean")
    if chat:
        prompt = render_chat(body.get("messages"), model, effective["thinking"])
    else:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "prompt must be a string")
    stops = body.get("stop", [])
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or len(stops) > 8 or any(not isinstance(s, str) for s in stops):
        raise APIError(400, "stop must be a string or a list of up to eight strings")
    return model, prompt, effective, stops

def render_chat(messages, model, thinking=False):
    if not isinstance(messages, list) or not messages or len(messages) > 256:
        raise APIError(400, "messages must be a nonempty array of at most 256 messages")
    parsed = []
    for message in messages:
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant"):
            raise APIError(400, f"Unsupported role: {role}")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text" or not isinstance(item.get("text"), str):
                    raise APIError(400, "Only text message parts are supported")
                parts.append(item["text"])
            content = "\n".join(parts)
        if not isinstance(content, str):
            raise APIError(400, "Message content must be text")
        if message.get("tool_calls"):
            raise APIError(400, "Tool calls are not supported")
        parsed.append((role, content))
    # Qwen3.5 native tokenizer recognizes this prefix and does not wrap it a second time.
    if "qwen35" in model.lower() or "qwen3.5" in model.lower() or "qwen38" in model.lower() or "qwen3.8" in model.lower():
        messages_text = "".join(f"<|im_start|>{'system' if r == 'developer' else r}\n{v}<|im_end|>\n" for r, v in parsed)
        messages_text += "<|im_start|>assistant\n"
        messages_text += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
        return messages_text
    return "\n\n".join(f"{r.upper()}:\n{v}" for r, v in parsed) + "\n\nASSISTANT:\n"

def make_prompt(model, prompt, options, request_id):
    mode = {"sampling_mode": "on" if options["sampling"] and options["temperature"] != 0 else "off"}
    if mode["sampling_mode"] == "on":
        mode.update({k: options[k] for k in (
            "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "presence_penalty")})
        mode["seed"] = options.get("seed", secrets.randbits(32))
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": model, "type": "stable_diffusion"}},
        "2": {"class_type": "TextGenerateV2", "inputs": {
            "clip": ["1", 0], "prompt": prompt, "max_length": int(options["max_tokens"]),
            "sampling_mode": mode, "thinking": options["thinking"],
            "use_default_template": options["use_default_template"], "mtp": options["mtp"]}},
        "3": {"class_type": "NativeLLMAPIOutput", "inputs": {
            "text": ["2", 0], "request_id": request_id}},
    }

async def submit(prompt):
    ps = PromptServer.instance
    prompt_id = str(uuid.uuid4())
    valid = await execution.validate_prompt(prompt_id, prompt, None)
    if not valid[0]:
        raise APIError(400, "Native ComfyUI workflow validation failed: " + str(valid[1]))
    number = ps.number
    ps.number += 1
    extra = {"create_time": int(time.time() * 1000), "comfy_usage_source": "native-llm-api"}
    ps.prompt_queue.put((number, prompt_id, prompt, extra, valid[2], {}))
    return prompt_id

def get_output(prompt_id):
    ps = PromptServer.instance
    entry = ps.prompt_queue.get_history(prompt_id=prompt_id).get(prompt_id)
    if not entry:
        return None
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        raise APIError(500, "ComfyUI native LLM execution failed; see ComfyUI console", "server_error")
    outputs = entry.get("outputs") or {}
    value = outputs.get("3", {}).get("text")
    if not value:
        raise APIError(500, "ComfyUI returned no output from the native LLM node", "server_error")
    return str(value[0])

async def cancel(prompt_id):
    ps = PromptServer.instance
    if not ps.prompt_queue.interrupt_if_running(prompt_id):
        ps.prompt_queue.delete_queue_item(lambda item: item[1] == prompt_id)

async def await_output(prompt_id, timeout, request):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = get_output(prompt_id)
        if output is not None:
            return output
        if request.transport is None or request.transport.is_closing():
            raise asyncio.CancelledError()
        await asyncio.sleep(.25)
    raise APIError(504, "Native LLM request timed out", "server_error")

def stop_text(text, stops):
    end = min([text.find(s) for s in stops if s and s in text], default=-1)
    return (text[:end], "stop") if end >= 0 else (text, None)

def format_answer(text, rid, created, model, chat, stop_reason):
    finish = stop_reason or "stop"
    if chat:
        return {"id": rid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}]}
    return {"id": rid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}]}

def sse_chunk(rid, created, model, chat, text="", finish=None):
    if chat:
        delta = {"role": "assistant", "content": text} if text else {}
        choices = [{"index": 0, "delta": delta, "finish_reason": finish}]
    else:
        choices = [{"index": 0, "text": text, "finish_reason": finish}]
    return {"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
            "created": created, "model": model, "choices": choices}

async def inference(request, chat):
    settings = config.read()
    authed(request, settings)
    try:
        body = await request.json()
    except (ValueError, TypeError):
        raise APIError(400, "Invalid JSON body")
    model, prompt, effective, stops = _validate_body(body, settings, chat)
    if len(prompt) > 250000:
        raise APIError(400, "Prompt exceeds the maximum accepted characters")
    async with _PENDING_LOCK:
        if len(_PENDING) >= settings["max_pending"]:
            raise APIError(429, "Native LLM API queue is full", "rate_limit_error")
        rid = str(uuid.uuid4())
        _PENDING.add(rid)
    prompt_id = None
    try:
        workflow = make_prompt(model, prompt, effective, rid)
        prompt_id = await submit(workflow)
        created = int(time.time())
        response_id = ("chatcmpl-" if chat else "cmpl-") + rid
        streaming = body.get("stream", False)
        if not streaming:
            result = await await_output(prompt_id, settings["timeout_seconds"], request)
            text, reason = stop_text(result, stops)
            return web.json_response(format_answer(text, response_id, created, model, chat, reason))
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        await response.prepare(request)
        deadline = time.monotonic() + settings["timeout_seconds"]
        while True:
            output = get_output(prompt_id)
            if output is not None:
                answer, reason = stop_text(output, stops)
                # Buffered SSE: native generate has no stable token streaming callback.
                first = sse_chunk(response_id, created, model, chat, answer)
                final = sse_chunk(response_id, created, model, chat, finish=reason or "stop")
                await response.write(("data: " + json.dumps(first) + "\n\n").encode())
                await response.write(("data: " + json.dumps(final) + "\n\n").encode())
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response
            if time.monotonic() >= deadline:
                await response.write(b"event: error\ndata: {\"error\":\"Native generation timed out\"}\n\n")
                return response
            await response.write(b": waiting for native ComfyUI generation\n\n")
            await asyncio.sleep(2)
    except (asyncio.CancelledError, ConnectionError, ConnectionResetError):
        if prompt_id:
            await cancel(prompt_id)
        raise
    except Exception:
        if prompt_id:
            await cancel(prompt_id)
        raise
    finally:
        _PENDING.discard(rid)

async def models_handler(request):
    authed(request, config.read())
    stamp = int(time.time())
    return web.json_response({"object": "list", "data": [{"id": m, "object": "model", "created": stamp,
                                                           "owned_by": "comfyui-native"} for m in models()]})

async def model_handler(request):
    authed(request, config.read())
    model = request.match_info["model_id"]
    if model not in models():
        raise APIError(404, "Model not found", code="model_not_found")
    return web.json_response({"id": model, "object": "model", "created": int(time.time()), "owned_by": "comfyui-native"})

async def chat_handler(request):
    return await inference(request, True)

async def completion_handler(request):
    return await inference(request, False)

def error_wrapper(handler):
    async def wrapped(request):
        try:
            return await handler(request)
        except APIError as e:
            return json_error(e.status, e.message, e.kind, e.code)
        except web.HTTPException:
            raise
        except Exception:
            LOG.exception("Native LLM API request failed")
            return json_error(500, "Internal server error; see ComfyUI log", "server_error")
    return wrapped

def register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    routes = PromptServer.instance.routes
    routes.get(PREFIX + "/settings")(error_wrapper(get_settings))
    routes.post(PREFIX + "/settings")(error_wrapper(save_settings))
    routes.get(PREFIX + "/status")(error_wrapper(status))
    routes.post(PREFIX + "/free")(error_wrapper(free))
    routes.get("/v1/models")(error_wrapper(models_handler))
    routes.get("/v1/models/{model_id:.+}")(error_wrapper(model_handler))
    routes.post("/v1/chat/completions")(error_wrapper(chat_handler))
    routes.post("/v1/completions")(error_wrapper(completion_handler))
    _ROUTES_REGISTERED = True
    LOG.info("Native LLM API registered: /v1/chat/completions")
