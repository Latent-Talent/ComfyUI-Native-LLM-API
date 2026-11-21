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
from . import streaming
from . import sillytavern as st
from . import image_context as ic

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
    response_format = body.get("response_format") or {"type": "text"}
    if response_format not in ({"type": "text"}, {"type": "json_object"}):
        raise APIError(400, "Only text and json_object response formats are supported")
    for field in ("logprobs", "top_logprobs", "logit_bias"):
        if field in body and body[field] not in (None, False, 0, [], {}):
            raise APIError(400, f"{field} is not supported by this native backend", code="unsupported_parameter")
    stream_opts = body.get("stream_options")
    if stream_opts is not None and not isinstance(stream_opts, dict):
        raise APIError(400, "stream_options must be an object")
    if stream_opts and type(stream_opts.get("include_usage", False)) is not bool:
        raise APIError(400, "include_usage must be boolean")
    if body.get("frequency_penalty") not in (None, 0):
        raise APIError(400, "frequency_penalty is unavailable in native ComfyUI; use repetition_penalty")
    try:
        tools, tool_choice = st.parse_tool_options(body) if chat else ({}, "none")
    except st.CompatibilityError as exc:
        raise APIError(400, str(exc)) from exc
    effective = dict(settings)
    mapping = {"max_tokens": "max_tokens", "max_completion_tokens": "max_tokens",
               "temperature": "temperature", "top_p": "top_p", "top_k": "top_k",
               "min_p": "min_p", "repetition_penalty": "repetition_penalty",
               "presence_penalty": "presence_penalty", "seed": "seed",
               "thinking": "thinking", "mtp": "mtp", "rep_pen": "repetition_penalty"}
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
        instructions = st.tool_instructions(tools, tool_choice)
        if response_format["type"] == "json_object":
            instructions += "\nReturn exactly one valid JSON object without markdown."
        image_urls = []
        prompt = render_chat(body.get("messages"), model, effective["thinking"],
                             extra_system=instructions, image_urls=image_urls)
        if image_urls and not any(x in model.lower() for x in
                ("qwen35", "qwen3.5", "qwen38", "qwen3.8", "gemma4")):
            raise APIError(400, "Selected model is not recognized as a native vision-capable model")
    else:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "prompt must be a string")
        if response_format["type"] == "json_object":
            prompt += "\nReturn only one valid JSON object."
    stops = body.get("stop", [])
    if stops is None:
        stops = []
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or len(stops) > 8 or any(not isinstance(s, str) for s in stops):
        raise APIError(400, "stop must be a string or a list of up to eight strings")
    if len(prompt) > 250000:
        raise APIError(400, "Prompt is too long")
    if len(stops) > 8 or any(len(x) > 128 for x in stops):
        raise APIError(400, "Stop sequences must be at most 128 characters")
    if response_format["type"] == "json_object" and tool_choice != "none":
        raise APIError(400, "json_object mode and tools cannot be combined")
    metadata = {"tools": tools, "tool_choice": tool_choice, "format": response_format["type"],
                "image_urls": image_urls if chat else [],
                "include_usage": bool(stream_opts and stream_opts.get("include_usage")),
                "include_reasoning": body.get("include_reasoning", True) is not False}
    return model, prompt, effective, stops, metadata

def render_chat(messages, model, thinking=False, extra_system="", image_urls=None):
    """Preserve RP history and interleaved image positions for native vision.

    SillyTavern's quietImage is an OpenAI image_url content part. We represent
    the associated first frame as <Picture 1> + a real native vision token;
    the source image itself remains private, outside ComfyUI's workflow JSON.
    """
    if not isinstance(messages, list) or not messages or len(messages) > 256:
        raise APIError(400, "messages must be a nonempty array of at most 256 messages")
    parsed = []
    model_lower = model.lower()
    is_gemma4 = "gemma4" in model_lower
    for message in messages:
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool", "function"):
            raise APIError(400, f"Unsupported role: {role}")
        content = message.get("content", "")
        if content is None:
            content = ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    raise APIError(400, "Invalid message content part")
                if item.get("type") == "text":
                    if not isinstance(item.get("text"), str):
                        raise APIError(400, "Text part must contain a string")
                    parts.append(item["text"])
                    continue
                if item.get("type") in ("image_url", "input_image"):
                    # SillyTavern quietImage is attached to a system-role quiet prompt.
                    # Preserve its role and text; native Qwen handles the image pad in that turn.
                    if role not in ("system", "developer", "user", "assistant"):
                        raise APIError(400, "Image context is only supported in system, user or assistant turns")
                    if image_urls is None:
                        raise APIError(400, "Image context is unavailable in this request")
                    if len(image_urls) >= ic.MAX_IMAGES:
                        raise APIError(400, "Maximum eight images per request")
                    try:
                        image_url = ic.parse_image_url(item)
                    except ic.ImageContextError as exc:
                        raise APIError(400, str(exc)) from exc
                    image_urls.append(image_url)
                    index = len(image_urls)
                    # The supplied first-frame instructions refer to <Picture 1>.
                    # Keep the exact user text; attach the picture marker beside
                    # the actual vision embedding in order of message parts.
                    vision_token = ("<|image><|image|><image|>" if is_gemma4
                                    else "<|vision_start|><|image_pad|><|vision_end|>")
                    parts.append(f"\n<Picture {index}> {vision_token}\n")
                    continue
                raise APIError(400, "Only text and image_url message parts are supported")
            content = "\n".join(parts)
        if not isinstance(content, str):
            raise APIError(400, "Message content must be text or an array of text/image parts")
        if message.get("tool_calls"):
            content += "\nPrevious function calls: " + json.dumps(message["tool_calls"], ensure_ascii=False)
        if role in ("tool", "function"):
            content = str(message.get("name", message.get("tool_call_id", "tool"))) + ": " + content
            role = "tool"
        parsed.append((role, content))
    if extra_system:
        parsed.insert(0, ("system", extra_system))
    if is_gemma4:
        def gemma_role(role):
            return "model" if role == "assistant" else "system" if role == "developer" else "user" if role == "tool" else role
        formatted = "".join(f"<|turn>{gemma_role(role)}\n{value}<turn|>\n" for role, value in parsed)
        return formatted + "<|turn>model\n"
    if any(marker in model_lower for marker in ("qwen35", "qwen3.5", "qwen38", "qwen3.8")):
        formatted = "".join(f"<|im_start|>{'system' if r == 'developer' else r}\n{v}<|im_end|>\n"
                            for r, v in parsed)
        formatted += "<|im_start|>assistant\n"
        formatted += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
        return formatted
    return "\n\n".join(f"{r.upper()}:\n{v}" for r, v in parsed) + "\n\nASSISTANT:\n"

def make_prompt(model, prompt, options, request_id):
    # ComfyUI V3 DynamicCombo is serialized as the selected STRING option
    # plus dotted child keys. A nested dict at the top level is silently
    # discarded by get_finalized_class_inputs(), causing a missing argument
    # when NativeLLMAPITextGenerate.execute() is called.
    sampling_enabled = bool(options["sampling"] and options["temperature"] != 0)
    generation_inputs = {
        "clip": ["1", 0], "prompt": "", "max_length": int(options["max_tokens"]),
        "sampling_mode": "on" if sampling_enabled else "off",
        "thinking": options["thinking"],
        "use_default_template": options["use_default_template"],
        "mtp": options["mtp"], "request_id": request_id,
    }
    if sampling_enabled:
        for key in ("temperature", "top_k", "top_p", "min_p",
                    "repetition_penalty", "presence_penalty"):
            generation_inputs[f"sampling_mode.{key}"] = options[key]
        generation_inputs["sampling_mode.seed"] = (
            options["seed"] if options.get("seed") is not None else secrets.randbits(32)
        )
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": model, "type": "stable_diffusion"}},
        "2": {"class_type": "NativeLLMAPITextGenerate", "inputs": generation_inputs},
        "3": {"class_type": "NativeLLMAPIOutput", "inputs": {
            "text": ["2", 0], "reasoning": ["2", 1], "request_id": request_id}},
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
    """Read completion marker from ComfyUI, actual content from private memory."""
    ps = PromptServer.instance
    entry = ps.prompt_queue.get_history(prompt_id=prompt_id).get(prompt_id)
    if not entry:
        return None
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        raise APIError(500, "ComfyUI native LLM execution failed; see ComfyUI console", "server_error")
    outputs = entry.get("outputs") or {}
    marker = (outputs.get("3") or {}).get("status") or []
    if marker != ["completed"]:
        raise APIError(500, "Native LLM did not return a completion marker", "server_error")
    workflow = entry.get("prompt") or []
    # ComfyUI /history prompt tuple includes the sanitized workflow, not user text.
    try:
        request_id = workflow[2]["3"]["inputs"]["request_id"]
    except (IndexError, KeyError, TypeError):
        raise APIError(500, "Native LLM output request ID is unavailable", "server_error")
    session = streaming.lookup(request_id)
    if session is None:
        raise APIError(500, "Native LLM request session expired", "server_error")
    snap = session.snapshot()
    if not snap["generation_completed"]:
        raise APIError(500, "Native LLM finished without output", "server_error")
    return snap["content"], snap["reasoning"]

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

def _compat_result(content, reasoning, metadata, model, response_id, created, chat, session):
    tool_calls = None
    try:
        if chat:
            tool_calls = st.parse_tool_calls(content, metadata["tools"], metadata["tool_choice"])
        if tool_calls is None and metadata["format"] == "json_object":
            st.validate_json_object(content)
    except st.CompatibilityError as exc:
        raise APIError(422, str(exc)) from exc
    snap = session.snapshot()
    return st.format_response(content, reasoning, model, response_id, created, chat,
                              snap["finish_reason"] or "stop", st.count_usage(session),
                              metadata["include_reasoning"], tool_calls)


async def _sse_write(response, payload):
    await response.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode())


async def _stream_response(request, prompt_id, model, response_id, created, chat, session,
                           stops, metadata, timeout_seconds):
    response = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })
    await response.prepare(request)
    sent_text = ""
    sent_reasoning = ""
    last_ping = time.monotonic()
    deadline = time.monotonic() + timeout_seconds
    buffer_tool_calls = metadata["tool_choice"] != "none"
    # Decoding a partial UTF-8/token sequence can revise its tail. Hold it back.
    holdback = max(24, max((len(stop) for stop in stops), default=0))
    try:
        if chat:
            await _sse_write(response, st.stream_chunk(response_id, created, model, True, {"role": "assistant"}))
        while True:
            final = get_output(prompt_id)
            snap = session.snapshot()
            if snap["error"]:
                raise APIError(500, "Native generation failed: " + snap["error"], "server_error")
            if time.monotonic() > deadline:
                raise APIError(504, "Native generation timed out", "server_error")
            completed = final is not None
            if completed:
                content, reasoning = final
                index, _ = streaming.first_stop(content, stops)
                if index is not None:
                    content = content[:index]
                # Check function JSON/schema before any final response.
                response_obj = _compat_result(content, reasoning, metadata, model,
                                              response_id, created, chat, session)
                calls = response_obj["choices"][0].get("message", {}).get("tool_calls") if chat else None
            else:
                content, reasoning = snap["content"], snap["reasoning"]
                calls = None
            if not buffer_tool_calls:
                if chat and metadata["include_reasoning"]:
                    safe = reasoning if completed else reasoning[:max(0, len(reasoning) - holdback)]
                    if not safe.startswith(sent_reasoning):
                        raise APIError(500, "Native reasoning decode revised emitted text", "server_error")
                    if len(safe) > len(sent_reasoning):
                        await _sse_write(response, st.stream_chunk(
                            response_id, created, model, True,
                            {"reasoning_content": safe[len(sent_reasoning):]}))
                        sent_reasoning = safe
                safe = content if completed else content[:max(0, len(content) - holdback)]
                if not safe.startswith(sent_text):
                    raise APIError(500, "Native token decode revised emitted text", "server_error")
                if len(safe) > len(sent_text):
                    await _sse_write(response, st.stream_chunk(
                        response_id, created, model, chat, {"content": safe[len(sent_text):]}))
                    sent_text = safe
            if completed:
                if buffer_tool_calls:
                    if chat and metadata["include_reasoning"] and reasoning:
                        await _sse_write(response, st.stream_chunk(
                            response_id, created, model, True, {"reasoning_content": reasoning}))
                    if calls:
                        await _sse_write(response, st.stream_chunk(
                            response_id, created, model, True, {"tool_calls": [
                                {"index": i, "id": call["id"], "type": "function",
                                 "function": call["function"]} for i, call in enumerate(calls)]}))
                    elif content:
                        await _sse_write(response, st.stream_chunk(
                            response_id, created, model, chat, {"content": content}))
                finish = "tool_calls" if calls else snap["finish_reason"] or "stop"
                await _sse_write(response, st.stream_chunk(response_id, created, model, chat, finish=finish))
                if metadata["include_usage"]:
                    usage = st.count_usage(session)
                    if usage is not None:
                        await _sse_write(response, {"id": response_id,
                            "object": "chat.completion.chunk" if chat else "text_completion",
                            "created": created, "model": model, "choices": [], "usage": usage})
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response
            now = time.monotonic()
            if now - last_ping > 10:
                await response.write(b": native generation active\n\n")
                last_ping = now
            if request.transport is None or request.transport.is_closing():
                raise asyncio.CancelledError()
            await asyncio.sleep(0.07)
    except (ConnectionError, ConnectionResetError, asyncio.CancelledError):
        raise
    except Exception as exc:
        # The stream has already sent 200; send a structured error event instead of
        # trying to overwrite headers with JSON. The caller will cancel the prompt.
        error = {"error": {"message": str(exc), "type": "server_error"}}
        try:
            await response.write(("event: error\ndata: " + json.dumps(error) + "\n\n").encode())
            await response.write_eof()
        except (ConnectionError, ConnectionResetError):
            pass
        session.cancelled = True
        await cancel(prompt_id)
        return response


async def inference(request, chat):
    settings = config.read()
    authed(request, settings)
    # Read with a strict limit: ComfyUI's default aiohttp JSON body limit is
    # too small for SillyTavern's base64 quietImage first-frame payload.
    # Request.content.read() avoids altering the global ComfyUI body limit.
    raw = await request.content.read(ic.MAX_REQUEST_BYTES + 1)
    if len(raw) > ic.MAX_REQUEST_BYTES:
        raise APIError(413, "Chat request exceeds 48 MiB image/context limit")
    try:
        body = json.loads(raw)
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise APIError(400, "Invalid JSON request body") from exc
    model, prompt, effective, stops, metadata = _validate_body(body, settings, chat)
    image_urls = metadata.pop("image_urls", [])
    try:
        images = await ic.resolve_images(image_urls)
    except ic.ImageContextError as exc:
        raise APIError(400, str(exc)) from exc
    rid = str(uuid.uuid4())
    async with _PENDING_LOCK:
        if len(_PENDING) >= settings["max_pending"]:
            raise APIError(429, "Native LLM API queue is full", "rate_limit_error")
        _PENDING.add(rid)
    prompt_id = None
    succeeded = False
    session = None
    try:
        session = streaming.register(rid, thinking=effective["thinking"], stops=stops)
        session.prompt = prompt
        session.images = images
        prompt_id = await submit(make_prompt(model, prompt, effective, rid))
        created = int(time.time())
        response_id = ("chatcmpl-" if chat else "cmpl-") + rid
        if body.get("stream", False):
            result = await _stream_response(request, prompt_id, model, response_id, created,
                                            chat, session, stops, metadata, settings["timeout_seconds"])
            succeeded = True
            return result
        result = await await_output(prompt_id, settings["timeout_seconds"], request)
        content, reasoning = result
        if session.snapshot()["finish_reason"] is None:
            session.finish(content, reasoning, max_tokens=effective["max_tokens"])
        index, _ = streaming.first_stop(content, stops)
        if index is not None:
            content = content[:index]
        answer = _compat_result(content, reasoning, metadata, model, response_id, created, chat, session)
        succeeded = True
        return web.json_response(answer)
    finally:
        if prompt_id and not succeeded:
            if session is not None:
                session.cancelled = True
            await cancel(prompt_id)
        streaming.release(rid)
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
