"""OpenAI/SillyTavern compatibility helpers; native inference remains in ComfyUI."""
import json
import re
import uuid

class CompatibilityError(ValueError):
    pass

FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")

def parse_tool_options(body):
    tools = body.get("tools", body.get("functions", [])) or []
    if not isinstance(tools, list) or len(tools) > 32:
        raise CompatibilityError("tools must be a list of at most 32 functions")
    definitions = {}
    for item in tools:
        if not isinstance(item, dict):
            raise CompatibilityError("Invalid function definition")
        fn = item.get("function", item)
        if item.get("type", "function") != "function" or not isinstance(fn, dict):
            raise CompatibilityError("Only function tools are supported")
        name = fn.get("name")
        if not isinstance(name, str) or not FUNCTION_NAME.fullmatch(name) or name in definitions:
            raise CompatibilityError("Invalid or duplicate function name")
        schema = fn.get("parameters", {"type": "object"})
        if not isinstance(schema, dict) or len(json.dumps(schema)) > 20000:
            raise CompatibilityError("Invalid or oversized parameter schema")
        definitions[name] = {"name": name, "description": str(fn.get("description", ""))[:2000],
                             "parameters": schema}
    choice = body.get("tool_choice", body.get("function_call", "auto"))
    if not definitions or choice is None or choice == "none":
        choice = "none"
    elif choice not in ("auto", "required"):
        if isinstance(choice, dict):
            choice = choice.get("function", choice)
            if isinstance(choice, dict):
                choice = choice.get("name")
        if not isinstance(choice, str) or choice not in definitions:
            raise CompatibilityError("Unknown function in tool_choice")
    return definitions, choice

def tool_instructions(definitions, choice):
    if choice == "none":
        return ""
    names = list(definitions.values())
    mode = "MUST" if choice == "required" or choice in definitions else "MAY"
    restriction = f" Only call the function named {choice}." if choice in definitions else ""
    return ("\n\nAvailable functions: " + json.dumps(names, ensure_ascii=False)
            + f"\nYou {mode} call a function.{restriction}"
            + '\nIf calling a function, respond ONLY with JSON: {"tool_calls":[{"name":"function_name","arguments":{"arg":"value"}}]}.'
            + "\nFollow parameter schemas. If not calling a function, respond normally."
            + "\nNever claim a tool was executed. The API client executes the function.\n")

def parse_tool_calls(text, definitions, choice):
    if choice == "none":
        return None
    raw = text.strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = re.sub(r"^" + fence + r"(?:json)?\s*|\s*" + fence + r"$", "", raw)
    try:
        value = json.loads(raw)
    except ValueError:
        if choice == "auto":
            return None
        raise CompatibilityError("The model did not output a valid tool-call JSON object")
    if not isinstance(value, dict) or "tool_calls" not in value:
        if choice == "auto":
            return None
        raise CompatibilityError("The model did not choose the required function")
    candidates = value["tool_calls"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 8:
        raise CompatibilityError("Invalid tool_calls list")
    out = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise CompatibilityError("Invalid tool call")
        name, args = candidate.get("name"), candidate.get("arguments")
        if name not in definitions or (choice in definitions and name != choice):
            raise CompatibilityError("Model selected an unknown function")
        if not isinstance(args, dict):
            raise CompatibilityError("Function arguments must be a JSON object")
        out.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return out

def validate_json_object(text):
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise CompatibilityError("The native model did not return valid JSON") from exc
    if not isinstance(parsed, dict):
        raise CompatibilityError("The native model returned JSON but not a JSON object")

def format_response(text, reasoning, model, request_id, created, chat, reason, usage=None,
                    include_reasoning=True, tool_calls=None):
    if chat:
        message = {"role": "assistant", "content": None if tool_calls else text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if include_reasoning and reasoning:
            message["reasoning_content"] = reasoning
        output = {"id": request_id, "object": "chat.completion", "created": created,
                  "model": model, "choices": [{"index": 0, "message": message,
                                               "finish_reason": "tool_calls" if tool_calls else reason}]}
    else:
        output = {"id": request_id, "object": "text_completion", "created": created,
                  "model": model, "choices": [{"index": 0, "text": text, "finish_reason": reason}]}
    if usage is not None:
        output["usage"] = usage
    return output

def stream_chunk(request_id, created, model, chat, delta=None, finish=None):
    if chat:
        choice = {"index": 0, "delta": delta or {}, "finish_reason": finish}
        kind = "chat.completion.chunk"
    else:
        choice = {"index": 0, "text": (delta or {}).get("content", ""), "finish_reason": finish}
        kind = "text_completion"
    return {"id": request_id, "object": kind, "created": created, "model": model,
            "choices": [choice]}

def count_usage(session):
    snap = session.snapshot()
    if snap["prompt_tokens"] is None:
        return None
    return {"prompt_tokens": snap["prompt_tokens"],
            "completion_tokens": snap["completion_tokens"],
            "total_tokens": snap["prompt_tokens"] + snap["completion_tokens"]}
