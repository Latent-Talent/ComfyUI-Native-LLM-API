"""Simulated HTTP SSE test proving output is emitted before native generation ends."""
import ast
import asyncio
import importlib.util
import json
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / "api.py").read_text(encoding="utf-8"))
names = {"APIError", "_compat_result", "_sse_write", "_stream_response"}
subset = ast.fix_missing_locations(ast.Module(
    body=[node for node in source.body
          if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
          and node.name in names], type_ignores=[]))

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

st = load("http_compat_test", "sillytavern.py")
stream = load("http_stream_test", "streaming.py")

class FakeStreamResponse:
    active = None
    def __init__(self, **kwargs):
        self.writes = []
        self.complete = False
        self.headers = kwargs.get("headers", {})
        FakeStreamResponse.active = self
    async def prepare(self, request):
        return self
    async def write(self, data):
        self.writes.append(data.decode("utf-8"))
    async def write_eof(self):
        self.complete = True

class Session:
    def __init__(self):
        self.index = 0
        self.cancelled = False
    def snapshot(self):
        return {"content": "The answer is " + ("more text " * (self.index + 4)),
                "reasoning": "", "completion_tokens": self.index + 4,
                "prompt_tokens": 3, "done": self.index >= 4,
                "finish_reason": "stop", "error": None}
    def finish(self, *_args, **_kwargs):
        pass

class Request:
    transport = types.SimpleNamespace(is_closing=lambda: False)

class TestHTTPStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_emits_content_prior_to_final_result(self):
        session = Session()
        end = "The answer is " + ("more text " * 12)
        end_result_at = [None]
        async def cancel(_):
            session.cancelled = True
        def output(_):
            session.index += 1
            return (end, "") if session.index >= 4 else None

        scope = {
            "asyncio": asyncio, "json": json, "time": time,
            "web": types.SimpleNamespace(StreamResponse=FakeStreamResponse),
            "st": st, "streaming": stream,
            "get_output": output, "cancel": cancel,
        }
        exec(compile(subset, str(ROOT / "api.py"), "exec"), scope)
        meta = {"tools": {}, "tool_choice": "none", "format": "text",
                "include_reasoning": False, "include_usage": True}
        result = await scope["_stream_response"](
            Request(), "prompt-id", "model", "chatcmpl-test", 123, True,
            session, [], meta, 20)
        self.assertTrue(result.complete)
        events = [json.loads(line[6:]) for write in result.writes
                  for line in write.splitlines() if line.startswith("data: {")]
        messages = [ev for ev in events if ev.get("choices") and ev["choices"][0].get("delta", {}).get("content")]
        self.assertGreaterEqual(len(messages), 2, "Must stream before completion, not buffer one final answer")
        combined = "".join(ev["choices"][0]["delta"]["content"] for ev in messages)
        self.assertEqual(combined, end)
        self.assertIn("data: [DONE]", "".join(result.writes))
        usage = [ev.get("usage") for ev in events if ev.get("usage") is not None]
        self.assertEqual(usage[-1]["prompt_tokens"], 3)
        self.assertFalse(session.cancelled)


if __name__ == "__main__":
    unittest.main()
