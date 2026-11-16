"""Exercise native accepted-token streaming and the SillyTavern wire helpers."""
import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

stream = load("native_llm_stream_test", "streaming.py")
compat = load("native_llm_compat_test", "sillytavern.py")


class TestStreaming(unittest.TestCase):
    def test_non_reasoning(self):
        self.assertEqual(stream.split_reasoning("hello"), ("hello", ""))
        self.assertEqual(stream.split_reasoning("<think>plan</think>answer", True),
                         ("answer", "plan"))
        self.assertEqual(stream.split_reasoning("partial thinking", True),
                         ("", "partial thinking"))

    def test_stops_are_early_and_safe(self):
        session = stream.CaptureSession("s", stops=["STOP"])
        with self.assertRaises(stream.StopAtSequence):
            session.feed([1, 2, 3], lambda ids: "helloSTOP trailing", 4)
        snap = session.snapshot()
        self.assertEqual(snap["content"], "hello")
        self.assertEqual(snap["completion_tokens"], 3)
        self.assertEqual(snap["prompt_tokens"], 4)
        self.assertEqual(snap["finish_reason"], "stop")
        session.finish("hello", "", max_tokens=4)
        self.assertEqual(session.snapshot()["finish_reason"], "stop")

    def test_mtp_committed_ids_and_standard_ids(self):
        fake_comfy = types.ModuleType("comfy")
        utils = types.ModuleType("comfy.utils")
        class ProgressBar:
            def __init__(self):
                self.current = 0
            def update(self, n):
                self.current += n
        utils.ProgressBar = ProgressBar
        fake_comfy.utils = utils
        old_comfy = sys.modules.get("comfy")
        old_utils = sys.modules.get("comfy.utils")
        sys.modules["comfy"] = fake_comfy
        sys.modules["comfy.utils"] = utils
        previous_installed = stream._INSTALLED
        stream._INSTALLED = False
        try:
            stream.install_progress_tap()
            # Execute code under upstream module names so the capture contract is tested.
            base_scope = {"__name__": "comfy.text_encoders.llama"}
            exec("def generate(progress):\n"
                 "    generated_token_ids = []\n"
                 "    for value in (1, 2, 3):\n"
                 "        generated_token_ids.append(value)\n"
                 "        progress.update(1)\n", base_scope)
            mtp_scope = {"__name__": "comfy.text_encoders.qwen35"}
            exec("def _generate_mtp(progress):\n"
                 "    ids = [1]\n"
                 "    def update_progress(n):\n"
                 "        progress.update(n)\n"
                 "    ids.extend([2, 3, 4])\n"
                 "    update_progress(3)\n", mtp_scope)
            for operation, expected in ((base_scope["generate"], 3), (mtp_scope["_generate_mtp"], 4)):
                rid = str(id(operation))
                session = stream.register(rid)
                class Clip:
                    @staticmethod
                    def decode(values):
                        return "".join(chr(64 + i) for i in values)
                with stream.native_capture(rid, Clip()):
                    progress = utils.ProgressBar()
                    operation(progress)
                self.assertEqual(session.snapshot()["content"], "ABC" if expected == 3 else "ABCD")
                self.assertEqual(session.snapshot()["completion_tokens"], expected)
                self.assertEqual(progress.current, 3)
                stream.release(rid)
            untouched = utils.ProgressBar()
            untouched.update(2)
            self.assertEqual(untouched.current, 2)
        finally:
            stream._INSTALLED = previous_installed
            if old_comfy is None:
                sys.modules.pop("comfy", None)
            else:
                sys.modules["comfy"] = old_comfy
            if old_utils is None:
                sys.modules.pop("comfy.utils", None)
            else:
                sys.modules["comfy.utils"] = old_utils

    def test_prompt_tokens(self):
        count = stream._prompt_token_count({"tokenizer": [[(11, 1.0), (22, 1.0)]]})
        self.assertEqual(count, 2)
        self.assertIsNone(stream._prompt_token_count({"tokenizer": [[({"type": "image"},)]]}))


class TestCompatibility(unittest.TestCase):
    def test_function_call(self):
        defs, choice = compat.parse_tool_options({
            "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
            "tool_choice": "required",
        })
        self.assertEqual(choice, "required")
        calls = compat.parse_tool_calls('{"tool_calls":[{"name":"weather","arguments":{"city":"Paris"}}]}', defs, choice)
        self.assertEqual(calls[0]["function"]["name"], "weather")
        self.assertIn("Paris", calls[0]["function"]["arguments"])
        message = compat.format_response("", "", "model", "chatcmpl-1", 1, True, "stop", tool_calls=calls)
        self.assertEqual(message["choices"][0]["finish_reason"], "tool_calls")
        self.assertIsNone(message["choices"][0]["message"]["content"])

    def test_tool_call_invalid_and_text_fallback(self):
        defs = {"search": {"name": "search", "parameters": {}}}
        self.assertIsNone(compat.parse_tool_calls("A normal assistant answer", defs, "auto"))
        with self.assertRaises(compat.CompatibilityError):
            compat.parse_tool_calls("not json", defs, "required")

    def test_json_mode(self):
        compat.validate_json_object('{"ok":true}')
        with self.assertRaises(compat.CompatibilityError):
            compat.validate_json_object("[1,2]")
        with self.assertRaises(compat.CompatibilityError):
            compat.validate_json_object("unstructured")

    def test_stream_chunk(self):
        chunk = compat.stream_chunk("id", 1, "model", True, {"content": "Hello"})
        self.assertEqual(chunk["choices"][0]["delta"]["content"], "Hello")
        self.assertEqual(chunk["object"], "chat.completion.chunk")
        result = compat.format_response("hi", "reason", "model", "id", 1, True, "stop",
                                        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
        self.assertEqual(result["choices"][0]["message"]["reasoning_content"], "reason")
        self.assertEqual(result["usage"]["total_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
