"""Tests of the actual HTTP gateway's prompt construction without requiring a GPU."""
import ast
import importlib.util
import json
import secrets
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "api.py").read_text(encoding="utf-8"))
chosen = [node for node in tree.body
          if isinstance(node, (ast.FunctionDef, ast.ClassDef))
          and node.name in {"APIError", "render_chat", "_validate_body", "make_prompt"}]
subset = ast.fix_missing_locations(ast.Module(body=chosen, type_ignores=[]))

def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

compat = load("native_contract_compat", "sillytavern.py")
config = load("native_contract_config", "native_llm_config.py")
scope = {"json": json, "secrets": secrets, "st": compat,
         "config": config, "models": lambda: ["Qwen3.5-9B.safetensors", "gemma4.safetensors"]}
exec(compile(subset, str(ROOT / "api.py"), "exec"), scope)
validate = scope["_validate_body"]
workflow = scope["make_prompt"]


class APITests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(config.DEFAULTS, model="Qwen3.5-9B.safetensors")
        self.messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]

    def test_native_workflow_and_multiturn(self):
        model, prompt, options, stops, meta = validate(
            {"messages": self.messages, "stop": ["<END>"], "max_tokens": 18,
             "top_k": 20, "temperature": 0.4, "mtp": "auto"}, self.settings, True)
        self.assertTrue(prompt.startswith("<|im_start|>"))
        self.assertIn("<|im_start|>user\nHow are you?<|im_end|>", prompt)
        graph = workflow(model, prompt, options, "request")
        self.assertEqual(graph["2"]["class_type"], "NativeLLMAPITextGenerate")
        self.assertEqual(graph["2"]["inputs"]["sampling_mode"]["top_k"], 20)
        self.assertEqual(graph["2"]["inputs"]["mtp"], "auto")
        self.assertEqual(graph["3"]["inputs"]["reasoning"], ["2", 1])
        self.assertEqual(stops, ["<END>"])

    def test_tools_preserve_qwen_native_template(self):
        body = {
            "messages": self.messages,
            "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
            "tool_choice": "required", "stream": True,
        }
        _, prompt, _, _, metadata = validate(body, self.settings, True)
        self.assertTrue(prompt.startswith("<|im_start|>system\n"))
        self.assertIn("weather", prompt)
        self.assertIn("tool_calls", prompt)
        self.assertEqual(metadata["tool_choice"], "required")

    def test_json_and_optional_openai_parameters(self):
        _, _, _, stops, metadata = validate({
            "messages": self.messages, "response_format": {"type": "json_object"},
            "logprobs": False, "top_logprobs": 0, "stop": None,
            "frequency_penalty": 0,
        }, self.settings, True)
        self.assertEqual(stops, [])
        self.assertEqual(metadata["format"], "json_object")

    def test_invalid_inputs_fail(self):
        cases = [
            {"n": 2}, {"temperature": 3}, {"top_k": 1.1}, {"mtp": "invalid"},
            {"frequency_penalty": 0.3}, {"seed": -1},
            {"response_format": {"type": "json_schema", "json_schema": {}}},
            {"stop": ["X" * 129]},
        ]
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(scope["APIError"]):
                validate({"messages": self.messages, **extra}, self.settings, True)


if __name__ == "__main__":
    unittest.main()
