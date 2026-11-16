"""Privacy regression tests for ComfyUI queued prompts and /history output."""
import ast
import importlib.util
import json
import secrets
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def isolate_function(source_path, name, scope):
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    matches = [x for x in module.body if isinstance(x, ast.FunctionDef) and x.name == name]
    if len(matches) != 1:
        raise AssertionError("Expected function " + name)
    unit = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    exec(compile(unit, str(source_path), "exec"), scope)
    return scope[name]


def isolate_class(source_path, name):
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = [x for x in module.body if isinstance(x, ast.ClassDef) and x.name == name]
    if len(classes) != 1:
        raise AssertionError("Expected class " + name)
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])),
                 str(source_path), "exec"), scope)
    return scope[name]


class TestPrivacyContract(unittest.TestCase):
    def test_output_node_returns_only_status_not_chat(self):
        capture_node = isolate_class(ROOT / "__init__.py", "NativeLLMAPIOutput")()
        outcome = capture_node.capture(
            "PRIVATE user-specific generated response", "PRIVATE reasoning", "request-guid"
        )
        self.assertEqual(outcome["ui"], {"status": ["completed"]})
        self.assertEqual(outcome["result"], ("completed", ""))
        self.assertNotIn("PRIVATE", json.dumps(outcome))

    def test_history_contains_no_chat_content_but_api_reads_private_result(self):
        secret = "PRIVATE user-specific generated response"
        rid = "request-guid"
        history = {
            "prompt-1": {
                "status": {"status_str": "success"},
                "prompt": [0, "prompt-1", {"3": {"inputs": {"request_id": rid}}}, {}],
                "outputs": {"3": {"status": ["completed"]}},
            }
        }
        private_session = types.SimpleNamespace(snapshot=lambda: {
            "generation_completed": True, "content": secret, "reasoning": "PRIVATE thought"
        })
        scoped = {
            "PromptServer": types.SimpleNamespace(instance=types.SimpleNamespace(
                prompt_queue=types.SimpleNamespace(get_history=lambda prompt_id: history)
            )),
            "streaming": types.SimpleNamespace(lookup=lambda request_id: private_session
                if request_id == rid else None),
        }
        isolate_class(ROOT / "api.py", "APIError")
        exception_scope = {}
        exception_scope["APIError"] = isolate_class(ROOT / "api.py", "APIError")
        scoped.update(exception_scope)
        get_output = isolate_function(ROOT / "api.py", "get_output", scoped)
        self.assertEqual(get_output("prompt-1"), (secret, "PRIVATE thought"))
        self.assertNotIn("PRIVATE", json.dumps(history))

    def test_prompt_graph_cannot_expose_sensitive_message(self):
        # The native sampler's prompt is resolved from request-scoped memory,
        # not a text field in the queued ComfyUI workflow.
        init = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("prompt = session.prompt", init)
        api = (ROOT / "api.py").read_text(encoding="utf-8")
        self.assertIn('session.prompt = prompt', api)
        self.assertIn('"prompt": ""', api)


if __name__ == "__main__":
    unittest.main()
