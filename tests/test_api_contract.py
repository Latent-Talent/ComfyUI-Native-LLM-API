"""Tests of the actual HTTP gateway's prompt construction without requiring a GPU."""
import ast
import importlib.util
import json
import secrets
import base64
import io
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
vision = load("native_contract_images", "image_context.py")
config = load("native_contract_config", "native_llm_config.py")
scope = {"json": json, "secrets": secrets, "st": compat,
         "config": config, "ic": vision, "models": lambda: ["Qwen3.5-9B.safetensors", "gemma4.safetensors", "other-model.safetensors"]}
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
        self.assertEqual(graph["2"]["inputs"]["prompt"], "")
        self.assertNotIn("Hello", json.dumps(graph))
        self.assertNotIn("How are you?", json.dumps(graph))
        self.assertEqual(graph["2"]["inputs"]["sampling_mode"], "on")
        self.assertEqual(graph["2"]["inputs"]["sampling_mode.top_k"], 20)
        self.assertNotIsInstance(graph["2"]["inputs"]["sampling_mode"], dict)
        self.assertIn("sampling_mode.seed", graph["2"]["inputs"])
        self.assertEqual(graph["2"]["inputs"]["mtp"], "auto")
        self.assertEqual(graph["3"]["inputs"]["reasoning"], ["2", 1])
        self.assertEqual(stops, ["<END>"])


    def test_dynamic_sampling_combo_comfy_v3_contract(self):
        # V3's get_finalized_class_inputs() uses the root value to select
        # an option, then build_nested_inputs() rebuilds a nested dict.
        for sampling, temperature, expected in ((True, 0.7, "on"),
                                                (False, 0.7, "off"),
                                                (True, 0, "off")):
            with self.subTest(sampling=sampling, temperature=temperature):
                opts = {**self.settings, "sampling": sampling, "temperature": temperature}
                fields = workflow(opts["model"], "Hello", opts, "request")["2"]["inputs"]
                self.assertEqual(fields["sampling_mode"], expected)
                if expected == "on":
                    self.assertIn("sampling_mode.seed", fields)
                    self.assertIn("sampling_mode.temperature", fields)
                    normalized = {"sampling_mode": fields["sampling_mode"]}
                    normalized.update({key.split(".", 1)[1]: value
                                       for key, value in fields.items()
                                       if key.startswith("sampling_mode.")})
                    self.assertEqual(normalized["sampling_mode"], "on")
                    self.assertEqual(normalized["temperature"], temperature)
                else:
                    self.assertFalse(any(key.startswith("sampling_mode.") for key in fields))

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

    def test_sillytavern_quiet_image_first_frame(self):
        # SillyTavern generateQuietPrompt({quietPrompt, quietImage}) attaches
        # the image via Message.addImage() to a SYSTEM quiet prompt.
        # Preserve the user's exact first-frame continuity instructions.
        continuity = (
            "The supplied <Picture 1> is the exact FIRST FRAME of the target video. "
            "Prioritize the image for visible starting conditions and the RP chat "
            "for narrative intentions and subsequent behavior. "
            "Do not reset or reconstruct the first frame."
        )
        image = "data:image/png;base64,QUJD"
        body = {
            "messages": [
                {"role": "system", "content": "RP and character context"},
                {"role": "user", "content": "They walk through a hall."},
                {"role": "system", "content": [
                    {"type": "text", "text": continuity},
                    {"type": "image_url", "image_url": {
                        "url": image, "detail": "high",
                    }},
                ]},
            ],
            "stream": True,
        }
        model, prompt, options, stops, metadata = validate(body, self.settings, True)
        self.assertIn(continuity, prompt)
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<Picture 1>", prompt)
        self.assertEqual(prompt.count("<|image_pad|>"), 1)
        self.assertLess(prompt.index(continuity), prompt.index("<|image_pad|>"))
        self.assertNotIn(image, prompt)
        self.assertEqual(metadata["image_urls"], [image])
        graph = workflow(model, prompt, options, "request-id")
        self.assertEqual(graph["2"]["inputs"]["prompt"], "")
        self.assertNotIn("QUJD", json.dumps(graph))
        self.assertNotIn(continuity, json.dumps(graph))

    def test_quiet_image_disabled_stays_text_only(self):
        instructions = "Continue from <Picture 1> as the exact first frame."
        body = {"messages": [{"role": "system", "content": instructions}]}
        model, prompt, options, stops, metadata = validate(body, self.settings, True)
        self.assertEqual(metadata["image_urls"], [])
        self.assertEqual(prompt.count("<|image_pad|>"), 0)
        self.assertIn(instructions, prompt)

    def test_multiple_images_preserve_order_and_gemma_tokens(self):
        model = "gemma4.safetensors"
        body = {"model": model, "messages": [
            {"role": "system", "content": "Keep continuity."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": "data:image/png;base64,QUJD"},
                {"type": "text", "text": "Now see <Picture 2>"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64,REVG", "detail": "low",
                }},
            ]},
        ]}
        _, prompt, _, _, metadata = validate(body, self.settings, True)
        self.assertTrue(prompt.startswith("<|turn>system"))
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Picture 2>", prompt)
        self.assertEqual(prompt.count("<|image><|image|><image|>"), 2)
        self.assertEqual(metadata["image_urls"], [
            "data:image/png;base64,QUJD", "data:image/jpeg;base64,REVG",
        ])

    def test_unsupported_model_with_image_is_rejected(self):
        with self.assertRaisesRegex(scope["APIError"], "vision-capable"):
            validate({"model": "other-model.safetensors",
                      "messages": [{"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                      ]}]},
                     {**self.settings, "model": "other-model.safetensors"}, True)

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
