"""Prove the native worker receives private first-frame image tensors, not graph payloads."""
import ast
import types
import threading
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_test_streaming", ROOT / "streaming.py")
native_streaming = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_streaming)


class NativeVisionBridgeTests(unittest.TestCase):
    def test_session_frame_reaches_native_generate_text(self):
        parsed = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        node = next(node for node in parsed.body
                    if isinstance(node, ast.ClassDef) and node.name == "NativeLLMAPITextGenerate")
        calls = []
        session = types.SimpleNamespace(
            prompt="The supplied <Picture 1> is the exact FIRST FRAME.",
            images=[b"opaque compressed frame bytes"],
            images_bound=0,
            lock=threading.Lock(),
            prompt_tokens=None,
            finish=lambda *a, **kw: calls.append(("finish", a)),
            fail=lambda *a, **kw: calls.append(("fail", a)),
        )

        class StubTextGenerate:
            @classmethod
            def execute(cls, **kwargs):
                calls.append(("execute", kwargs))
                kwargs["clip"].tokenize(kwargs["prompt"], images=kwargs["images"])
                return ("completed prompt", "")

        class NoopCapture:
            def __init__(self, *_args):
                pass
            def __enter__(self):
                return session
            def __exit__(self, *_args):
                return False

        def convert(private_bytes):
            self.assertEqual(private_bytes, [b"opaque compressed frame bytes"])
            return {"image1": "PRIVATE_IMAGE_TENSOR"}

        scope = {
            "TextGenerateV2": StubTextGenerate,
            "lookup": lambda rid: session if rid == "id" else None,
            "ic": types.SimpleNamespace(image_tensors=convert),
            "count_bound_images": native_streaming.count_bound_images,
            "_prompt_token_count": native_streaming._prompt_token_count,
            "set_vision_status": lambda *args, **kwargs: calls.append(("vision", kwargs)),
            "native_capture": NoopCapture,
            "kernel_fallback": types.SimpleNamespace(install=lambda: None),
            "io": types.SimpleNamespace(NodeOutput=lambda *args: args),
            "StopAtSequence": type("StopAtSequence", (Exception,), {}),
            "GenerationCancelled": type("GenerationCancelled", (Exception,), {}),
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(ROOT / "__init__.py"), "exec"), scope)
        clip = types.SimpleNamespace(
            generate=lambda *args, **kwargs: [],
            tokenize=lambda *args, **kwargs: {"test": [[({"type": "image", "data": "pixels"}, 1, 0)]]},
            decode=lambda ids: "decoded",
        )
        orig_generate, orig_tokenize = clip.generate, clip.tokenize
        outcome = scope["NativeLLMAPITextGenerate"].execute(
            clip=clip, prompt="", max_length=32, sampling_mode={"sampling_mode": "off"},
            request_id="id",
        )
        self.assertEqual(outcome, ("completed prompt", ""))
        self.assertEqual(calls[0][0], "execute")
        self.assertEqual(calls[0][1]["prompt"], session.prompt)
        self.assertEqual(calls[0][1]["images"], {"image1": "PRIVATE_IMAGE_TENSOR"})
        self.assertEqual(calls[-2][0], "finish")
        self.assertEqual(calls[-1][1]["stage"], "completed_with_vision")
        self.assertEqual(session.images_bound, 1)
        self.assertIs(clip.generate, orig_generate)
        self.assertIs(clip.tokenize, orig_tokenize)

        # A tokenizer that keeps the placeholder as text must never return an
        # apparently successful answer while silently ignoring the image.
        session.images_bound = 0
        text_only_clip = types.SimpleNamespace(
            generate=lambda *args, **kwargs: [],
            tokenize=lambda *args, **kwargs: {"test": [[(123, 1, 0), (456, 1, 0)]]},
            decode=lambda ids: "decoded",
        )
        with self.assertRaisesRegex(RuntimeError, "tokenizer bound 0"):
            scope["NativeLLMAPITextGenerate"].execute(
                clip=text_only_clip, prompt="", max_length=32,
                sampling_mode={"sampling_mode": "off"}, request_id="id",
            )
        self.assertEqual(calls[-1][1]["stage"], "tokenizer_missing_images")


if __name__ == "__main__":
    unittest.main()
