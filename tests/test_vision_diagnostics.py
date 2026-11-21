"""Verify image bind detection and checkpoint visual-weight inspection."""
import importlib.util
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

vision = load("native_vision_support_tests", "vision_support.py")
streaming = load("native_streaming_vision_tests", "streaming.py")


class VisionTest(unittest.TestCase):
    def test_actual_image_embeddings_not_text_placeholders(self):
        tokens = {"qwen": [[(1001, 1, 0), ({"type": "image", "data": object()}, 1, 0)]]}
        self.assertEqual(streaming.count_bound_images(tokens), 1)
        placeholders = {"qwen": [[(1001, 1, 0), (1002, 1, 0), (1003, 1, 0)]]}
        self.assertEqual(streaming.count_bound_images(placeholders), 0)
        self.assertEqual(streaming._prompt_token_count(tokens), None)
        self.assertEqual(streaming._prompt_token_count(placeholders), 3)

    def test_visual_weights_present_and_absent(self):
        self.assertEqual(vision.classify_vision_weights(
            ["model.language_model.layers.0.weight", "model.visual.patch_embed.proj.weight"],
            "qwen3.8_27b_uncensored_w4a8_convrotMTP.safetensors")[0], "present")
        self.assertEqual(vision.classify_vision_weights(
            ["model.language_model.layers.0.weight", "mtp.fc.weight"],
            "qwen3.8_27b_uncensored_w4a8_convrotMTP.safetensors")[0], "missing")
        self.assertEqual(vision.classify_vision_weights(
            ["model.layers.0.weight", "vision_model.encoder.layers.0.weight"],
            "gemma4.safetensors")[0], "present")
        self.assertEqual(vision.classify_vision_weights(
            ["other.weight"], "unrecognized.safetensors")[0], "unknown")

    def test_last_request_status_has_only_numbers_enums_and_no_chat(self):
        streaming.set_vision_status("one", start=True, image_parts=1, stage="image_received")
        streaming.set_vision_status("one", decoded_images=1, embedded_images=1,
                                   checkpoint_vision="present", stage="completed_with_vision")
        status = streaming.latest_vision_status()
        self.assertEqual(status["image_parts"], 1)
        self.assertEqual(status["embedded_images"], 1)
        self.assertEqual(status["stage"], "completed_with_vision")
        self.assertNotIn("one", str(status))
        # A later text-only request must not be overwritten by a slow prior job.
        streaming.set_vision_status("two", start=True, image_parts=0, stage="text_only")
        streaming.set_vision_status("one", embedded_images=9, stage="failed")
        status = streaming.latest_vision_status()
        self.assertEqual(status["image_parts"], 0)
        self.assertEqual(status["embedded_images"], 0)
        self.assertEqual(status["stage"], "text_only")

if __name__ == "__main__":
    unittest.main()
