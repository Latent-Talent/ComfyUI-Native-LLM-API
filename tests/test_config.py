import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "native_llm_config.py"

class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.old = os.environ.get("COMFY_NATIVE_LLM_API_CONFIG")
        os.environ["COMFY_NATIVE_LLM_API_CONFIG"] = str(Path(self.tempdir.name) / "settings.json")
        spec = importlib.util.spec_from_file_location("native_llm_config_test", SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def tearDown(self):
        if self.old is None:
            os.environ.pop("COMFY_NATIVE_LLM_API_CONFIG", None)
        else:
            os.environ["COMFY_NATIVE_LLM_API_CONFIG"] = self.old
        self.tempdir.cleanup()

    def test_creates_key_and_persists(self):
        a = self.module.read()
        b = self.module.read()
        self.assertEqual(a["api_key"], b["api_key"])
        self.assertGreaterEqual(len(a["api_key"]), 32)
        self.assertEqual(json.loads(self.module.CONFIG_PATH.read_text())["api_key"], a["api_key"])

    def test_update_and_rotate(self):
        original = self.module.read()["api_key"]
        updated = self.module.update({"top_p": 0.8, "mtp": "tree2s1", "thinking": True})
        self.assertEqual(updated["top_p"], 0.8)
        self.assertTrue(updated["thinking"])
        self.assertEqual(self.module.read()["mtp"], "tree2s1")
        new = self.module.update({"rotate_api_key": True})
        self.assertNotEqual(original, new["api_key"])

    def test_rejects_unknown_and_invalid(self):
        for bad in ({"unexpected": True}, {"max_tokens": -1}, {"top_k": 1.3},
                    {"sampling": 1}, {"mtp": "unsupported"}, {"model": 123}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.module.update(bad)

if __name__ == "__main__":
    unittest.main()
