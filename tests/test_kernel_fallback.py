"""Unit tests for the W4A8 shared-memory fallback and request-only activation."""
import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "test_native_llm_package"

# Load the package-level relative import without importing ComfyUI, Torch or CUDA.
pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = pkg

def load(module):
    name = PACKAGE + "." + module
    spec = importlib.util.spec_from_file_location(name, ROOT / (module + ".py"))
    instance = importlib.util.module_from_spec(spec)
    sys.modules[name] = instance
    spec.loader.exec_module(instance)
    return instance

streaming = load("streaming")
kernel = load("kernel_fallback")


class FakeArray:
    device = "cuda:0"
    shape = (1, 12288)


class KernelFallbackTests(unittest.TestCase):
    def setUp(self):
        kernel._BAD_SHAPES.clear()
        streaming._THREAD.capture = None
        self.args = (FakeArray(), None, None, None, None, None, None,
                     None, None, 256, 16, False, None)

    def tearDown(self):
        streaming._THREAD.capture = None
        kernel._BAD_SHAPES.clear()

    def test_known_error_falls_back_and_caches_bad_shape(self):
        called = []
        def broken(*a, **k):
            called.append("fused")
            raise RuntimeError(
                "convrot fused kernel shared memory request (110592 bytes) failed: invalid argument"
            )
        def staged(cuda, a, k):
            called.append("staged")
            return True, "q", "scales"
        wrapped = kernel.wrap_chunked(broken, object(), staged)
        # Non-API executions are not modified.
        with self.assertRaises(RuntimeError):
            wrapped(*self.args)
        self.assertEqual(called, ["fused"])
        streaming._THREAD.capture = ("session", "decode")
        self.assertEqual(wrapped(*self.args), (True, "q", "scales"))
        self.assertEqual(wrapped(*self.args), (True, "q", "scales"))
        self.assertEqual(called, ["fused", "fused", "staged", "staged"])

    def test_other_kernel_errors_are_not_swallowed(self):
        streaming._THREAD.capture = ("session", "decode")
        def fail(*args, **kwargs):
            raise RuntimeError("CUDA illegal memory access")
        wrapped = kernel.wrap_chunked(fail, object(), lambda *_: "not expected")
        with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
            wrapped(*self.args)
        self.assertEqual(len(kernel._BAD_SHAPES), 0)

    def test_unaffected_kernel_keeps_native_path(self):
        streaming._THREAD.capture = ("session", "decode")
        def native(*args, **kwargs):
            return True, "native", "scales"
        wrapped = kernel.wrap_chunked(native, object(), lambda *_: "fallback")
        self.assertEqual(wrapped(*self.args), (True, "native", "scales"))
        self.assertEqual(len(kernel._BAD_SHAPES), 0)


if __name__ == "__main__":
    unittest.main()
