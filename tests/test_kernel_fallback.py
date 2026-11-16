"""Unit tests for the W4A8 shared-memory fallback and request-only activation."""
import importlib.util
import sys
import threading
import types
import unittest
from unittest.mock import patch
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

    def test_two_pass_kernel_fallback_uses_existing_quantized_ops(self):
        calls = []
        class Tensor:
            def __init__(self, shape, dtype="bf16"):
                self.shape, self.dtype, self.device = tuple(shape), dtype, "cuda:0"
            def contiguous(self):
                return self
            def get_device(self):
                return 0
            def numel(self):
                count = 1
                for n in self.shape:
                    count *= n
                return count
            def copy_(self, other):
                calls.append(("copy", other.shape))
                return self
            def view(self, dtype):
                return self
            def reshape(self, *dims):
                return Tensor(dims, self.dtype)

        fake_torch = types.SimpleNamespace(
            int8="int8", uint8="uint8",
            empty=lambda *shape, **kw: Tensor(shape, kw.get("dtype", "bf16")),
            cuda=types.SimpleNamespace(current_stream=lambda device: types.SimpleNamespace(cuda_stream=7)),
        )

        class CUDAKernel:
            @staticmethod
            def w4a8_codebook_gemm_chunked(*args):
                calls.append(("gemm", len(args)))
                return True

        cuda = types.SimpleNamespace(
            torch=fake_torch, _C=CUDAKernel(), DTYPE_TO_CODE={"bf16": 3},
            _build_hadamard=lambda *a, **k: "hadamard",
            _rotate_activation=lambda x, h, groupsize: x,
            quantize_int8_rowwise=lambda x: (Tensor(x.shape, "int8"), Tensor((2, 1), "float32")),
            _w4a8_weight_chunk_cols=lambda m, n, device: 512,
            _wrap_for_dlpack=lambda x: x,
        )
        x = Tensor((2, 512))
        args = (
            x, Tensor((128, 256), "int8"), Tensor((128, 32), "float8"),
            Tensor((128,)), None, None, Tensor((2, 512), "int8"),
            Tensor((2, 1), "float32"), Tensor((2, 128)),
            256, 16, False, None,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            success, xq, scales = kernel._staged_w4a8(cuda, args, {})
        self.assertTrue(success)
        self.assertEqual(xq.shape, (2, 512))
        self.assertEqual(scales.shape, (2, 1))
        self.assertEqual([item[0] for item in calls], ["copy", "copy", "gemm"])
        self.assertEqual(calls[-1], ("gemm", 14))

    def test_unaffected_kernel_keeps_native_path(self):
        streaming._THREAD.capture = ("session", "decode")
        def native(*args, **kwargs):
            return True, "native", "scales"
        wrapped = kernel.wrap_chunked(native, object(), lambda *_: "fallback")
        self.assertEqual(wrapped(*self.args), (True, "native", "scales"))
        self.assertEqual(len(kernel._BAD_SHAPES), 0)


if __name__ == "__main__":
    unittest.main()
