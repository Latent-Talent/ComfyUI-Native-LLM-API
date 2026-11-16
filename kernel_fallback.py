"""Narrow workaround for Comfy Kitchen's oversized W4A8 ConvRot shared-memory launch.

Only API executions in the current worker thread use this fallback. Other
ComfyUI workloads preserve the upstream backend. We do not alter kernels,
quantization formats, model loading, MTP, or ComfyUI's CUDA memory manager.

The fused chunked CUDA entrypoint can throw when its dynamic shared memory
request exceeds the device limit (e.g. 110592 bytes on an Ada 4090). For that
specific failure, retry as two existing native Kitchen operations:
ConvRot/Hadamard activation + rowwise INT8 quantization, then W4A8 chunked
dequant/GEMM. This uses the ORIGINAL quantized weights, not BF16 expansion.
"""
import logging
import threading

from . import streaming

LOG = logging.getLogger("native_llm_api.kernel")
_LOCK = threading.Lock()
_INSTALLED = False
_BAD_SHAPES = set()
_FAILURE_FRAGMENT = "convrot fused kernel shared memory request"


def _known_shared_memory_error(exc):
    return (isinstance(exc, RuntimeError)
            and _FAILURE_FRAGMENT in str(exc).lower()
            and "invalid argument" in str(exc).lower())


def _shape_key(args):
    x = args[0]
    return (str(x.device), tuple(x.shape), args[9], args[10], args[11],
            args[12] if len(args) > 12 else None)


def _staged_w4a8(cuda, args, kwargs):
    """Run the native two-pass W4A8 kernel instead of a failing fused launch."""
    from torch import empty, int8
    (x_2d, qdata, s_rel, s_channel, codebook, bias_arg, xq, xs, out,
     convrot_groupsize, group_size, mma_packed, *rest) = args
    input_act = rest[0] if len(rest) > 0 else kwargs.get("input_act")
    act_weight = rest[1] if len(rest) > 1 else kwargs.get("input_act_weight")
    act_eps = rest[2] if len(rest) > 2 else kwargs.get("input_act_eps", 0.0)

    if input_act not in (None, "none"):
        x_2d = cuda._apply_input_act(x_2d, input_act, act_weight, act_eps)

    # A full Hadamard rotation can run with bounded memory even when the
    # monolithic fused shared-memory row quantizer is too large.
    h = cuda._build_hadamard(convrot_groupsize, device=x_2d.device, dtype=x_2d.dtype)
    rotated = cuda._rotate_activation(x_2d, h, convrot_groupsize).contiguous()
    quantized, scales = cuda.quantize_int8_rowwise(rotated)
    xq.copy_(quantized)
    xs.copy_(scales)

    m, k = x_2d.shape
    n = s_channel.numel()
    chunk_cols = cuda._w4a8_weight_chunk_cols(m, n, x_2d.get_device())
    workspace = empty(min(chunk_cols, n), k, dtype=int8, device=x_2d.device)
    wrap = cuda._wrap_for_dlpack
    used = cuda._C.w4a8_codebook_gemm_chunked(
        wrap(xq), wrap(qdata), wrap(s_rel.view(cuda.torch.uint8)),
        wrap(codebook) if codebook is not None else None,
        wrap(s_channel), wrap(xs.reshape(m)),
        wrap(bias_arg) if bias_arg is not None else None,
        wrap(workspace), wrap(out), group_size, chunk_cols,
        cuda.DTYPE_TO_CODE[out.dtype],
        cuda.torch.cuda.current_stream(x_2d.device).cuda_stream, mma_packed,
    )
    return used, xq, xs


def wrap_chunked(original, cuda, fallback=None):
    """Add a failure-scoped fallback; leave unrelated exceptions unchanged."""
    fallback = fallback or _staged_w4a8

    def wrapped(*args, **kwargs):
        if getattr(streaming._THREAD, "capture", None) is None:
            return original(*args, **kwargs)
        key = _shape_key(args)
        with _LOCK:
            fallback_first = key in _BAD_SHAPES
        if not fallback_first:
            try:
                return original(*args, **kwargs)
            except RuntimeError as exc:
                if not _known_shared_memory_error(exc):
                    raise
                with _LOCK:
                    _BAD_SHAPES.add(key)
                LOG.warning(
                    "Native LLM API: W4A8 fused kernel exceeds device shared-memory limits; "
                    "using native two-pass ConvRot + INT8 GEMM fallback for shape %s.",
                    key[1],
                )
        return fallback(cuda, args, kwargs)

    wrapped._native_llm_w4a8_workaround = True
    return wrapped


def install():
    """Patch only the stable Python dispatch point, never the compiled CUDA library."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        import comfy_kitchen.backends.cuda as cuda
    except ImportError:
        return False
    with _LOCK:
        if _INSTALLED:
            return True
        original = getattr(cuda, "_w4a8_linear_chunked", None)
        if original is None:
            LOG.info("Native LLM W4A8 fallback unavailable: Kitchen backend changed.")
            return False
        if not getattr(original, "_native_llm_w4a8_workaround", False):
            cuda._w4a8_linear_chunked = wrap_chunked(original, cuda)
        _INSTALLED = True
    return True
