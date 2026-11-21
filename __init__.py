"""OpenAI-compatible gateway built on ComfyUI's native TextGenerateV2 execution."""
import logging

WEB_DIRECTORY = "./web"

from comfy_api.latest import io
from comfy_extras.nodes_textgen import TextGenerateV2
from .streaming import native_capture, lookup, StopAtSequence, GenerationCancelled, count_bound_images, set_vision_status
from . import kernel_fallback
from . import image_context as ic


class NativeLLMAPITextGenerate(TextGenerateV2):
    """Native V3 node with a request-scoped accepted-token streaming bridge."""

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "NativeLLMAPITextGenerate"
        schema.display_name = "Native LLM API Generate Text"
        schema.category = "Native LLM API"
        schema.inputs.append(io.String.Input("request_id", default="", optional=True))
        return schema

    @classmethod
    def execute(cls, clip, prompt, max_length, sampling_mode, request_id="",
                image=None, thinking=False, use_default_template=True, video=None, audio=None,
                mtp="auto", system_prompt="", images=None, videos=None, audios=None):
        session = lookup(request_id)
        if not request_id or session is None or session.prompt is None:
            raise RuntimeError("Native LLM API request expired or unavailable")
        # Retrieve private text by opaque ID, never from history-visible workflow inputs.
        prompt = session.prompt
        # Convert images only inside the native worker. The workflow carries an
        # opaque request id, not JPEG/PNG bytes or GPU tensor data.
        images = ic.image_tensors(session.images) if session.images else None
        # Configure a native two-pass fallback only for a known Kitchen CUDA launch error.
        kernel_fallback.install()
        # Also capture final accepted IDs: MTP can finish without a progress update.
        original_generate = clip.generate
        original_tokenize = clip.tokenize

        def capturing_generate(*args, **kwargs):
            ids = original_generate(*args, **kwargs)
            if session is not None and isinstance(ids, list):
                session.feed(ids, clip.decode, force=True)
            return ids

        def capturing_tokenize(*args, **kwargs):
            tokens = original_tokenize(*args, **kwargs)
            if session is not None:
                from .streaming import _prompt_token_count
                count = _prompt_token_count(tokens)
                expected = len(session.images)
                bound = count_bound_images(tokens) if expected else 0
                with session.lock:
                    session.images_bound = bound
                    if count is not None:
                        session.prompt_tokens = count
                if expected:
                    if bound != expected:
                        set_vision_status(request_id, embedded_images=bound,
                                          stage="tokenizer_missing_images")
                        raise RuntimeError(
                            "Native vision tokenization failed: received "
                            f"{expected} image(s) but tokenizer bound {bound}. "
                            "The selected model may lack a compatible vision tokenizer."
                        )
                    set_vision_status(request_id, embedded_images=bound,
                                      stage="vision_embeddings_verified")
            return tokens

        if session is not None:
            clip.generate = capturing_generate
            clip.tokenize = capturing_tokenize
        try:
          with native_capture(request_id, clip):
            try:
                result = TextGenerateV2.execute(
                    clip=clip, prompt=prompt, max_length=max_length,
                    sampling_mode=sampling_mode, image=image, thinking=thinking,
                    use_default_template=use_default_template, video=video, audio=audio,
                    mtp=mtp, system_prompt=system_prompt,
                    images=images, videos=videos, audios=audios,
                )
                if session is not None:
                    session.finish(result[0], result[1], max_tokens=max_length)
                    if session.images:
                        set_vision_status(request_id, stage="completed_with_vision")
                return result
            except StopAtSequence:
                # Native generate's finally block has run, including cache cleanup.
                # Only actual committed tokens are available in the session.
                if session is None:
                    raise
                snap = session.snapshot()
                session.finish(snap["content"], snap["reasoning"], max_tokens=max_length)
                return io.NodeOutput(snap["content"], snap["reasoning"])
            except GenerationCancelled:
                if session is not None:
                    session.fail("Client disconnected")
                raise RuntimeError("Native LLM API request cancelled")
            except Exception as error:
                if session is not None:
                    session.fail(str(error))
                    if session.images:
                        # Preserve the more specific missing-embedding stage.
                        if session.images_bound == len(session.images):
                            set_vision_status(request_id, stage="generation_failed")
                raise
        finally:
            if session is not None:
                clip.generate = original_generate
                clip.tokenize = original_tokenize


class NativeLLMAPIOutput:
    """Write final assistant/reasoning outputs into ComfyUI's native prompt history."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"forceInput": True}),
            "reasoning": ("STRING", {"forceInput": True}),
            "request_id": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning")
    FUNCTION = "capture"
    CATEGORY = "Native LLM API"
    OUTPUT_NODE = True

    def capture(self, text, reasoning, request_id):
        # This output node is intentionally history-safe. Native ComfyUI persists
        # OUTPUT_NODE UI fields and workflow inputs in /history; no chat content
        # or reasoning is allowed here.
        return {"ui": {"status": ["completed"]}, "result": ("completed", "")}


NODE_CLASS_MAPPINGS = {
    "NativeLLMAPITextGenerate": NativeLLMAPITextGenerate,
    "NativeLLMAPIOutput": NativeLLMAPIOutput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NativeLLMAPITextGenerate": "Native LLM API Generate Text",
    "NativeLLMAPIOutput": "Native LLM API Output",
}

try:
    from .api import register_routes
    register_routes()
except Exception:
    logging.exception("[Native LLM API] Could not register API routes")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
