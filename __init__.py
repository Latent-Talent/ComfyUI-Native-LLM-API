"""OpenAI-compatible gateway built on ComfyUI's native TextGenerateV2 execution."""
import logging

WEB_DIRECTORY = "./web"

from comfy_api.latest import io
from comfy_extras.nodes_textgen import TextGenerateV2
from .streaming import native_capture, lookup, StopAtSequence, GenerationCancelled


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
                raise


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
        return {"ui": {"text": [text], "reasoning": [reasoning]}, "result": (text, reasoning)}


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
