"""ComfyUI Native LLM API: OpenAI-compatible gateway using native ComfyUI nodes."""
import logging
import os

WEB_DIRECTORY = "./web"
NODE_DISPLAY_NAME_MAPPINGS = {"NativeLLMAPIOutput": "Native LLM API Output"}

class NativeLLMAPIOutput:
    """Capture native TextGenerateV2 output in ComfyUI's prompt history."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"forceInput": True}),
            "request_id": ("STRING", {"default": ""}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "capture"
    CATEGORY = "Native LLM API"
    OUTPUT_NODE = True

    def capture(self, text, request_id):
        return {"ui": {"text": [text]}, "result": (text,)}

NODE_CLASS_MAPPINGS = {"NativeLLMAPIOutput": NativeLLMAPIOutput}

try:
    from .api import register_routes
    register_routes()
except Exception:
    logging.exception("[Native LLM API] Could not register API routes")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
