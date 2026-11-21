"""Inspect checkpoint headers, not filenames, for native vision weights.

ComfyUI's Qwen3.5/3.8 model class creates a visual module even for
text-only/pruned checkpoints, so architecture detection alone is insufficient.
This uses safetensors header indices, not actual weight reads.
"""

VISION_PREFIXES = {
    "qwen": ("model.visual.", "visual.", "language_model.visual."),
    "gemma": ("vision_model.", "model.vision_model."),
}


def classify_vision_weights(tensor_names, model_name):
    """Return (status, explanation) without logging names or tensor contents."""
    name = model_name.lower()
    if any(token in name for token in ("qwen35", "qwen3.5", "qwen38", "qwen3.8")):
        expected = VISION_PREFIXES["qwen"]
    elif "gemma4" in name:
        expected = VISION_PREFIXES["gemma"]
    else:
        return "unknown", "Model architecture does not establish vision support"
    present = any(any(key.startswith(prefix) for prefix in expected) for key in tensor_names)
    if not present:
        return "missing", "Checkpoint contains no detected visual-encoder tensors"
    return "present", "Visual-encoder tensor names found in checkpoint"


def inspect_checkpoint(model_name, get_model_path):
    """Return 'present', 'missing' or 'unknown'; never read tensor data."""
    import os
    path = get_model_path("text_encoders", model_name)
    if not path or not os.path.isfile(path):
        return "unknown", "Checkpoint path unavailable for vision inspection"
    if not path.lower().endswith((".safetensors", ".sft")):
        return "unknown", "Checkpoint format does not support lightweight vision inspection"
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as file:
            return classify_vision_weights(file.keys(), model_name)
    except (OSError, ValueError, RuntimeError) as exc:
        # Never expose path or other sensitive exception details to clients.
        return "unknown", "Could not inspect checkpoint vision tensor header"
