"""Stream accepted native tokens without changing the model's sampler or VRAM behavior.

ComfyUI's BaseGenerate and Qwen35 MTP generator both call ProgressBar.update()
AFTER committing accepted token ids. A guarded, thread-local tap observes those ids.
No model forward methods, samplers, or CUDA graphs are replaced. Fail closed if a
future upstream version changes either generator's progress contract.
"""
import logging
import sys
import threading
import time

LOG = logging.getLogger("native_llm_api.streaming")
REGISTRY = {}
_REGISTRY_LOCK = threading.RLock()
_THREAD = threading.local()
_INSTALLED = False

class StopAtSequence(Exception):
    """Intentional early termination after a user stop string."""

class GenerationCancelled(Exception):
    """Intentional termination of abandoned API work."""

def split_reasoning(raw, thinking=False):
    """Return (content, reasoning), protecting incomplete reasoning markers."""
    before, separator, after = raw.partition("</think>")
    if separator:
        return after.strip(), before.replace("<think>", "", 1).strip()
    if before.lstrip().startswith("<think>") or thinking:
        return "", before.replace("<think>", "", 1).strip()
    return raw.strip(), ""

def first_stop(text, stops):
    positions = [(text.index(s), s) for s in stops if s and s in text]
    return min(positions, default=(None, None), key=lambda p: p[0])

class CaptureSession:
    """One HTTP request. Latest snapshot only: no unbounded token queues."""
    def __init__(self, request_id, *, thinking=False, stops=()):
        self.request_id = request_id
        self.thinking = thinking
        self.stops = tuple(stops)
        self.lock = threading.Lock()
        self.raw = ""
        self.content = ""
        self.reasoning = ""
        self.completion_tokens = 0
        self.prompt_tokens = None
        self.done = False
        self.finish_reason = None
        self.cancelled = False
        self.error = None
        self.generation_completed = False
        self.last_decoded_at = 0.0

    def feed(self, token_ids, decoder, prompt_tokens=None, force=False):
        if self.cancelled:
            raise GenerationCancelled()
        if not token_ids:
            return
        if len(token_ids) <= self.completion_tokens:
            return
        now = time.monotonic()
        # Keep tokenizer decoding off the GPU hot path where possible; cap it
        # near 20 Hz while still observing every MTP commit and stop sequence.
        if not force and self.last_decoded_at and now - self.last_decoded_at < 0.05 and len(token_ids) - self.completion_tokens < 3:
            return
        self.last_decoded_at = now
        # Decode actual accepted model tokens, including MTP's multi-token commits.
        raw = decoder(token_ids)
        content, reasoning = split_reasoning(raw, self.thinking)
        index, _ = first_stop(content, self.stops)
        if index is not None:
            content = content[:index]
        with self.lock:
            self.raw = raw
            self.content = content
            self.reasoning = reasoning
            self.completion_tokens = len(token_ids)
            if self.prompt_tokens is None and prompt_tokens is not None:
                self.prompt_tokens = prompt_tokens
            if index is not None:
                self.finish_reason = "stop"
        if index is not None:
            raise StopAtSequence()

    def finish(self, text, reasoning="", *, max_tokens=None):
        """Native final output always wins over partially decoded Unicode."""
        with self.lock:
            index, _ = first_stop(text, self.stops)
            if index is not None:
                text = text[:index]
                self.finish_reason = "stop"
            elif self.finish_reason is None:
                self.finish_reason = "length" if max_tokens and self.completion_tokens >= max_tokens else "stop"
            self.content = text
            self.reasoning = reasoning
            self.done = True
            self.generation_completed = True

    def fail(self, reason):
        with self.lock:
            self.error = str(reason)
            self.done = True

    def snapshot(self):
        with self.lock:
            return {
                "content": self.content, "reasoning": self.reasoning,
                "completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens,
                "done": self.done, "finish_reason": self.finish_reason, "error": self.error,
                "generation_completed": self.generation_completed,
            }

def register(request_id, thinking=False, stops=()):
    with _REGISTRY_LOCK:
        if request_id in REGISTRY:
            raise ValueError("Duplicate native API request")
        session = CaptureSession(request_id, thinking=thinking, stops=stops)
        REGISTRY[request_id] = session
        return session

def lookup(request_id):
    with _REGISTRY_LOCK:
        return REGISTRY.get(request_id)

def release(request_id):
    with _REGISTRY_LOCK:
        REGISTRY.pop(request_id, None)

def _prompt_token_count(tokens):
    """Native tokenized text length. None if there is nontext/multimodal input."""
    if isinstance(tokens, dict):
        tokens = next(iter(tokens.values()), [])
    if not isinstance(tokens, list):
        return None
    count = 0
    for batch in tokens:
        if not isinstance(batch, list):
            return None
        for token in batch:
            if not isinstance(token, tuple) or not isinstance(token[0], int):
                return None
            count += 1
    return count

def _accepted_ids_and_prompt(frame):
    """Read only known native progress call sites. Never read speculative drafts."""
    ids = None
    prompt_tokens = None
    depth = 0
    while frame is not None and depth < 24:
        name = frame.f_code.co_name
        module = frame.f_globals.get("__name__", "")
        loc = frame.f_locals
        if name == "generate" and module == "comfy.text_encoders.llama" and "generated_token_ids" in loc:
            ids = loc["generated_token_ids"]
        elif name == "_generate_mtp" and module == "comfy.text_encoders.qwen35" and "ids" in loc:
            ids = loc["ids"]
        elif name == "execute" and module == "comfy_extras.nodes_textgen" and "tokens" in loc:
            prompt_tokens = _prompt_token_count(loc["tokens"])
        frame = frame.f_back
        depth += 1
    if not isinstance(ids, list) or not all(type(x) is int for x in ids):
        return None, prompt_tokens
    return ids, prompt_tokens

def install_progress_tap():
    global _INSTALLED
    if _INSTALLED:
        return
    import comfy.utils
    original = comfy.utils.ProgressBar.update

    def update(self, value):
        # Preserve native progress and existing ComfyUI progress hooks.
        result = original(self, value)
        active = getattr(_THREAD, "capture", None)
        if active is None:
            return result
        session, decoder = active
        ids, prompt_tokens = _accepted_ids_and_prompt(sys._getframe(1))
        if ids is not None:
            session.feed(ids, decoder, prompt_tokens)
        return result

    comfy.utils.ProgressBar.update = update
    _INSTALLED = True

class native_capture:
    """Bind one session to the ComfyUI worker thread for a single native generation."""
    def __init__(self, request_id, clip):
        self.request_id = request_id
        self.clip = clip

    def __enter__(self):
        self.previous = getattr(_THREAD, "capture", None)
        session = lookup(self.request_id)
        if session is not None:
            install_progress_tap()
            _THREAD.capture = (session, self.clip.decode)
        return session

    def __exit__(self, *_):
        _THREAD.capture = self.previous
