"""Persistent settings and API-key management, independent of ComfyUI internals."""
import json
import os
import secrets
import tempfile
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("COMFY_NATIVE_LLM_API_CONFIG", str(BASE_DIR / "data" / "settings.json")))
LOCK = threading.RLock()
MTP = {"auto", "off", "2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"}

DEFAULTS = {
    "model": "",
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
    "presence_penalty": 0.0,
    "max_tokens": 512,
    "thinking": False,
    "mtp": "auto",
    "use_default_template": True,
    "sampling": True,
    "max_pending": 16,
    "timeout_seconds": 600,
}

def read():
    with LOCK:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not CONFIG_PATH.exists():
            _write({**DEFAULTS, "api_key": secrets.token_urlsafe(32)})
        try:
            obj = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            raise RuntimeError("Native LLM settings are corrupt; repair the settings file manually") from ex
        return {**DEFAULTS, **obj}

def _write(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=CONFIG_PATH.parent, prefix=".native-llm-", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(name, 0o600)
        os.replace(name, CONFIG_PATH)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def update(changes):
    with LOCK:
        current = read()
        allowed = set(DEFAULTS)
        if not isinstance(changes, dict) or any(k not in allowed and k != "rotate_api_key" for k in changes):
            raise ValueError("Unknown settings field")
        if "model" in changes and (not isinstance(changes["model"], str) or len(changes["model"]) > 512):
            raise ValueError("Invalid model")
        for k, lo, hi in (
            ("temperature", 0.01, 2.0), ("top_p", 0.0, 1.0), ("top_k", 0, 1000),
            ("min_p", 0.0, 1.0), ("repetition_penalty", 0.0, 5.0),
            ("presence_penalty", 0.0, 5.0), ("max_tokens", 1, 32768),
            ("max_pending", 1, 128), ("timeout_seconds", 15, 3600),
        ):
            if k in changes:
                v = changes[k]
                if isinstance(v, bool) or not isinstance(v, (float, int)) or not lo <= v <= hi:
                    raise ValueError(f"Invalid {k}; expected number in [{lo}, {hi}]")
                if k in {"top_k", "max_tokens", "max_pending", "timeout_seconds"} and int(v) != v:
                    raise ValueError(f"{k} must be an integer")
        for k in ("thinking", "use_default_template", "sampling"):
            if k in changes and not isinstance(changes[k], bool):
                raise ValueError(f"{k} must be boolean")
        if "mtp" in changes and changes["mtp"] not in MTP:
            raise ValueError("Invalid MTP mode")
        if changes.get("rotate_api_key") is True:
            current["api_key"] = secrets.token_urlsafe(32)
        current.update({k: v for k, v in changes.items() if k in allowed})
        _write(current)
        return current
