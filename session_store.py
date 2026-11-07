"""Session + op persistence for LLM Tinkerer.

`sessions/<name>.json`     — model name, safetensors flag, gen params,
                             config fingerprint
`sessions/<name>.ops.jsonl` — ordered serializable op specs (one per line)

Ops are replayed in order onto a freshly loaded model; tensor backups are
re-derived at apply time, never serialized.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

SESSIONS_DIR = "sessions"


def _path(name: str, suffix: str = ".json") -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    return os.path.join(SESSIONS_DIR, f"{safe}{suffix}")


def config_fingerprint(config) -> Dict[str, Any]:
    """Small identity signature for a loaded model's config."""
    keys = (
        "architectures", "hidden_size", "n_embd", "num_hidden_layers",
        "n_layer", "intermediate_size", "n_inner", "vocab_size",
        "num_attention_heads", "n_head",
    )
    fp = {}
    for k in keys:
        v = getattr(config, k, None)
        if v is not None:
            fp[k] = v
    return fp


def fingerprint_matches(fp_a: Dict[str, Any], fp_b: Dict[str, Any]) -> List[str]:
    """Return list of mismatched keys (empty = match)."""
    return [k for k in fp_a if k in fp_b and fp_a[k] != fp_b[k]]


def save_session(name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    payload = dict(payload)
    payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = _path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_session(name: str) -> Optional[Dict[str, Any]]:
    path = _path(name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ops(name: str, specs: List[Dict[str, Any]]) -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path = _path(name, ".ops.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for spec in specs:
            f.write(json.dumps(spec) + "\n")
    return path


def load_ops(name: str) -> List[Dict[str, Any]]:
    path = _path(name, ".ops.jsonl")
    if not os.path.exists(path):
        return []
    specs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    specs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return specs


def list_sessions() -> List[str]:
    if not os.path.isdir(SESSIONS_DIR):
        return []
    names = [
        f[:-5] for f in os.listdir(SESSIONS_DIR)
        if f.endswith(".json") and not f.endswith(".ops.jsonl")
    ]
    return sorted(
        names,
        key=lambda n: os.path.getmtime(_path(n)),
        reverse=True,
    )
