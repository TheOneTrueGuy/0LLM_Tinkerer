"""
Weight-Space Programming agent — an LLM driving tinker ops through
OpenRouter (OpenAI-compatible tool calling).

The agent never touches tensors directly: it emits tool calls that map to
the existing op surface (inspect, stats, generate, sculpt, undo), so every
action lands in the same op log a human would produce. Secrets follow the
RunPod pattern: env var -> .env -> OR_key.txt / venv/OR_key.txt.
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import requests
import torch

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-32b"


def load_or_key() -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    for path in (".env", "OR_key.txt", os.path.join("venv", "OR_key.txt")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY="):
                        return line.split("=", 1)[1].strip()
                    if line and not line.startswith("#"):
                        return line
        except OSError:
            continue
    return None


def chat(model: str, messages: List[Dict[str, Any]],
         tools: Optional[List[Dict]] = None,
         temperature: float = 0.4) -> Dict[str, Any]:
    """One OpenRouter chat completion. Returns the raw response JSON."""
    key = load_or_key()
    if not key:
        raise RuntimeError(
            "No OpenRouter key (set OPENROUTER_API_KEY or write OR_key.txt)")
    body: Dict[str, Any] = {
        "model": model, "messages": messages, "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json=body, timeout=120,
    )
    r.raise_for_status()
    return r.json()


# ---------- tool surface ----------

def make_tools(model_manager) -> List[Dict[str, Any]]:
    """OpenAI function-calling schemas for the tinker op surface."""
    def fn(name, desc, props=None, required=None):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object",
                           "properties": props or {},
                           "required": required or []}}}

    return [
        fn("get_model_info",
           "Name, parameter count and architecture of the loaded model"),
        fn("list_layers",
           "List named weight parameters, optionally filtered by substring",
           {"filter": {"type": "string", "description":
                       "substring filter, e.g. 'mlp.c_proj' or 'h.5'"}}),
        fn("weight_stats",
           "Shape, mean, std, abs-max and L2 norm of a named parameter",
           {"name": {"type": "string", "description": "exact parameter name"}},
           ["name"]),
        fn("generate",
           "Generate text with the CURRENT weights (use to evaluate edits)",
           {"prompt": {"type": "string"},
            "max_new_tokens": {"type": "integer", "default": 40},
            "temperature": {"type": "number", "default": 0.7}},
           ["prompt"]),
        fn("sculpt_op",
           "Apply a persistent neuron edit. scale_write multiplies the "
           "neuron's output column by factor; ablate zeros it; scale_read "
           "scales the upstream row; bias_push adds dose*direction to the "
           "down-proj bias (GPT-2 only).",
           {"kind": {"type": "string",
                     "enum": ["scale_write", "ablate", "scale_read",
                              "bias_push"]},
            "layer": {"type": "integer"},
            "neuron": {"type": "integer"},
            "value": {"type": "number",
                      "description": "scale factor (scale_*) or dose "
                                     "(bias_push)"}},
           ["kind", "layer", "neuron", "value"]),
        fn("undo_sculpt",
           "Undo the last N sculpt ops (n=0 undoes all)",
           {"n": {"type": "integer", "default": 1}}),
        fn("write_fact",
           "Closed-form fact write (MEMIT-style, no gradients): teach the "
           "model that questions in 'asks' are answered by 'answer'. Give "
           "3-6 phrasings for robustness; optional 'contrast' protects "
           "look-alike questions. Expensive: ~1-3 min on small models.",
           {"asks": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "string"},
            "contrast": {"type": "array", "items": {"type": "string"}}},
           ["asks", "answer"]),
        fn("undo_fact", "Remove the most recently written fact"),
    ]


def make_executor(model_manager) -> Callable[[str, Dict[str, Any]], str]:
    """Map tool calls onto the loaded model. Returns JSON-encoded strings."""
    def execute(name: str, args: Dict[str, Any]) -> str:
        mm = model_manager
        try:
            if mm.model is None:
                return json.dumps({"error": "no model loaded"})
            if name == "get_model_info":
                n = sum(p.numel() for p in mm.model.parameters())
                return json.dumps({
                    "model": mm.model_name, "params": n,
                    "layers": getattr(mm.model.config, "n_layer",
                                      getattr(mm.model.config,
                                              "num_hidden_layers", None)),
                    "vocab": getattr(mm.model.config, "vocab_size", None),
                    "hidden": getattr(mm.model.config, "n_embd",
                                      getattr(mm.model.config,
                                              "hidden_size", None))})
            if name == "list_layers":
                filt = args.get("filter", "")
                names = [n for n, _ in mm.model.named_parameters()
                         if not filt or filt in n]
                return json.dumps({"count": len(names),
                                   "names": names[:200]})
            if name == "weight_stats":
                p = dict(mm.model.named_parameters()).get(args["name"])
                if p is None:
                    return json.dumps({"error": f"no param {args['name']!r}; "
                                       "call list_layers first"})
                d = p.data.float()
                return json.dumps({
                    "name": args["name"], "shape": list(p.shape),
                    "mean": float(d.mean()), "std": float(d.std()),
                    "abs_max": float(d.abs().max()),
                    "l2": float(d.norm())})
            if name == "generate":
                out = mm.generate_text(
                    args["prompt"],
                    int(args.get("max_new_tokens", 40)),
                    float(args.get("temperature", 0.7)))
                return json.dumps({"output": out})
            if name == "sculpt_op":
                sc = mm.neuron_sculptor
                if sc is None:
                    from neuron_explorer import NeuronExplorer, NeuronSculptor
                    sc = mm.neuron_sculptor = NeuronSculptor(
                        NeuronExplorer(mm.model, mm.tokenizer))
                kind, layer = args["kind"], int(args["layer"])
                neuron, value = int(args["neuron"]), float(args["value"])
                if kind == "scale_write":
                    r = sc.apply_scale_write(layer, neuron, value)
                elif kind == "ablate":
                    r = sc.apply_ablate(layer, neuron)
                elif kind == "scale_read":
                    r = sc.apply_scale_read(layer, neuron, value)
                elif kind == "bias_push":
                    r = sc.apply_bias_push(layer, neuron, value)
                else:
                    return json.dumps({"error": f"unknown kind {kind}"})
                if isinstance(r, dict):
                    r.pop("backups", None)
                    r.pop("backup", None)
                return json.dumps(r, default=str)
            if name == "undo_sculpt":
                sc = mm.neuron_sculptor
                if sc is None:
                    return json.dumps({"error": "no sculpt ops yet"})
                n = int(args.get("n", 1))
                if n <= 0:
                    msg = sc.undo_all()
                else:
                    msgs = [sc.undo() for _ in range(n)]
                    msg = msgs[-1]
                return json.dumps({"result": msg})
            if name == "write_fact":
                from fact_editor import FactEditor
                if mm.fact_editor is None:
                    mm.fact_editor = FactEditor(mm.model, mm.tokenizer)
                stats = mm.fact_editor.teach(
                    args["asks"], args["answer"],
                    args.get("contrast") or [])
                stats.pop("history", None)
                return json.dumps(stats, default=str)
            if name == "undo_fact":
                if mm.fact_editor is None:
                    return json.dumps({"error": "no facts written"})
                return json.dumps({"result": mm.fact_editor.undo_last()})
            return json.dumps({"error": f"unknown tool {name}"})
        except Exception as e:
            logger.warning("tool %s failed: %s", name, e)
            return json.dumps({"error": str(e)})
    return execute


SYSTEM_PROMPT = (
    "You are a weight-space programming agent embedded in LLM Tinkerer, a "
    "workbench that has a small language model loaded and exposes tools to "
    "inspect its parameters, edit them, and generate text. Work like a "
    "researcher: inspect before editing, make small reversible changes, "
    "generate text to evaluate each edit, and undo anything that hurts the "
    "model. Explain briefly what you did and what you observed."
)


def run_agent(model_manager, or_model: str, instruction: str,
              max_steps: int = 8, temperature: float = 0.4):
    """Agent loop. Yields (log_line, done) so callers can stream progress."""
    tools = make_tools(model_manager)
    execute = make_executor(model_manager)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    for step in range(max_steps):
        resp = chat(or_model, messages, tools, temperature)
        msg = resp["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            yield f"--- agent reply ---\n{msg.get('content', '')}", True
            return
        if msg.get("content"):
            yield msg["content"], False
        for c in calls:
            fname = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            yield f"[{step + 1}] {fname}({json.dumps(args)})", False
            result = execute(fname, args)
            yield f"    -> {result[:300]}", False
            messages.append({
                "role": "tool", "tool_call_id": c["id"],
                "content": result[:8000],
            })
    yield "(step limit reached)", True
