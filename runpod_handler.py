"""
Remote Flask handler for LLM Tinkerer running on a RunPod GPU pod.

Exposes a small HTTP API that mirrors the local ModelManager operations:
- /health
- /load_model
- /generate
- /invert_weights
- /restore_weights
- /list_layers
- /structure
- /neuron_info
- /neuron_steer_generate
- /neuron_sweep + /neuron_sweep_status + /neuron_sweep_result
"""

import logging
import os
import threading
from typing import Dict, Any, List

from flask import Flask, jsonify, request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Configure logging to stdout so RunPod console captures it
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

_lock = threading.Lock()
_model = None
_tokenizer = None
_config = None
_model_name = None
_inverter = None
_explorer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

_sweep_job = {
    "state": "idle",      # idle | running | done | error
    "done": 0,
    "total": 0,
    "msg": "",
    "stats": None,
    "ranking": None,
    "error": None,
    "run_name": None,
    "run_path": None,
}


def _get_inverter():
    global _inverter
    if _inverter is None and _model is not None:
        from weight_inversion_implementation import WeightInverter
        _inverter = WeightInverter(_model)
    return _inverter


def _get_explorer():
    global _explorer
    if _model is not None:
        if _explorer is None or _explorer.model is not _model or _explorer.tokenizer is not _tokenizer:
            from neuron_explorer import NeuronExplorer
            _explorer = NeuronExplorer(_model, _tokenizer)
    return _explorer


def _get_model_state():
    return _model, _tokenizer, _config, _model_name


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint polled by the local manager."""
    return jsonify({
        "status": "healthy",
        "device": _device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": _model is not None,
    })


@app.route("/structure", methods=["GET"])
def structure():
    if _config is None:
        return jsonify({"error": "No model loaded"}), 400
    arch = getattr(_config, "architectures", ["Unknown"])
    return jsonify({
        "model": _model_name,
        "architecture": arch[0] if arch else "Unknown",
        "hidden_size": getattr(_config, "hidden_size", None),
        "num_hidden_layers": getattr(_config, "num_hidden_layers", None),
        "num_attention_heads": getattr(_config, "num_attention_heads", None),
        "intermediate_size": getattr(_config, "intermediate_size", None),
        "vocab_size": getattr(_config, "vocab_size", None),
        "device": _device,
    })


@app.route("/list_layers", methods=["GET"])
def list_layers():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    layers = []
    for name, module in _model.named_modules():
        if name == "":
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.MultiheadAttention)):
            layers.append(name)
    return jsonify({"layers": layers})


@app.route("/load_model", methods=["POST"])
def load_model():
    global _model, _tokenizer, _config, _model_name, _inverter, _explorer
    data = request.get_json(force=True)
    model_name = data.get("model_name", "gpt2")
    use_safetensors = data.get("use_safetensors", True)

    if model_name.lower().endswith(".gguf") or "gguf" in model_name.lower():
        return jsonify({"error": "GGUF models are not supported"}), 400

    with _lock:
        _model = None
        _tokenizer = None
        _config = None
        _model_name = None
        _inverter = None
        _explorer = None
        try:
            logger.info(f"Loading model {model_name} on device {_device}")
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
                device_map="auto",
                use_safetensors=use_safetensors,
            )
            _config = config
            _tokenizer = tokenizer
            _model = model
            _model_name = model_name
            return jsonify({
                "status": "success",
                "model": model_name,
                "device": _device,
                "cuda_available": torch.cuda.is_available(),
            })
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        return jsonify({"error": "No model loaded"}), 400

    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    max_length = int(data.get("max_length", 100))
    temperature = float(data.get("temperature", 0.7))

    try:
        with _lock:
            inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
            outputs = _model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=_tokenizer.eos_token_id,
            )
            text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({
            "status": "success",
            "output": text,
            "model": _model_name,
            "parameters": {"max_length": max_length, "temperature": temperature},
        })
    except Exception as e:
        logger.error(f"Error generating text: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/invert_weights", methods=["POST"])
def invert_weights():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400

    data = request.get_json(force=True)
    inversion_type = data.get("inversion_type", "output_only")
    multiplier = float(data.get("multiplier", -1.0))
    layer_start = int(data.get("layer_start", 0))
    layer_end = int(data.get("layer_end", 999))
    custom_rules = data.get("custom_rules", "")

    try:
        with _lock:
            inverter = _get_inverter()
            if inverter is None:
                return jsonify({"error": "No model loaded"}), 400
            inverter.backup_weights()
            config = {
                "type": inversion_type.lower().replace(" ", "_"),
                "multiplier": multiplier,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "custom_rules": [r.strip() for r in custom_rules.split(",") if r.strip()],
            }
            result = inverter.invert_weights(config)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        logger.error(f"Error inverting weights: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/restore_weights", methods=["POST"])
def restore_weights():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    try:
        with _lock:
            inverter = _get_inverter()
            if inverter is None:
                return jsonify({"error": "No inverter available"}), 400
            inverter.restore_weights()
        return jsonify({"status": "success", "message": "Weights restored"})
    except Exception as e:
        logger.error(f"Error restoring weights: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# --- Neuron explorer endpoints ---
# Mirrors neuron_explorer.py locally: steer one MLP neuron at generation time,
# or sweep many neurons x doses x prompts and keep every rollout.


@app.route("/neuron_info", methods=["GET"])
def neuron_info():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    try:
        ex = _get_explorer()
        return jsonify({
            "model": _model_name,
            "num_layers": ex.num_layers(),
            "intermediate_sizes": {str(k): v for k, v in ex.layer_info().items()},
            "hidden_size": ex.hidden_size(),
        })
    except Exception as e:
        logger.error(f"neuron_info failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/neuron_steer_generate", methods=["POST"])
def neuron_steer_generate():
    if _model is None or _tokenizer is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    layer = int(data.get("layer", 0))
    neuron = data.get("neuron", None)
    dose = float(data.get("dose", 0.0))
    dose_mode = data.get("dose_mode", "rel")
    max_new_tokens = int(data.get("max_new_tokens", 40))
    temperature = float(data.get("temperature", 0.7))
    seed = data.get("seed", None)

    try:
        with _lock:
            ex = _get_explorer()
            kwargs = {}
            if neuron is not None:
                import neuron_explorer as nx
                _, mod = ex.find_down_proj(layer)
                col = nx.neuron_write_column(mod, int(neuron)).detach().float()
                col = col / (col.norm() + 1e-8)
                med = ex.median_residual_norm(layer, [prompt]) if dose_mode == "rel" else 1.0
                kwargs["vector"] = (col * dose * med).to(device=_model.device, dtype=col.dtype)
            out = ex.steered_generate(
                prompt, layer, max_new_tokens=max_new_tokens,
                temperature=temperature, seed=seed, **kwargs,
            )
        return jsonify({"status": "success", **out})
    except Exception as e:
        logger.error(f"neuron_steer_generate failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_sweep_job(cfg: Dict[str, Any]) -> None:
    global _sweep_job
    try:
        import random
        import neuron_explorer as nx_mod
        ex = _get_explorer()
        layer = int(cfg["layer"])
        intermediate = ex.intermediate_size(layer)
        neurons = nx_mod.NeuronExplorer.parse_neuron_spec(
            cfg.get("neuron_spec", "all"), intermediate, random.Random(int(cfg.get("seed", 0)))
        )
        prompts = [p for p in cfg.get("prompts", []) if str(p).strip()]
        doses = [float(d) for d in cfg.get("doses", [0.8])]

        def prog(done, total, msg):
            _sweep_job.update({"done": done, "total": total, "msg": msg})

        run_dir = "/workspace/neuron_runs"
        path, stats = ex.run_sweep(
            layer=layer,
            neurons=neurons,
            prompts=prompts,
            doses=doses,
            n_controls=int(cfg.get("n_controls", 0)),
            max_new_tokens=int(cfg.get("max_new_tokens", 40)),
            temperature=float(cfg.get("temperature", 0.7)),
            dose_mode=cfg.get("dose_mode", "rel"),
            run_dir=run_dir,
            run_name=cfg.get("run_name") or None,
            seed=int(cfg.get("seed", 0)),
            progress=prog,
        )
        ranking = nx_mod.NeuronExplorer.rank_run(path)
        _sweep_job.update({
            "state": "done",
            "stats": stats,
            "ranking": ranking,
            "run_name": stats["run"].split("/")[-1].split("\\")[-1],
            "run_path": path,
            "msg": "done",
        })
    except Exception as e:
        logger.error(f"neuron sweep failed: {e}", exc_info=True)
        _sweep_job.update({"state": "error", "error": str(e)})


@app.route("/neuron_sweep", methods=["POST"])
def neuron_sweep():
    if _model is None or _tokenizer is None:
        return jsonify({"error": "No model loaded"}), 400
    if _sweep_job["state"] == "running":
        return jsonify({"error": "A sweep is already running"}), 409
    cfg = request.get_json(force=True) or {}
    _sweep_job.update({
        "state": "running", "done": 0, "total": 0, "msg": "starting",
        "stats": None, "ranking": None, "error": None,
        "run_name": None, "run_path": None,
    })
    threading.Thread(target=_run_sweep_job, args=(cfg,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/neuron_sweep_status", methods=["GET"])
def neuron_sweep_status():
    job = {k: v for k, v in _sweep_job.items() if k != "run_path"}
    return jsonify(job)


@app.route("/neuron_sweep_result", methods=["GET"])
def neuron_sweep_result():
    path = _sweep_job.get("run_path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "No sweep result available"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return app.response_class(f.read(), mimetype="application/jsonl")


def run_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_server()
