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

from flask import Flask, jsonify, request, send_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    _HAS_VLM = True
except Exception:
    _HAS_VLM = False

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
_fact_editor = None
_eval_prober = None
_merger = None
_sculptor = None
_processor = None
_vlm_editor = None
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


def _get_fact_editor():
    global _fact_editor
    if _model is not None and (
            _fact_editor is None or _fact_editor.model is not _model):
        from fact_editor import FactEditor
        _fact_editor = FactEditor(_model, _tokenizer)
    return _fact_editor


def _get_eval_prober():
    global _eval_prober
    if _model is not None and (
            _eval_prober is None or _eval_prober.model is not _model):
        from eval_probe import EvalProber
        _eval_prober = EvalProber(_model, _tokenizer)
    return _eval_prober


def _get_sculptor():
    global _sculptor
    if _model is not None and (
            _sculptor is None or _sculptor.ex.model is not _model):
        from neuron_explorer import NeuronSculptor
        _sculptor = NeuronSculptor(_get_explorer())
    return _sculptor


def _get_vlm_editor():
    global _vlm_editor
    if _model is not None and _processor is not None and (
            _vlm_editor is None or _vlm_editor.model is not _model):
        from vlm_fact_editor import VLMFactEditor
        _vlm_editor = VLMFactEditor(_model, _tokenizer, _processor)
    return _vlm_editor


def _get_merger():
    global _merger
    if _model is not None and (
            _merger is None or _merger.model is not _model):
        from weight_space import WeightMerger
        _merger = WeightMerger(_model)
    return _merger


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
    n_params = (
        sum(p.numel() for p in _model.parameters()) if _model is not None else None
    )
    return jsonify({
        "model": _model_name,
        "params": n_params,
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
    global _fact_editor, _eval_prober, _merger, _sculptor
    global _processor, _vlm_editor
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
        _fact_editor = None
        _eval_prober = None
        _merger = None
        _sculptor = None
        _processor = None
        _vlm_editor = None
        try:
            logger.info(f"Loading model {model_name} on device {_device}")
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
                    device_map="auto",
                    use_safetensors=use_safetensors,
                )
            except Exception:
                if not _HAS_VLM:
                    raise
                logger.info("CausalLM load failed; trying ImageTextToText")
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
                    device_map="auto",
                )
                _processor = AutoProcessor.from_pretrained(model_name)
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


# ---------------------------------------------------------------------------
# Weight-space / eval / fact-write endpoints (mirror weight_space.py,
# eval_probe.py, fact_editor.py locally)
# ---------------------------------------------------------------------------


@app.route("/eval", methods=["POST"])
def eval_probe():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json(force=True) or {}
    try:
        with _lock:
            r = _get_eval_prober().evaluate(data.get("label", ""))
            return jsonify(r)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fact_write", methods=["POST"])
def fact_write():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json(force=True)
    asks = data.get("asks") or []
    answer = (data.get("answer") or "").strip()
    if not asks or not answer:
        return jsonify({"error": "need 'asks' (list) and 'answer'"}), 400
    try:
        with _lock:
            stats = _get_fact_editor().teach(
                asks, answer, data.get("contrast") or [])
            return jsonify(stats)
    except Exception as e:
        logger.error(f"fact_write failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/fact_compare", methods=["POST"])
def fact_compare():
    if _fact_editor is None:
        return jsonify({"error": "No facts written yet"}), 400
    data = request.get_json(force=True) or {}
    try:
        r = _fact_editor.compare(
            data.get("q") or "Tell me a fact.", int(data.get("n", 60)))
        return jsonify(r)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fact_undo", methods=["POST"])
def fact_undo():
    if _fact_editor is None:
        return jsonify({"error": "No facts written yet"}), 400
    return jsonify({"result": _fact_editor.undo_last(),
                    "facts": _fact_editor.describe()})


@app.route("/fact_commit", methods=["POST"])
def fact_commit():
    if _fact_editor is None:
        return jsonify({"error": "No facts written yet"}), 400
    return jsonify({"result": _fact_editor.commit(),
                    "facts": _fact_editor.describe()})


@app.route("/ws_check", methods=["POST"])
def ws_check():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "need 'name'"}), 400
    try:
        return jsonify({"report": _get_merger().check_compatibility(name)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ws_load_ref", methods=["POST"])
def ws_load_ref():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "need 'name'"}), 400
    try:
        return jsonify({"result": _get_merger().load_reference(name)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ws_merge", methods=["POST"])
def ws_merge():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json(force=True) or {}
    try:
        with _lock:
            r = _get_merger().merge(
                data.get("mode", "task_add"),
                float(data.get("lam", 1.0)),
                (data.get("filter") or "").strip())
            if "error" in r:
                return jsonify(r), 400
            return jsonify(r)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ws_undo", methods=["POST"])
def ws_undo():
    m = _get_merger()
    if m is None:
        return jsonify({"error": "No model loaded"}), 400
    return jsonify({"result": m.undo(), "history": m.describe()})


@app.route("/weight_stats", methods=["POST"])
def weight_stats():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    p = dict(_model.named_parameters()).get(name)
    if p is None:
        return jsonify({"error": f"no param {name!r}; call /list_layers first"}), 400
    d = p.data.float()
    return jsonify({
        "name": name, "shape": list(p.shape),
        "mean": float(d.mean()), "std": float(d.std()),
        "abs_max": float(d.abs().max()), "l2": float(d.norm()),
    })


@app.route("/sculpt_op", methods=["POST"])
def sculpt_op():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json(force=True)
    kind, layer = data.get("kind", ""), int(data.get("layer", 0))
    neuron, value = int(data.get("neuron", 0)), float(data.get("value", 0))
    try:
        sc = _get_sculptor()
        if kind == "scale_write":
            r = sc.apply_scale_write(layer, neuron, value)
        elif kind == "ablate":
            r = sc.apply_ablate(layer, neuron)
        elif kind == "scale_read":
            r = sc.apply_scale_read(layer, neuron, value)
        elif kind == "bias_push":
            r = sc.apply_bias_push(layer, neuron, value)
        else:
            return jsonify({"error": f"unknown kind {kind}"}), 400
        if isinstance(r, dict):
            r.pop("backups", None)
        return jsonify(r, default=str)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/undo_sculpt", methods=["POST"])
def undo_sculpt():
    if _sculptor is None:
        return jsonify({"error": "no sculpt ops yet"}), 400
    data = request.get_json(force=True) or {}
    n = int(data.get("n", 1))
    if n <= 0:
        msg = _sculptor.undo_all()
    else:
        msgs = [_sculptor.undo() for _ in range(n)]
        msg = msgs[-1]
    return jsonify({"result": msg})


# ---------------------------------------------------------------------------
# VLM fact writing + model save/download
# ---------------------------------------------------------------------------


def _decode_images(items):
    """items: list of {"image": <b64 png> or null, "q": str} -> [(PIL|None, q)]"""
    import base64
    import io
    from PIL import Image
    out = []
    for it in items or []:
        img = None
        b64 = it.get("image")
        if b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        out.append((img, it.get("q", "")))
    return out


@app.route("/vlm_fact_write", methods=["POST"])
def vlm_fact_write():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    if _get_vlm_editor() is None:
        return jsonify({"error": "loaded model is not a VLM "
                        "(no processor)"}), 400
    data = request.get_json(force=True)
    asks = _decode_images(data.get("asks"))
    contrast = _decode_images(data.get("contrast"))
    answer = (data.get("answer") or "").strip()
    if not asks or not answer:
        return jsonify({"error": "need asks[{image,q}] + answer"}), 400
    try:
        with _lock:
            stats = _vlm_editor.teach(asks, answer, contrast)
            return jsonify(stats)
    except Exception as e:
        logger.error(f"vlm_fact_write failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/vlm_fact_compare", methods=["POST"])
def vlm_fact_compare():
    if _vlm_editor is None:
        return jsonify({"error": "No VLM facts written yet"}), 400
    data = request.get_json(force=True) or {}
    pairs = _decode_images([data])
    if not pairs:
        return jsonify({"error": "need image + q"}), 400
    try:
        r = _vlm_editor.compare(pairs[0], n=int(data.get("n", 60)))
        return jsonify(r)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/vlm_fact_undo", methods=["POST"])
def vlm_fact_undo():
    if _vlm_editor is None:
        return jsonify({"error": "No VLM facts written yet"}), 400
    return jsonify({"result": _vlm_editor.undo_last(),
                    "facts": _vlm_editor.describe()})


@app.route("/save_model", methods=["POST"])
def save_model():
    """Serialize the live (edited) model to /workspace/edited_model."""
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    try:
        out = "/workspace/edited_model"
        with _lock:
            _model.save_pretrained(out)
            if _tokenizer is not None:
                _tokenizer.save_pretrained(out)
        return jsonify({"saved": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_model", methods=["GET"])
def download_model():
    """Zip /workspace/edited_model and stream it back."""
    import shutil
    import tempfile
    src = "/workspace/edited_model"
    if not os.path.isdir(src):
        return jsonify({"error": "no saved model; call /save_model first"}), 404
    tmp = os.path.join(tempfile.gettempdir(), "edited_model")
    zpath = shutil.make_archive(tmp, "zip", src)
    return send_file(zpath, as_attachment=True,
                     download_name="edited_model.zip",
                     mimetype="application/zip")


def run_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_server()
