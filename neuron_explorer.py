"""
Neuron steering + sweep exploration for loaded causal LMs.

Implements the steering-first neuron exploration method from
https://github.com/lyramakesmusic/neuron-explorer, model-agnostically:

- Every MLP neuron i writes through one column of the down-projection.
  Pushing neuron i by dose d adds ``d * down_col_i`` to the MLP output
  (equivalently, to the residual stream) at every position.
- A sweep pushes many neurons on a few prompts and keeps every rollout,
  including unsteered baselines and random-direction controls.
- Ranking scores each neuron by how much its steered vocabulary shifts
  relative to the controls.

Model-agnostic: the decoder block stack and each block's down-projection
are discovered by shape (intermediate -> hidden), so this works for
GPT-2 (Conv1D ``mlp.c_proj``), Llama (``mlp.down_proj``), OPT (``fc2``),
Bloom (``dense_4h_to_h``), etc.

Runs live on the already-loaded model, so steered rollouts compose with
weight inversion, edits, and layer surgery applied elsewhere in the app.
"""

import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

RUNS_DIR = "neuron_runs"

# config attribute names that store decoder depth, by arch family
DEPTH_CONFIG_ATTRS = (
    "num_hidden_layers",
    "n_layer",
    "num_layers",
    "n_layers",
)

# config attribute names that store hidden size
HIDDEN_CONFIG_ATTRS = ("hidden_size", "n_embd", "d_model", "n_embed")

_WORD_RE = re.compile(r"[a-z0-9']+")


def _param_io_dims(module: nn.Module) -> Optional[Tuple[int, int]]:
    """Return (in_features, out_features) for a Linear/Conv1D-like module."""
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return int(module.in_features), int(module.out_features)
    # transformers Conv1D: nx = in, nf = out
    if hasattr(module, "nx") and hasattr(module, "nf"):
        return int(module.nx), int(module.nf)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.dim() == 2:
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor) and bias.dim() == 1:
            out_dim = int(bias.shape[0])
            in_dim = int(weight.shape[0] if weight.shape[1] == out_dim else weight.shape[1])
            return in_dim, out_dim
    return None


def neuron_write_column(module: nn.Module, index: int) -> torch.Tensor:
    """The residual-stream vector neuron ``index`` writes through.

    nn.Linear weight is [out, in] -> column ``index``.
    transformers Conv1D weight is [in, out] -> row ``index``.
    """
    dims = _param_io_dims(module)
    weight = module.weight
    if dims is not None:
        in_dim, out_dim = dims
        if weight.shape[0] == out_dim and weight.shape[1] == in_dim:
            return weight[:, index]
        if weight.shape[0] == in_dim and weight.shape[1] == out_dim:
            return weight[index, :]
    # fallback: guess the hidden-sized axis
    return weight[:, index] if weight.shape[0] < weight.shape[1] else weight[index, :]


def _neuron_write_index(module: nn.Module, index: int) -> Tuple:
    """Weight-tensor index selecting a down-proj neuron's write column."""
    dims = _param_io_dims(module)
    weight = module.weight
    if dims is not None:
        in_dim, out_dim = dims
        if weight.shape[0] == out_dim and weight.shape[1] == in_dim:
            return (slice(None), index)
        if weight.shape[0] == in_dim and weight.shape[1] == out_dim:
            return (index, slice(None))
    return (slice(None), index) if weight.shape[0] < weight.shape[1] else (index, slice(None))


def _neuron_read_index(module: nn.Module, index: int) -> Tuple:
    """Weight-tensor index selecting an up-proj neuron's read row."""
    dims = _param_io_dims(module)
    weight = module.weight
    if dims is not None:
        in_dim, out_dim = dims
        if weight.shape[0] == out_dim and weight.shape[1] == in_dim:
            return (index, slice(None))
        if weight.shape[0] == in_dim and weight.shape[1] == out_dim:
            return (slice(None), index)
    return (index, slice(None)) if weight.shape[0] < weight.shape[1] else (slice(None), index)


class NeuronExplorer:
    """Steering, sweeping, and activation capture for one loaded model."""

    def __init__(self, model: nn.Module, tokenizer=None, block_stack: Optional[nn.ModuleList] = None):
        self.model = model
        self.tokenizer = tokenizer
        # a caller that already resolved the stack (e.g. LayerSurgeon, which
        # caches it before making it non-uniform) can pass it in directly
        self._stack_cache: Optional[Tuple[str, nn.ModuleList]] = (
            ("(provided)", block_stack) if block_stack is not None else None
        )
        self._down_cache: Dict[int, Tuple[str, nn.Module]] = {}
        self._resid_norm_cache: Dict[Tuple[int, Tuple[str, ...]], float] = {}

    # ---------- discovery ----------

    def _expected_depth(self) -> Optional[int]:
        cfg = getattr(self.model, "config", None)
        for attr in DEPTH_CONFIG_ATTRS:
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return None

    def hidden_size(self) -> Optional[int]:
        cfg = getattr(self.model, "config", None)
        for attr in HIDDEN_CONFIG_ATTRS:
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return None

    def find_block_stack(self) -> Tuple[str, nn.ModuleList]:
        """Locate the repeating decoder block ModuleList."""
        if self._stack_cache is not None:
            return self._stack_cache
        expected = self._expected_depth()
        best = None
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.ModuleList) or len(module) < 2:
                continue
            if len({type(m) for m in module}) != 1:
                continue
            score = (1 if expected and len(module) == expected else 0, len(module))
            if best is None or score > best[0]:
                best = (score, name, module)
        if best is None:
            raise RuntimeError("Could not locate a repeating decoder block ModuleList.")
        self._stack_cache = (best[1], best[2])
        return self._stack_cache

    def num_layers(self) -> int:
        return len(self.find_block_stack()[1])

    def find_down_proj(self, layer: int) -> Tuple[str, nn.Module]:
        """Find the MLP down-projection (intermediate -> hidden) in a block."""
        if layer in self._down_cache:
            return self._down_cache[layer]
        _, stack = self.find_block_stack()
        if not 0 <= layer < len(stack):
            raise IndexError(f"Layer {layer} out of range (0..{len(stack)-1})")
        block = stack[layer]
        hidden = self.hidden_size()
        candidates = []
        for name, module in block.named_modules():
            dims = _param_io_dims(module)
            if dims is None:
                continue
            in_dim, out_dim = dims
            if in_dim <= out_dim:
                continue
            if hidden is not None and out_dim != hidden:
                continue
            mlp_hint = any(
                frag in name.lower()
                for frag in ("mlp", "fc", "ffn", "feed_forward", "dense_4h_to_h", "down_proj", "linear2")
            )
            candidates.append((0 if mlp_hint else 1, name, module))
        if not candidates:
            raise RuntimeError(f"No down-projection-like module found in block {layer}.")
        candidates.sort(key=lambda c: (c[0], c[1]))
        _, name, module = candidates[0]
        self._down_cache[layer] = (name, module)
        return name, module

    def intermediate_size(self, layer: int = 0) -> int:
        _, module = self.find_down_proj(layer)
        dims = _param_io_dims(module)
        return dims[0]

    def layer_info(self) -> Dict[int, int]:
        """{layer_index: intermediate_size} for every block."""
        _, stack = self.find_block_stack()
        return {i: self.intermediate_size(i) for i in range(len(stack))}

    # ---------- hooks ----------

    @contextmanager
    def steering(
        self,
        layer: int,
        add: Optional[torch.Tensor] = None,
        neuron_doses: Optional[Dict[int, float]] = None,
        neuron_scales: Optional[Dict[int, float]] = None,
        neuron_sets: Optional[Dict[int, float]] = None,
    ):
        """Context manager: steer one layer's MLP during a forward pass.

        add:          raw hidden-sized vector added to the MLP output
        neuron_doses: {i: d} adds d * down_col_i to the MLP output
        neuron_scales:{i: s} multiplies neuron's pre-down-proj activation by s
        neuron_sets:  {i: v} sets the pre-down-proj activation to v
        """
        _, module = self.find_down_proj(layer)
        handles = []

        delta = None
        if add is not None:
            delta = add.detach().clone()
        if neuron_doses:
            weight = module.weight
            for idx, dose in neuron_doses.items():
                col = neuron_write_column(module, int(idx)).detach() * float(dose)
                delta = col if delta is None else delta + col

        if delta is not None:
            def add_hook(mod, args, out, _delta=delta):
                if isinstance(out, torch.Tensor):
                    return out + _delta.to(out.dtype)
                return out
            handles.append(module.register_forward_hook(add_hook))

        if neuron_scales or neuron_sets:
            def pre_hook(mod, args, _scales=neuron_scales or {}, _sets=neuron_sets or {}):
                x = args[0]
                if not isinstance(x, torch.Tensor):
                    return None
                x = x.clone()
                for idx, s in _scales.items():
                    x[..., int(idx)] = x[..., int(idx)] * float(s)
                for idx, v in _sets.items():
                    x[..., int(idx)] = float(v)
                return (x,) + tuple(args[1:])
            handles.append(module.register_forward_pre_hook(pre_hook))

        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ---------- residual norm calibration (rel doses) ----------

    def median_residual_norm(self, layer: int, prompts: Sequence[str]) -> float:
        """Median ||residual stream|| at this layer's block output over the prompts."""
        key = (layer, tuple(prompts))
        if key in self._resid_norm_cache:
            return self._resid_norm_cache[key]
        if not self.tokenizer:
            raise RuntimeError("Tokenizer required for residual-norm calibration.")
        _, stack = self.find_block_stack()
        block = stack[layer]
        norms: List[float] = []

        def capture(mod, args, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                norms.extend(t.float().norm(dim=-1).flatten().tolist())

        handle = block.register_forward_hook(capture)
        try:
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    self.model(**inputs)
        finally:
            handle.remove()
        if not norms:
            return 1.0
        norms.sort()
        med = norms[len(norms) // 2]
        self._resid_norm_cache[key] = med
        return med

    # ---------- generation ----------

    def steered_generate(
        self,
        prompt: str,
        layer: int,
        neuron_doses: Optional[Dict[int, float]] = None,
        vector: Optional[torch.Tensor] = None,
        neuron_scales: Optional[Dict[int, float]] = None,
        neuron_sets: Optional[Dict[int, float]] = None,
        max_new_tokens: int = 40,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a continuation with steering active on every position."""
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_prompt = inputs["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad(), self.steering(
            layer,
            add=vector,
            neuron_doses=neuron_doses,
            neuron_scales=neuron_scales,
            neuron_sets=neuron_sets,
        ):
            result = self.model.generate(**inputs, **gen_kwargs)

        seq = result.sequences[0]
        new_ids = seq[n_prompt:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        mean_logprob = None
        if getattr(result, "scores", None):
            logps = []
            for step, scores in enumerate(result.scores):
                lp = torch.log_softmax(scores[0].float(), dim=-1)
                logps.append(float(lp[new_ids[step]]))
            if logps:
                mean_logprob = sum(logps) / len(logps)

        return {
            "text": text,
            "new_tokens": int(new_ids.shape[0]),
            "mean_logprob": mean_logprob,
        }

    # ---------- activation capture ----------

    def activations_for_text(self, text: str, layer: int) -> Tuple[List[str], torch.Tensor]:
        """Per-token MLP activations at one layer.

        Returns (token_strings, activations[n_tokens, intermediate]).
        Captures the input to the down-projection, i.e. the post-activation
        vector a in ``MLP(x) = down(a)``.
        """
        _, module = self.find_down_proj(layer)
        captured: List[torch.Tensor] = []

        def pre_hook(mod, args):
            if isinstance(args[0], torch.Tensor):
                captured.append(args[0].detach())
            return None

        handle = module.register_forward_pre_hook(pre_hook)
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                self.model(**inputs)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("No activation captured; is this a decoder-only causal LM?")
        acts = captured[0][0].float().cpu()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
        return tokens, acts

    def top_neurons(self, text: str, layer: int, k: int = 10) -> List[Dict[str, Any]]:
        """Top-k neurons by mean absolute activation on the text."""
        _, acts = self.activations_for_text(text, layer)
        score = acts.abs().mean(dim=0)
        vals, idxs = torch.topk(score, min(k, score.shape[0]))
        return [
            {"neuron": int(i), "mean_abs_act": float(v)}
            for v, i in zip(vals.tolist(), idxs.tolist())
        ]

    # ---------- sweep ----------

    @staticmethod
    def parse_neuron_spec(spec: str, intermediate: int, rng: random.Random) -> List[int]:
        """'all' | 'random:N' | 'a-b' | 'a,b,c' -> list of neuron indices."""
        spec = (spec or "").strip().lower()
        if spec in ("", "all"):
            return list(range(intermediate))
        if spec.startswith("random:"):
            n = int(spec.split(":", 1)[1])
            n = max(1, min(n, intermediate))
            return sorted(rng.sample(range(intermediate), n))
        if "," in spec:
            return sorted({int(p) for p in spec.split(",") if p.strip()})
        if "-" in spec:
            a, b = spec.split("-", 1)
            return list(range(max(0, int(a)), min(intermediate - 1, int(b)) + 1))
        return [int(spec)]

    def run_sweep(
        self,
        layer: int,
        neurons: Sequence[int],
        prompts: Sequence[str],
        doses: Sequence[float],
        n_controls: int = 0,
        max_new_tokens: int = 40,
        temperature: float = 0.7,
        dose_mode: str = "rel",
        run_dir: str = RUNS_DIR,
        run_name: Optional[str] = None,
        seed: int = 0,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Push every listed neuron up/down on every prompt; keep every rollout.

        Writes one JSONL record per rollout to run_dir/<run_name>.jsonl as it
        goes, so partial runs stay browsable. Returns (path, stats).
        """
        os.makedirs(run_dir, exist_ok=True)
        run_name = run_name or f"{getattr(self.model.config, '_name_or_path', 'model')}-L{layer}-{int(time.time())}"
        path = os.path.join(run_dir, f"{run_name}.jsonl")

        med = 1.0
        if dose_mode == "rel":
            med = self.median_residual_norm(layer, prompts)
        hidden = self.hidden_size()
        device = self.model.device
        dtype = next(self.model.parameters()).dtype

        def vec_for_neuron(i: int, dose: float) -> torch.Tensor:
            _, mod = self.find_down_proj(layer)
            col = neuron_write_column(mod, i).detach().float()
            col = col / (col.norm() + 1e-8)
            return (col * dose * med).to(device=device, dtype=dtype)

        rng = random.Random(seed)
        records: List[Dict[str, Any]] = []
        total = len(prompts) + len(neurons) * len(prompts) * len(doses) + n_controls * len(prompts)
        done = 0
        t0 = time.time()

        def emit(rec: Dict[str, Any]):
            records.append(rec)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        def tick(msg: str):
            if progress:
                progress(done, total, msg)

        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "kind": "meta",
                "model": getattr(self.model.config, "_name_or_path", "model"),
                "layer": layer,
                "down_proj": self.find_down_proj(layer)[0],
                "prompts": list(prompts),
                "doses": list(doses),
                "neurons": list(neurons),
                "n_controls": n_controls,
                "dose_mode": dose_mode,
                "median_residual_norm": med,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }) + "\n")

        rollout = 0

        # unsteered baselines, one per prompt
        for pid, prompt in enumerate(prompts):
            out = self.steered_generate(
                prompt, layer, max_new_tokens=max_new_tokens,
                temperature=temperature, seed=seed + rollout,
            )
            rollout += 1
            emit({"kind": "baseline", "prompt_id": pid, **out})
            done += 1
            tick("baseline")

        # neuron rollouts
        for n_i, neuron in enumerate(neurons):
            for pid, prompt in enumerate(prompts):
                for dose in doses:
                    vec = vec_for_neuron(neuron, dose)
                    out = self.steered_generate(
                        prompt, layer, vector=vec,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature, seed=seed + rollout,
                    )
                    rollout += 1
                    emit({
                        "kind": "neuron", "neuron": int(neuron),
                        "prompt_id": pid, "dose": dose, **out,
                    })
                    done += 1
                    tick(f"neuron {neuron} ({n_i+1}/{len(neurons)})")

        # random-direction controls
        for c in range(n_controls):
            for pid, prompt in enumerate(prompts):
                dose = rng.choice(list(doses)) if doses else 1.0
                v = torch.randn(hidden, device=device, dtype=torch.float32)
                v = (v / (v.norm() + 1e-8) * dose * med).to(dtype)
                out = self.steered_generate(
                    prompt, layer, vector=v,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature, seed=seed + rollout,
                )
                rollout += 1
                emit({
                    "kind": "control", "control": c,
                    "prompt_id": pid, "dose": dose, **out,
                })
                done += 1
                tick(f"control {c+1}/{n_controls}")

        stats = {
            "run": path,
            "layer": layer,
            "neurons": len(neurons),
            "rollouts": done,
            "seconds": round(time.time() - t0, 1),
            "median_residual_norm": med,
        }
        return path, stats

    # ---------- run files: load + rank ----------

    @staticmethod
    def load_run(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        meta, records = {}, []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == "meta":
                    meta = rec
                else:
                    records.append(rec)
        return meta, records

    @staticmethod
    def _words(text: str) -> List[str]:
        return _WORD_RE.findall((text or "").lower())

    @classmethod
    def rank_run(cls, path: str, top_words: int = 8) -> List[Dict[str, Any]]:
        """Score each unit by word-frequency shift vs the whole run.

        A unit is one neuron or one random-direction control. For each unit,
        words that are over-represented relative to the run's background
        distribution contribute their excess count to a shift score. Each
        neuron's score is reported with its percentile against the control
        score distribution — the same role controls play in the original
        explorer's cluster-mass ranking.
        """
        meta, records = cls.load_run(path)
        by_unit: Dict[str, List[str]] = {}
        unit_info: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            if rec["kind"] == "neuron":
                uid = f"n{rec['neuron']}"
                unit_info.setdefault(uid, {"kind": "neuron", "neuron": rec["neuron"]})
            elif rec["kind"] == "control":
                uid = f"c{rec.get('control', 0)}"
                unit_info.setdefault(uid, {"kind": "control", "control": rec.get("control", 0)})
            else:
                uid = "baseline"
                unit_info.setdefault(uid, {"kind": "baseline"})
            by_unit.setdefault(uid, []).append(rec.get("text", ""))

        # background word distribution over the whole run
        bg: Dict[str, int] = {}
        bg_total = 0
        for texts in by_unit.values():
            for t in texts:
                for w in cls._words(t):
                    bg[w] = bg.get(w, 0) + 1
                    bg_total += 1
        if bg_total == 0:
            return []

        alpha = 0.5
        rows = []
        for uid, texts in by_unit.items():
            counts: Dict[str, int] = {}
            n_words = 0
            for t in texts:
                for w in cls._words(t):
                    counts[w] = counts.get(w, 0) + 1
                    n_words += 1
            if n_words == 0:
                continue
            contributions = []
            for w, c in counts.items():
                expected = (bg[w] + alpha) / (bg_total + alpha * len(bg)) * n_words
                if c > expected and c >= 2:
                    contributions.append((c - expected, w, c))
            contributions.sort(reverse=True)
            score = sum(x[0] for x in contributions) / n_words
            rows.append({
                "unit": uid,
                **unit_info[uid],
                "score": score,
                "top_words": ", ".join(f"{w}×{c}" for _, w, c in contributions[:top_words]),
                "rollouts": len(texts),
            })

        control_scores = sorted(r["score"] for r in rows if r["kind"] == "control")
        for r in rows:
            if r["kind"] == "neuron" and control_scores:
                r["control_pctl"] = round(
                    100.0 * sum(s <= r["score"] for s in control_scores) / len(control_scores), 1
                )
            elif r["kind"] == "neuron":
                r["control_pctl"] = None
            else:
                r["control_pctl"] = None

        rows.sort(key=lambda r: (r["kind"] != "neuron", -r["score"]))
        return rows

    @staticmethod
    def neuron_detail(path: str, neuron: int) -> Dict[str, Any]:
        """All rollouts for one neuron, grouped prompt x dose, plus baselines."""
        meta, records = NeuronExplorer.load_run(path)
        prompts = meta.get("prompts", [])
        rollouts = []
        baselines = {}
        for rec in records:
            if rec["kind"] == "neuron" and rec.get("neuron") == neuron:
                rollouts.append(rec)
            elif rec["kind"] == "baseline":
                baselines[rec["prompt_id"]] = rec
        return {
            "meta": meta,
            "prompts": prompts,
            "baselines": baselines,
            "rollouts": sorted(rollouts, key=lambda r: (r["prompt_id"], r["dose"])),
        }


def list_runs(run_dir: str = RUNS_DIR) -> List[str]:
    if not os.path.isdir(run_dir):
        return []
    return sorted(
        (os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith(".jsonl")),
        key=os.path.getmtime,
        reverse=True,
    )


class NeuronSculptor:
    """Persistent weight edits equivalent to neuron steering.

    Each op edits the live model's weights and stores the original tensor
    slice, so ops can be undone individually or all at once:

    - ``scale_write`` / ``ablate``: scale or zero a neuron's down-projection
      column. Exactly equivalent to multiplying/zeroing its activation.
    - ``bias_push``: add ``dose x column`` to the down-projection bias.
      Exactly equivalent to constant additive steering — only possible on
      modules that have a bias (GPT-2 ``mlp.c_proj`` does; Llama
      ``down_proj`` does not).
    - ``scale_read``: scale a neuron's up/gate-projection row, changing how
      strongly it detects its input feature. Has no exact steering
      equivalent (the activation nonlinearity sits in between).

    Wraps a NeuronExplorer for module discovery.
    """

    UP_HINTS = (
        "mlp", "fc", "ffn", "feed_forward", "dense_h_to_4h",
        "up_proj", "gate_proj", "c_fc",
    )

    def __init__(self, explorer: NeuronExplorer):
        self.ex = explorer
        self.ops: List[Dict[str, Any]] = []
        self._next_id = 0

    def find_up_projs(self, layer: int) -> List[Tuple[str, nn.Module]]:
        """Hidden -> intermediate projections in a block (up/gate rows).

        For gated MLPs this returns both gate_proj and up_proj.
        """
        _, stack = self.ex.find_block_stack()
        block = stack[layer]
        try:
            inter = self.ex.intermediate_size(layer)
        except Exception:
            inter = None
        cands = []
        for name, module in block.named_modules():
            dims = _param_io_dims(module)
            if dims is None or not isinstance(getattr(module, "weight", None), torch.Tensor):
                continue
            in_dim, out_dim = dims
            if in_dim >= out_dim:
                continue
            hinted = any(h in name.lower() for h in self.UP_HINTS)
            if hinted or (inter is not None and out_dim == inter):
                cands.append((name, module))
        cands.sort(key=lambda c: c[0])
        return cands

    def _record(self, kind: str, layer: int, module_name: str,
                details: str, backups: List[Tuple]) -> Dict[str, Any]:
        self._next_id += 1
        op = {
            "id": self._next_id,
            "kind": kind,
            "layer": layer,
            "module": module_name,
            "details": details,
            "backups": backups,
            "ts": time.strftime("%H:%M:%S"),
        }
        self.ops.append(op)
        return op

    def apply_scale_write(self, layer: int, neuron: int, factor: float) -> Dict[str, Any]:
        name, mod = self.ex.find_down_proj(layer)
        idx = _neuron_write_index(mod, int(neuron))
        orig = mod.weight.data[idx].clone()
        mod.weight.data[idx] *= float(factor)
        return self._record(
            "scale_write", layer, name,
            f"neuron {neuron} write column ×{factor}",
            [(mod.weight, idx, orig)],
        )

    def apply_ablate(self, layer: int, neuron: int) -> Dict[str, Any]:
        name, mod = self.ex.find_down_proj(layer)
        idx = _neuron_write_index(mod, int(neuron))
        orig = mod.weight.data[idx].clone()
        mod.weight.data[idx] = 0
        return self._record(
            "ablate", layer, name,
            f"neuron {neuron} write column zeroed",
            [(mod.weight, idx, orig)],
        )

    def apply_bias_push(
        self,
        layer: int,
        neuron: int,
        dose: float,
        dose_mode: str = "rel",
        calib_prompts: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        name, mod = self.ex.find_down_proj(layer)
        if not isinstance(getattr(mod, "bias", None), torch.Tensor):
            raise RuntimeError(
                f"{name} has no bias — use scale/ablate, or scale_read instead."
            )
        col = neuron_write_column(mod, int(neuron)).detach().float()
        col = col / (col.norm() + 1e-8)
        med = (
            self.ex.median_residual_norm(layer, list(calib_prompts) if calib_prompts else ["The"])
            if dose_mode == "rel"
            else 1.0
        )
        vec = (col * float(dose) * med).to(mod.bias.dtype)
        orig = mod.bias.data.clone()
        mod.bias.data += vec
        return self._record(
            "bias_push", layer, name,
            f"neuron {neuron} bias push dose {dose} ({dose_mode}, med {med:.1f})",
            [(mod.bias, slice(None), orig)],
        )

    def apply_scale_read(self, layer: int, neuron: int, factor: float) -> Dict[str, Any]:
        ups = self.find_up_projs(layer)
        if not ups:
            raise RuntimeError(f"No up-projection found in block {layer}.")
        backups = []
        names = []
        for name, mod in ups:
            idx = _neuron_read_index(mod, int(neuron))
            orig = mod.weight.data[idx].clone()
            mod.weight.data[idx] *= float(factor)
            backups.append((mod.weight, idx, orig))
            names.append(name)
        return self._record(
            "scale_read", layer, ", ".join(names),
            f"neuron {neuron} read row(s) ×{factor} on {len(ups)} module(s)",
            backups,
        )

    def undo(self, op_id: Optional[int] = None) -> str:
        if not self.ops:
            return "No sculpt ops to undo."
        if op_id is None:
            op = self.ops.pop()
        else:
            op = next((o for o in self.ops if o["id"] == op_id), None)
            if op is None:
                return f"No op #{op_id}."
            self.ops.remove(op)
        for param, idx, orig in op["backups"]:
            param.data[idx] = orig
        return f"Undid op #{op['id']} ({op['kind']}: {op['details']})."

    def undo_all(self) -> str:
        n = len(self.ops)
        while self.ops:
            self.undo()
        return f"Undid {n} op(s)." if n else "No sculpt ops to undo."

    def describe_ops(self) -> str:
        if not self.ops:
            return "No sculpt ops applied."
        return "\n".join(
            f"#{o['id']} {o['kind']} L{o['layer']} {o['module']} — {o['details']} @ {o['ts']}"
            for o in self.ops
        )
