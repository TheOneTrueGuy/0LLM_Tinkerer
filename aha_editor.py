"""AHA-style self-supervised weight-space programming, ported for LLM Tinkerer.

Adapted from Eppy-EpHard/AHA (MIT license) — "Weight-Space Programming Without
LLMs" (Khamis, Tutunov & Bou Ammar, Nov 2026), which builds on Nathan Barry's
weight-space-lp (MIT) and Takehi et al.'s agent edit.

Pipeline (no external LLM anywhere):
  1. sample_traces   — generate answers on practice problems, label each token
                       productive / loop-onset / inside-loop via dup detector
  2. make_pairs      — cut each looping trace 8 tokens before onset, resample
                       continuations; a prefix yielding both a looping and a
                       productive branch is a matched pair
  3. build_direction — u_l = normalise(mean over pairs of resid(loop) - resid(prod))
  4. collect         — MLP inputs h at labelled positions, per layer
  5. pdhg            — min Σ|v_j| s.t. v·h_loop <= -δ, |v·h_prod| <= ε
  6. search_layers   — solve the LP at every layer, rank by held-out hit rate
  7. apply_edit      — W += u v^T on the chosen layers' down_proj (undoable)

Differences from upstream: Hugging Face generate instead of vLLM (runs on CPU,
though slowly), a live model object instead of a model dir, generic decoder
layer discovery (GPT-2, Llama/Qwen families, VLM language models), and the
edit returns an undo payload instead of writing a safetensors file.
"""

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ======================================================================
# Part 1 — loop detection (ported from aha/loops.py, itself adapted from
# weight-space-lp, MIT)

K, FRAC, GAP = 16, 0.8, 256


def dup_mask(ids: np.ndarray, k: int = K) -> np.ndarray:
    """dup[t] = 1 if the k-token window ending at t already occurred earlier."""
    seen, out = set(), np.zeros(len(ids), bool)
    for t in range(k - 1, len(ids)):
        w = ids[t - k + 1:t + 1].tobytes()
        out[t] = w in seen
        seen.add(w)
    return out


def onset(dup: np.ndarray, frac: float = FRAC, min_tail: int = 512) -> Optional[int]:
    """First copied token after which >= frac of the remaining tokens are
    copies, with at least min_tail tokens left (the trace never recovers)."""
    n = len(dup)
    if n < min_tail:
        return None
    suffix = np.cumsum(dup[::-1])[::-1] / np.arange(n, 0, -1)
    ok = np.flatnonzero((suffix >= frac) & dup & (np.arange(n) <= n - min_tail))
    return int(ok[0]) if len(ok) else None


@dataclass
class Labels:
    onset: Optional[int]     # in response-token coordinates
    R_onset: np.ndarray      # the decision to start copying
    R_during: np.ndarray     # copied tokens deep inside the loop
    P: np.ndarray            # productive tokens


def label(resp_ids: np.ndarray, cut: bool) -> Labels:
    dup = dup_mask(resp_ids)
    o = onset(dup) if cut else None
    n = len(resp_ids)
    if o is None:
        P = np.arange(n) if not cut else np.arange(0)
        return Labels(None, np.arange(0), np.arange(0), P)
    R_on = np.arange(max(o - 16, 0), min(o + 48, n))
    t = np.arange(o + 64, n)
    return Labels(o, R_on, t[dup[o + 64:]], np.arange(0, max(o - GAP, 0)))


# ======================================================================
# Part 2 — model plumbing (generic version of aha/model.py)


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decoder_layers(model) -> List[torch.nn.Module]:
    """Generic discovery of the decoder stack. Handles plain causal LMs
    (model.model.layers, model.transformer.h) and VLMs with a nested
    language model."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    if hasattr(inner, "layers"):
        return list(inner.layers)
    if hasattr(inner, "h"):          # GPT-2
        return list(inner.h)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("cannot find decoder layers on this architecture")


def _layer_down_proj(layer) -> torch.nn.Module:
    for name in ("down_proj", "fc_out", "c_proj"):
        m = getattr(getattr(layer, "mlp", layer), name, None)
        if m is None and hasattr(layer, "mlp"):
            m = getattr(layer.mlp, name, None)
        if m is not None:
            return m
    # gpt2: layer.mlp.c_proj ; llama/qwen: layer.mlp.down_proj
    if hasattr(layer, "mlp"):
        for attr in vars(layer.mlp):
            m = getattr(layer.mlp, attr)
            if isinstance(m, torch.nn.Linear) and m.out_features < m.in_features:
                return m
    raise ValueError("no down-projection found on layer")


@contextmanager
def capture(model, layer_ids, what=("h", "resid")):
    """Per layer: 'h' = down_proj input [T, d_mlp]; 'resid' = layer output
    [T, d_model]. Batch size 1."""
    store: Dict[int, Dict[str, torch.Tensor]] = {l: {} for l in layer_ids}
    hooks = []
    for l in layer_ids:
        layer = decoder_layers(model)[l]
        if "h" in what:
            mod = _layer_down_proj(layer)
            hooks.append(mod.register_forward_pre_hook(
                lambda m, args, l=l: store[l].__setitem__("h", args[0][0].detach())))
        if "resid" in what:
            hooks.append(layer.register_forward_hook(
                lambda m, args, out, l=l: store[l].__setitem__(
                    "resid", (out[0] if isinstance(out, tuple) else out)[0].detach())))
    try:
        yield store
    finally:
        for hook in hooks:
            hook.remove()


def down_proj_weight(model, layer_idx: int) -> torch.Tensor:
    return _layer_down_proj(decoder_layers(model)[layer_idx]).weight.detach()


def _backbone_forward(model, ids) -> None:
    """Run the decoder stack without the lm_head when possible — the capture
    hooks fire either way, and skipping the head saves a big logit matmul."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        inner.language_model(ids)
    elif inner is not None and hasattr(inner, "layers"):
        inner(ids)
    elif hasattr(model, "transformer"):
        model.transformer(ids)
    else:
        model(ids)


# ======================================================================
# Part 3 — trace sampling (HF generate instead of vLLM)

PROBLEMS_SMOKE = [
    "Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
    "A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?",
    "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
    "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
    "A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many as the downloads in the first month, but then reduced by 30% in the third month. How many downloads did the program have in the third month?",
    "Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?",
]


def chat_prompt_ids(tok, problem: str, thinking: bool = True) -> List[int]:
    msgs = [{"role": "user", "content": problem}]
    try:
        text = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=thinking)
    except TypeError:
        text = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        text = f"Question: {problem}\nAnswer:"
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def sample_traces(model, tok, problems: Sequence[str], max_new: int = 8192,
                  temperature: float = 0.6, top_p: float = 0.95,
                  top_k: int = 20, repetition_penalty: float = 1.0,
                  seed: int = 0, progress=None) -> List[dict]:
    """One trace per problem. Returns [{id, text, cut, resp_ids}]."""
    traces = []
    dev = next(model.parameters()).device
    for i, prob in enumerate(problems):
        ids = torch.tensor([chat_prompt_ids(tok, prob)], device=dev)
        gen = model.generate(
            ids, max_new_tokens=max_new, do_sample=True,
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tok.eos_token_id)[0]
        resp = gen[ids.shape[1]:]
        traces.append({"id": i, "problem": prob, "resp_ids": resp.cpu().numpy(),
                       "text": tok.decode(resp, skip_special_tokens=True),
                       "cut": len(resp) >= max_new})
        if progress:
            progress(i + 1, len(problems))
        logger.info(f"trace {i}: {len(resp)} tokens, cut={traces[-1]['cut']}")
    return traces


def looping_prefixes(tok, traces, margin: int = 8, max_prefix: int = 300):
    """Prefixes cut `margin` tokens before each looping trace's onset."""
    out = []
    for t in traces:
        if not t["cut"]:
            continue
        o = onset(dup_mask(t["resp_ids"]))
        if o is None or o < 64:
            continue
        prompt = chat_prompt_ids(tok, t["problem"])
        cut = max(o - margin, 16)
        out.append({"id": t["id"], "onset": int(o), "n_prompt": len(prompt),
                    "ids": prompt + t["resp_ids"][:cut].tolist()})
    if len(out) > max_prefix:
        out = [out[i] for i in
               np.random.default_rng(1).permutation(len(out))[:max_prefix]]
    return out


@torch.no_grad()
def make_pairs(model, tok, traces, n_cont: int = 8, max_new: int = 3072,
               margin: int = 64, max_prefix: int = 300,
               temperature: float = 1.0, top_p: float = 0.95,
               top_k: int = 20, progress=None) -> List[dict]:
    """Resample continuations of every looping prefix at penalty 0.
    Returns [{id, onset, n_prompt, prefix_ids, conts:[{ids, loops, onset, hit_cap}]}].
    """
    prefixes = looping_prefixes(tok, traces, margin, max_prefix)
    if not prefixes:
        raise RuntimeError("no looping traces to branch from")
    dev = next(model.parameters()).device
    records = []
    for i, p in enumerate(prefixes):
        ids = torch.tensor([p["ids"]], device=dev)
        gen = model.generate(
            ids, max_new_tokens=max_new, do_sample=True,
            temperature=temperature, top_p=top_p, top_k=top_k,
            num_return_sequences=n_cont,
            pad_token_id=tok.eos_token_id)
        conts = []
        for row in gen:
            resp = row[ids.shape[1]:].cpu().numpy()
            o = onset(dup_mask(resp))
            conts.append({"ids": resp.tolist(), "loops": o is not None,
                          "onset": o, "hit_cap": len(resp) >= max_new})
        records.append({"id": p["id"], "onset": p["onset"],
                        "n_prompt": p["n_prompt"], "prefix_ids": p["ids"],
                        "conts": conts})
        if progress:
            progress(i + 1, len(prefixes))
        logger.info(f"prefix {i}: {sum(c['loops'] for c in conts)}/{n_cont} looping")
    return records


def matched_pairs(records) -> List[Tuple[dict, dict, dict]]:
    """Per prefix with both kinds: earliest-looping + longest non-looping."""
    pairs = []
    for r in records:
        loops = [c for c in r["conts"] if c["loops"]]
        others = [c for c in r["conts"] if not c["loops"]]
        if loops and others:
            pairs.append((r, min(loops, key=lambda c: c["onset"]),
                          max(others, key=lambda c: len(c["ids"]))))
    return pairs


@torch.no_grad()
def build_direction(model, records, window: int = 160, offset: int = 64,
                    min_pairs: int = 10, heldout_share: float = 0.2,
                    progress=None) -> dict:
    """u per layer from matched pairs. Returns {u: {l: np.array},
    separation: [{layer, cohens_d, pair_agreement}], n_pairs}."""
    pairs = matched_pairs(records)
    if not pairs:
        raise RuntimeError(
            "no matched pairs — every prefix's continuations all looped or "
            "all finished. Try a larger margin (cut further back) or higher "
            "temperature for more divergent branches.")
    layers = list(range(len(decoder_layers(model))))
    dev = next(model.parameters()).device
    diffs = {l: [] for l in layers}
    means = {l: {"loop": [], "other": []} for l in layers}
    used = 0
    for i, (record, looping, other) in enumerate(pairs):
        lo, hi = looping["onset"] + offset, looping["onset"] + offset + window
        if hi > len(looping["ids"]) or hi > len(other["ids"]):
            continue
        prefix = record["prefix_ids"]
        window_means = {}
        for key, cont in (("loop", looping), ("other", other)):
            ids = torch.tensor(prefix + cont["ids"][:hi], device=dev)[None]
            positions = torch.arange(len(prefix) + lo, len(prefix) + hi,
                                     device=dev)
            with capture(model, layers, what=("resid",)) as store:
                _backbone_forward(model, ids)
            window_means[key] = {l: store[l]["resid"][positions]
                                 .float().mean(0).cpu().numpy()
                                 for l in layers}
        for l in layers:
            diffs[l].append(window_means["loop"][l] - window_means["other"][l])
            means[l]["loop"].append(window_means["loop"][l])
            means[l]["other"].append(window_means["other"][l])
        used += 1
        if progress:
            progress(i + 1, len(pairs))
    if used < min_pairs:
        raise RuntimeError(f"too few pairs ({used} < {min_pairs})")

    n = used
    heldout = np.random.default_rng(0).permutation(n)[
        :max(1, min(int(n * heldout_share), n - 1))]
    fit = np.setdiff1d(np.arange(n), heldout)
    u, separation = {}, []
    for l in layers:
        vec = np.stack(diffs[l])[fit].mean(0)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        a = np.stack(means[l]["loop"])[heldout] @ vec
        b = np.stack(means[l]["other"])[heldout] @ vec
        cohens_d = (a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-9)
        u[l] = vec.astype(np.float32)
        separation.append({"layer": l, "cohens_d": round(float(cohens_d), 3),
                           "pair_agreement": round(float((a > b).mean()), 3)})
    return {"u": u, "separation": separation, "n_pairs": n}


# ======================================================================
# Part 4 — activation collection for the LP

TAIL, MAX_FINISHED, MAX_CONTEXT = 1024, 8192, 16384
PER_TRACE = {0: 256, 1: 64, 2: 64}   # productive / onset / in-loop


@torch.no_grad()
def collect_activations(model, tok, traces, layers: List[int], seed: int = 0,
                        progress=None) -> dict:
    """h and resid at sampled labelled positions of each trace.
    Returns {h: {l: np.array}, meta: {label, trace, pos}}."""
    rng = np.random.default_rng(seed)
    dev = next(model.parameters()).device
    h = {l: [] for l in layers}
    resid = {l: [] for l in layers}
    meta = {"label": [], "trace": [], "pos": []}
    for t, trace in enumerate(traces):
        prompt = chat_prompt_ids(tok, trace["problem"])
        answer = trace["resp_ids"].astype(np.int64)
        labels = label(answer, trace["cut"])
        if labels.onset is None and trace["cut"]:
            continue
        end = labels.onset + TAIL if labels.onset is not None else MAX_FINISHED
        end = min(end, len(answer), MAX_CONTEXT - len(prompt))
        picked = []
        for lab, positions in ((0, labels.P), (1, labels.R_onset), (2, labels.R_during)):
            positions = positions[positions < end]
            if len(positions) > PER_TRACE[lab]:
                positions = np.sort(rng.choice(positions, PER_TRACE[lab], replace=False))
            picked += [(lab, int(p)) for p in positions]
        if not picked:
            continue
        ids = torch.tensor([prompt + answer[:end].tolist()], device=dev)
        rows = torch.tensor([len(prompt) + p for _, p in picked], device=dev)
        with capture(model, layers, what=("h", "resid")) as store:
            _backbone_forward(model, ids)
        for l in layers:
            h[l].append(store[l]["h"][rows].to(torch.float16).cpu().numpy())
            resid[l].append(
                store[l]["resid"][rows].to(torch.float16).cpu().numpy())
        meta["label"] += [lab for lab, _ in picked]
        meta["trace"] += [t] * len(picked)
        meta["pos"] += [p for _, p in picked]
        if progress:
            progress(t + 1, len(traces))
    return {"h": {l: np.concatenate(h[l]) for l in layers},
            "resid": {l: np.concatenate(resid[l]) for l in layers},
            "meta": {k: np.array(v) for k, v in meta.items()}}


def build_direction_unmatched(collected: dict) -> Dict[int, np.ndarray]:
    """Barry's fallback when no matched pairs exist: u is the difference
    between the mean residual stream on in-loop tokens and on productive
    tokens. Works 'most of the way' — no shared prefix, so other
    correlates ride along."""
    resid, meta = collected["resid"], collected["meta"]
    labels = meta["label"]
    out = {}
    for l in resid:
        r = np.asarray(resid[l], np.float32)
        loop, prod = r[labels == 2], r[labels == 0]
        if len(loop) < 50 or len(prod) < 50:
            continue
        vec = loop.mean(0) - prod.mean(0)
        out[l] = (vec / (np.linalg.norm(vec) + 1e-9)).astype(np.float32)
    return out


@contextmanager
def steer(model, layer_vecs: Dict[int, Tuple[np.ndarray, float]]):
    """Add alpha*u to each listed decoder layer's residual output during
    generation. {layer: (u, alpha)} — u unit-norm, alpha signed (negative
    subtracts the direction). Reversible diagnostic: is u causal at all?"""
    layers = decoder_layers(model)
    handles = []
    for l, (u, alpha) in layer_vecs.items():
        vec = torch.tensor(alpha * np.asarray(u, np.float32))

        def make(v=vec):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                hh = h + v.to(device=h.device, dtype=h.dtype)[None, None, :]
                return (hh,) + tuple(out[1:]) if isinstance(out, tuple) else hh
            return hook
        handles.append(layers[l].register_forward_hook(make()))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def proj_gap(collected: dict, u: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Mean residual-u gap between in-loop and productive positions, per
    layer: (loop_resid - prod_resid)·u. Natural scale for steer alpha —
    alpha = k * gap moves a productive token's projection by k gaps."""
    resid, labels = collected["resid"], collected["meta"]["label"]
    out = {}
    for l, vec in u.items():
        r = np.asarray(resid[l], np.float32)
        loop, prod = r[labels == 2], r[labels == 0]
        if len(loop) < 20 or len(prod) < 20:
            continue
        out[l] = float(loop.mean(0) @ vec - prod.mean(0) @ vec)
    return out


# ======================================================================
# Part 5 — the linear program (ported from aha/edit.py, from Barry's
# weight-space-lp, MIT): PDHG, L1-sparse v.


def pdhg(h_loop: np.ndarray, h_prod: np.ndarray, delta: float, eps: float,
         C: float = 10.0, cost: Optional[np.ndarray] = None,
         support: Optional[np.ndarray] = None, v_max: float = 10.0,
         iters: int = 20000, tol: float = 1e-5, omega: float = 100.0) -> np.ndarray:
    """min Σ c_j|v_j| + C Σ relu(h·v + δ) [loop] + C Σ relu(|h·v| − ε) [prod].
    Only products with K = [h_loop; h_prod] are needed; L1 step gives an
    exactly sparse v."""
    dev = device()
    cols = (np.arange(h_loop.shape[1]) if support is None
            else np.asarray(support))
    K_ = torch.tensor(np.concatenate([h_loop[:, cols], h_prod[:, cols]]),
                      dtype=torch.float32, device=dev)
    n_loop = len(h_loop)
    delta_t = torch.full((n_loop,), float(delta), device=dev)
    eps_t = torch.tensor(float(eps), device=dev)
    cost_t = (torch.ones(len(cols), device=dev) if cost is None
              else torch.tensor(cost[cols], dtype=torch.float32, device=dev))
    x = torch.randn(len(cols), device=dev)
    for _ in range(50):
        x = K_.T @ (K_ @ x)
        x /= x.norm()
    norm = float((K_ @ x).norm())
    tau, sigma = 0.95 / (norm * omega), 0.95 * omega / norm
    v = torch.zeros(len(cols), device=dev)
    v_bar, y = v.clone(), torch.zeros(K_.shape[0], device=dev)

    def objective(vv):
        z = K_ @ vv
        return float((cost_t * vv.abs()).sum()
                     + C * torch.relu(z[:n_loop] + delta_t).sum()
                     + C * torch.relu(z[n_loop:].abs() - eps_t).sum())

    best, history = (np.inf, v.clone()), []
    for it in range(iters):
        w = y + sigma * (K_ @ v_bar)
        y = torch.cat([(w[:n_loop] + sigma * delta_t).clamp(0, C),
                       torch.sign(w[n_loop:])
                       * (w[n_loop:].abs() - sigma * eps_t).clamp(min=0).clamp(max=C)])
        uu = v - tau * (K_.T @ y)
        v_new = (torch.sign(uu) * (uu.abs() - tau * cost_t).clamp(min=0)).clamp(-v_max, v_max)
        v_bar = 2 * v_new - v
        v = v_new
        if it % 200 == 0 or it == iters - 1:
            value = objective(v)
            if value < best[0]:
                best = (value, v.clone())
            history.append(value)
            if it > 2000 and len(history) > 5 and \
                    abs(history[-5] - value) < tol * max(1.0, abs(value)):
                break
    solution = np.zeros(h_loop.shape[1])
    solution[cols] = best[1].cpu().numpy()
    return solution


# ======================================================================
# Part 6 — layer search and the edit


def search_layers(model, u: Dict[int, np.ndarray], collected: dict,
                  eps: float = 1.0, k: int = 8, mult: float = 4.0,
                  seed: int = 0, heldout_share: float = 0.2,
                  pro: bool = False,
                  progress=None) -> Tuple[List[dict], Dict[int, np.ndarray]]:
    """Solve the LP at every layer; rank by held-out hit rate then
    damage-per-delta. Returns (ranked rows, {l: v}).
    pro=True inverts the edit: productive tokens are pushed TOWARD the loop
    direction (h·v >= +δ) while looping tokens are kept within ε — the
    bidirectional control experiment for the de-loop edit."""
    layers = sorted(u.keys())
    h_all, meta = collected["h"], collected["meta"]
    labels = meta["label"]
    # Position-level holdout, stratified implicitly: holding out whole traces
    # can remove a scarce class entirely when few traces carry it (e.g. only
    # finished traces carry productive positions in a mostly-looping run).
    heldout = np.random.default_rng(seed).random(len(labels)) < heldout_share
    rows, solutions = [], {}
    for idx, l in enumerate(layers):
        h = np.asarray(h_all[l], np.float32)
        fit_loop = (labels == 2) & ~heldout
        fit_prod = (labels == 0) & ~heldout
        if fit_loop.sum() < 50 or fit_prod.sum() < 50:
            logger.info(f"L{l}: skipped, too few positions")
            continue
        vec = u[l].astype(np.float32)
        W = down_proj_weight(model, l).float().cpu().numpy()  # [d_model, d_mlp]
        writes = h @ (vec @ W)     # how much u the layer writes per token
        delta = float(mult * (writes[fit_loop].mean() - writes[fit_prod].mean()))
        if not np.isfinite(delta) or delta <= 0:
            logger.info(f"L{l}: skipped, delta={delta:.3g}")
            continue
        rng = np.random.default_rng(l if seed == 0 else [seed, l])
        loop_rows = rng.choice(np.flatnonzero(fit_loop),
                               min(4000, int(fit_loop.sum())), replace=False)
        prod_rows = rng.choice(np.flatnonzero(fit_prod),
                               min(8000, int(fit_prod.sum())), replace=False)
        if pro:
            push, keep = -h[prod_rows], h[loop_rows]
        else:
            push, keep = h[loop_rows], h[prod_rows]
        dense = pdhg(push, keep, delta, eps, cost=np.ones(h.shape[1]))
        neurons = np.sort(np.argsort(
            -np.abs(dense) * h[fit_loop | fit_prod].std(0))[:k])
        v = pdhg(push, keep, delta, eps,
                 cost=np.full(h.shape[1], 1e-3), support=neurons)

        shift = h[heldout] @ v
        on_loops = shift[labels[heldout] == 2]
        on_prod = shift[labels[heldout] == 0]
        if pro:
            hit = float((on_prod >= delta).mean())
            damage = abs(float(on_loops.mean())) / delta
        else:
            hit = float((on_loops <= -delta).mean())
            damage = abs(float(on_prod.mean())) / delta
        rows.append({"layer": l, "delta": round(delta, 3),
                     "hit_rate": round(hit, 3),
                     "loop_shift": round(float(on_loops.mean()), 3),
                     "productive_shift": round(float(on_prod.mean()), 3),
                     "damage_per_delta": round(damage, 3),
                     "neurons": neurons.tolist(),
                     "dense_nonzeros": int((dense != 0).sum())})
        solutions[l] = v
        if progress:
            progress(idx + 1, len(layers))
    rows.sort(key=lambda r: (-r["hit_rate"], r["damage_per_delta"]))
    return rows, solutions


def apply_edit(model, u: Dict[int, np.ndarray], solutions: Dict[int, np.ndarray],
               layers: List[int]) -> dict:
    """W += u v^T on each layer's down_proj, in place. Returns an undo
    payload: {"layers": {l: W0}} (float32 copies of original weights)."""
    undo = {"layers": {}}
    for l in layers:
        mod = _layer_down_proj(decoder_layers(model)[l])
        W0 = mod.weight.detach().clone()
        undo["layers"][l] = W0
        u_t = torch.tensor(u[l], dtype=torch.float64, device=W0.device)
        v_t = torch.tensor(solutions[l], dtype=torch.float64, device=W0.device)
        with torch.no_grad():
            mod.weight.copy_(
                (W0.double() + torch.outer(u_t, v_t)).to(W0.dtype))
        rel = float((mod.weight.float() - W0.float()).norm()
                    / W0.float().norm())
        logger.info(f"L{l}: down_proj edited, relative change {rel:.4f}")
    return undo


def undo_edit(model, undo: dict) -> None:
    for l, W0 in undo["layers"].items():
        mod = _layer_down_proj(decoder_layers(model)[int(l)])
        with torch.no_grad():
            mod.weight.copy_(W0)
