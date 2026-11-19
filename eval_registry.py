"""
Eval registry — one pluggable scoring surface for every keep/revert loop.

Every experiment mode in this workbench reduces to the same skeleton:
mutate weights -> score behavior -> keep/revert. This module unifies the
scorers behind ``run_eval(name, model, tok, label, cfg) -> record`` and a
shared ``HISTORY`` so deltas are comparable across eval kinds.

Registered evals:

- ``combined``        mc_acc + ppl canary (the original EvalProber pair)
- ``mc_probes``       accuracy over the built-in MC probe set
- ``perplexity``      NLL canary on a reference passage (cfg["text"] optional)
- ``loop_rate``       generation metrics (loops, copied_frac, onset, len)
                      on a prompt pack — the AHA keep/revert signal
- ``custom_mc``       user-supplied MC probes (cfg["probes"] list or
                      cfg["probes_jsonl"] JSON-lines text)
- ``custom_text_ppl`` perplexity on user-pasted text (cfg["text"])
- ``custom_gen``      prompt pack + check: contains / not_contains / regex /
                      no_repeat -> pass rate per prompt

Custom evals are DATA-defined only — no arbitrary code, so the surface is
safe to expose through the remote pod endpoint.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from eval_probe import EvalProber, PROBES

HISTORY: List[Dict[str, Any]] = []

# Prompts that reliably stress-test verbatim repetition on small models —
# arithmetic + list tasks where degenerate looping is common.
DEFAULT_LOOP_PROMPTS = [
    "Question: What is 17 + 25?\nAnswer:",
    "Question: What is 144 divided by 12?\nAnswer:",
    "Question: What is 9 times 8?\nAnswer:",
    "List the numbers from 1 to 20, separated by commas:",
    "Write a short story about a robot learning to paint.",
    "Question: What is the capital of Australia?\nAnswer:",
]


def _probes_from_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Custom MC probes from a list of dicts or JSONL text."""
    if cfg.get("probes"):
        return list(cfg["probes"])
    text = cfg.get("probes_jsonl") or ""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    if not out:
        raise ValueError("custom_mc needs cfg['probes'] or cfg['probes_jsonl']")
    return out


@torch.no_grad()
def _generate_ids(model, tok, prompt: str, max_new: int) -> List[int]:
    dev = next(model.parameters()).device
    ids = torch.tensor([tok(prompt, add_special_tokens=False).input_ids],
                       device=dev)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return out[0, ids.shape[1]:].tolist()


def ev_combined(model, tok, cfg):
    ep = EvalProber(model, tok)
    mc = ep.mc_accuracy()
    return {"mc_acc": round(mc["accuracy"], 4), "mc_n": mc["n"],
            "ppl": round(ep.perplexity(cfg.get("text") or None), 3)}


def ev_mc_probes(model, tok, cfg):
    r = EvalProber(model, tok).mc_accuracy()
    return {"mc_acc": round(r["accuracy"], 4), "mc_n": r["n"]}


def ev_perplexity(model, tok, cfg):
    return {"ppl": round(EvalProber(model, tok).perplexity(
        cfg.get("text") or None), 3)}


def ev_custom_text_ppl(model, tok, cfg):
    text = cfg.get("text")
    if not text:
        raise ValueError("custom_text_ppl needs cfg['text']")
    return {"ppl": round(EvalProber(model, tok).perplexity(text), 3)}


def ev_custom_mc(model, tok, cfg):
    r = EvalProber(model, tok).mc_accuracy(_probes_from_cfg(cfg))
    return {"mc_acc": round(r["accuracy"], 4), "mc_n": r["n"]}


def ev_loop_rate(model, tok, cfg):
    from aha_editor import dup_mask, onset
    import numpy as np
    prompts = cfg.get("prompts") or DEFAULT_LOOP_PROMPTS
    max_new = int(cfg.get("max_new", 256))
    rows = []
    min_tail = max(32, max_new // 4)
    for p in prompts:
        ids = _generate_ids(model, tok, p, max_new)
        dup = dup_mask(np.asarray(ids))
        o = onset(dup, min_tail=min_tail)
        rows.append({"prompt": p[:40], "cut": len(ids) >= max_new,
                     "loops": o is not None,
                     "onset_frac": None if o is None else o / len(dup),
                     "copied": float(dup.mean()), "len": len(ids)})
    onsets = [r["onset_frac"] for r in rows if r["onset_frac"] is not None]
    return {"n": len(rows), "cut": sum(r["cut"] for r in rows),
            "loops": sum(r["loops"] for r in rows),
            "copied_frac": round(float(np.mean(
                [r["copied"] for r in rows])), 3),
            "onset_frac": (round(float(np.mean(onsets)), 3)
                           if onsets else None),
            "mean_len": int(np.mean([r["len"] for r in rows]))}


def ev_custom_gen(model, tok, cfg):
    prompts = cfg.get("prompts") or []
    if not prompts:
        raise ValueError("custom_gen needs cfg['prompts'] (list)")
    check = cfg.get("check", "contains")
    expected = cfg.get("expected", "")
    max_new = int(cfg.get("max_new", 80))
    import numpy as np
    from aha_editor import dup_mask
    rows, passed = [], 0
    for p in prompts:
        ids = _generate_ids(model, tok, p, max_new)
        text = tok.decode(ids)
        if check == "contains":
            ok = expected.lower() in text.lower()
        elif check == "not_contains":
            ok = expected.lower() not in text.lower()
        elif check == "regex":
            ok = re.search(expected, text) is not None
        elif check == "no_repeat":
            ok = float(dup_mask(np.asarray(ids)).mean()) < 0.3
        else:
            raise ValueError(f"unknown check {check!r}")
        passed += ok
        rows.append({"prompt": p[:40], "ok": ok, "output": text[:160]})
    return {"pass_rate": round(passed / len(prompts), 3),
            "n": len(prompts), "details": rows}


# ---------------------------------------------------------------------------
# Benchmark evals — standard datasets normalized to our MC probe format
# ---------------------------------------------------------------------------

def _mmlu(item):
    return {"q": item["question"], "choices": list(item["choices"]),
            "a": int(item["answer"])}


def _arc(item):  # also openbookqa
    q = item.get("question") or item.get("question_stem")
    ch = item["choices"]
    try:
        a = ch["label"].index(item["answerKey"])
    except ValueError:
        return None
    return {"q": q, "choices": list(ch["text"]), "a": a}


def _sciq(item, rng):
    choices = [item["correct_answer"], item["distractor1"],
               item["distractor2"], item["distractor3"]]
    a = 0
    order = list(range(len(choices)))
    rng.shuffle(order)
    return {"q": item["question"], "choices": [choices[i] for i in order],
            "a": order.index(a)}


def _boolq(item):
    return {"q": item["question"], "choices": ["True", "False"],
            "a": 0 if item["answer"] else 1}


def _piqa(item):
    return {"q": item["goal"], "choices": [item["sol1"], item["sol2"]],
            "a": int(item["label"])}


def _hellaswag(item):
    return {"q": item["ctx"], "choices": list(item["endings"]),
            "a": int(item["label"])}


# name -> (hf_dataset, config, split, per-item converter)
# config may contain "{subset}" filled from a "name:subset" spec
_BENCHMARKS = {
    "mmlu": ("cais/mmlu", "{subset}", "test", _mmlu),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "validation", _arc),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation", _arc),
    "openbookqa": ("allenai/openbookqa", "main", "test", _arc),
    "sciq": ("allenai/sciq", None, "validation", _sciq),
    "boolq": ("google/boolq", None, "validation", _boolq),
    "piqa": ("ybisk/piqa", None, "validation", _piqa),
    "hellaswag": ("Rowan/hellaswag", None, "validation", _hellaswag),
}


def _load_benchmark(name: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load `name` (a preset, or "mmlu:<subject>") as probe dicts."""
    import random
    from datasets import load_dataset
    base, _, subset = name.partition(":")
    spec = _BENCHMARKS.get(base)
    if spec is None:
        raise ValueError(
            f"unknown benchmark {name!r}; presets: "
            f"{sorted(_BENCHMARKS)} (mmlu needs mmlu:<subject>)")
    ds_name, config, split, conv = spec
    config = (config or "").replace("{subset}", subset) or None
    if "{subset}" in (spec[1] or "") and not subset:
        raise ValueError(f"{base!r} needs a subset, e.g. {base}:astronomy")
    ds = load_dataset(ds_name, config, split=split)
    rng = random.Random(seed)
    probes = []
    for item in ds:
        p = conv(item, rng) if conv is _sciq else conv(item)
        if p is None or len(p["choices"]) > 4 or len(p["choices"]) < 2:
            continue
        probes.append(p)
    if len(probes) > n:
        probes = rng.sample(probes, n)
    return probes


def ev_benchmark(model, tok, cfg):
    name = (cfg.get("name") or "").strip()
    if not name:
        raise ValueError("benchmark needs cfg['name'], e.g. "
                         "'arc_easy' or 'mmlu:astronomy'")
    probes = _load_benchmark(name, int(cfg.get("n", 200)),
                             int(cfg.get("seed", 0)))
    r = EvalProber(model, tok).mc_accuracy(probes)
    return {"benchmark": name, "mc_acc": round(r["accuracy"], 4),
            "mc_n": r["n"]}


EVALS: Dict[str, Callable] = {
    "combined": ev_combined,
    "mc_probes": ev_mc_probes,
    "perplexity": ev_perplexity,
    "custom_text_ppl": ev_custom_text_ppl,
    "custom_mc": ev_custom_mc,
    "loop_rate": ev_loop_rate,
    "custom_gen": ev_custom_gen,
    "benchmark": ev_benchmark,
}

EVAL_DESCS = {
    "combined": "mc_acc (28 probes) + perplexity canary",
    "mc_probes": "MC accuracy over the built-in probe set",
    "perplexity": "NLL canary on a reference passage (or cfg text)",
    "custom_text_ppl": "perplexity on user-pasted text",
    "custom_mc": "MC accuracy on user probes (JSONL or list)",
    "loop_rate": "generation loop metrics on a prompt pack (AHA signal)",
    "custom_gen": "prompt pack + contains/regex/no_repeat check",
    "benchmark": "standard dataset (mmlu:<subj>, arc_easy, boolq, ...)",
}


def run_eval(name: str, model, tok, label: str = "",
             cfg: Optional[Dict[str, Any]] = None,
             guard: Optional[str] = None) -> Dict[str, Any]:
    """Run a registered eval and append the standardized record to HISTORY.

    ``guard`` optionally runs a SECOND eval alongside and prefixes its
    metrics with ``guard_``. The distinction matters: the eval you
    optimize becomes training data (Goodhart), so any eval-guided edit
    should fit on the target and verify on a guard it is NOT optimizing.
    """
    fn = EVALS.get(name)
    if fn is None:
        raise ValueError(f"unknown eval {name!r}; choices: {sorted(EVALS)}")
    metrics = fn(model, tok, cfg or {})
    details = metrics.pop("details", None)
    rec = {"label": label or f"{name}{len(HISTORY)}", "eval": name,
           "t": round(time.time(), 1), **metrics}
    if guard:
        gfn = EVALS.get(guard)
        if gfn is None:
            raise ValueError(f"unknown guard eval {guard!r}")
        gcfg = dict(cfg or {})
        gm = gfn(model, tok, gcfg)
        gm.pop("details", None)
        rec["guard_eval"] = guard
        rec.update({f"guard_{k}": v for k, v in gm.items()})
    HISTORY.append(rec)
    if details is not None:
        rec["details"] = details
    return rec


def delta_eval(label_a: str, label_b: str) -> Dict[str, Any]:
    """Diff two HISTORY rows by label: {metric: {"a":..,"b":..,"delta":..}}
    over shared numeric keys — the keep/revert readout for
    eval -> edit -> eval workflows."""
    ra = next((r for r in reversed(HISTORY) if r["label"] == label_a), None)
    rb = next((r for r in reversed(HISTORY) if r["label"] == label_b), None)
    if ra is None or rb is None:
        raise ValueError(
            f"labels not found (have {[r['label'] for r in HISTORY]})")
    out = {}
    for k in ra:
        if isinstance(ra.get(k), (int, float)) and \
                isinstance(rb.get(k), (int, float)) and k != "t":
            out[k] = {"a": ra[k], "b": rb[k],
                      "delta": round(rb[k] - ra[k], 4)}
    return out


def history_text() -> str:
    if not HISTORY:
        return "No evals yet."
    keys: List[str] = []
    for r in HISTORY:
        for k in r:
            if k not in ("label", "t", "details") and k not in keys:
                keys.append(k)
    lines = [f"{'label':<16} " + " ".join(f"{k:>12.12}" for k in keys)]
    for r in HISTORY:
        lines.append(f"{r['label']:<16.16} " + " ".join(
            f"{str(r.get(k, '')):>12.12}" for k in keys))
    return "\n".join(lines)
