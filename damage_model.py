"""Shuffle-damage a model: permute p% of values within each tensor.

Preserves every tensor's exact marginal weight distribution while
destroying its structure — the cleanest 'randomized' damage model.
The damage curve (Qwen2.5-0.5B, measured): ~0.5% shuffled = degraded
but alive (acc .60, ppl 6); 1% = collapsed. Use --p 0.005 for the
signal-rich repairable regime.

Usage on pod:
    python -u damage_model.py --base Qwen/Qwen2.5-0.5B --out damaged_qwen \
        --p 0.005 --eval
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def shuffle_damage(model, p, seed=0):
    """Permute a fraction p of values within each float tensor, in place."""
    g = torch.Generator().manual_seed(seed)
    n_hit = 0
    for _, t in model.state_dict().items():
        if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue
        flat = t.view(-1)
        n = int(flat.numel() * p)
        if n < 2:
            continue
        idx = torch.randperm(flat.numel(), generator=g)[:n]
        flat[idx] = flat[idx][torch.randperm(len(idx), generator=g)]
        n_hit += n
    return n_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=float, default=0.005,
                    help="fraction of each tensor's values to permute")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true",
                    help="sanity eval (arc_easy n=50 + ppl) after damage")
    a = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.base)
    m = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype="auto")
    m.eval()

    n_hit = shuffle_damage(m, a.p, a.seed)
    print(f"=== damaged {n_hit/1e6:.1f}M weights "
          f"({a.p:.3%} per tensor, seed={a.seed}) ===", flush=True)

    metrics = {}
    if a.eval:
        import eval_registry
        r = eval_registry.run_eval(
            "benchmark", m, tok, label="damaged",
            cfg={"name": "arc_easy", "n": 50, "seed": 0},
            guard="perplexity")
        metrics = {"damaged_acc": r["mc_acc"], "damaged_ppl": r["guard_ppl"]}
        print(f"=== sanity: acc={r['mc_acc']:.3f} "
              f"ppl={r['guard_ppl']:.1f} ===", flush=True)

    m.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("=== save: damaged checkpoint ===", flush=True)
    print("RESULT: " + json.dumps({
        "out": a.out, "p": a.p, "seed": a.seed,
        "n_weights_hit": n_hit, "seconds": round(time.time() - t0, 1),
        **metrics}), flush=True)


if __name__ == "__main__":
    main()
