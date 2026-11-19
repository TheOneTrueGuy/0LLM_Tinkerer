"""
Lambda sweep — measure a standard eval along a weight-space direction.

Loads a base model and an edited reference on the pod, computes the delta
dW = W_edited - W_base, then for each lambda evaluates loop_rate on
W_lam = W_base + lam*dW. A random direction of the same norm runs as the
control arm. If loop_rate(lambda) is monotonic-ish for the real delta but
flat/degrading for the random one, the eval carries a finite-difference
gradient — the primitive for eval-guided WSP (no backprop).

Usage on pod:
    python sweep_eval.py --base Qwen/Qwen2.5-0.5B --ref aha_signfix_model \
        --eval loop_rate --prompts aha --max-new 2048
"""

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import eval_registry


def apply_delta(model, base_sd, delta, lam):
    """In-place: W = base + lam*delta (delta holds only nonzero tensors)."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            d = delta.get(n)
            p.copy_(base_sd[n] if d is None else base_sd[n] + lam * d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ref", required=True, help="edited checkpoint dir")
    ap.add_argument("--eval", default="loop_rate", choices=list(eval_registry.EVALS))
    ap.add_argument("--prompts", default="default", choices=["default", "aha"])
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--eval-cfg", default="{}",
                    help="extra eval cfg as JSON, e.g. '{\"name\":\"arc_easy\",\"n\":200}'")
    ap.add_argument("--guard", default=None, choices=list(eval_registry.EVALS),
                    help="held-out eval run alongside each point (guard_* metrics)")
    ap.add_argument("--guard-cfg", default="{}")
    ap.add_argument("--no-random", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    print(f"loading {args.base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16).cuda().eval()
    ref = AutoModelForCausalLM.from_pretrained(
        args.ref, torch_dtype=torch.float16).cuda().eval()

    base_sd = {n: p.detach().clone()
               for n, p in model.named_parameters()}
    delta, n_delta = {}, 0
    for (n, p), (_, r) in zip(model.named_parameters(),
                              ref.named_parameters()):
        d = (r.detach() - p.detach())
        if d.abs().max() > 0:
            delta[n] = d
            n_delta += 1
    print(f"delta tensors: {n_delta}, total norm "
          f"{torch.sqrt(sum(d.pow(2).sum() for d in delta.values())):.4f}",
          flush=True)

    # random control direction — same per-tensor norm as the real delta
    torch.manual_seed(0)
    rand_delta = {}
    for n, d in delta.items():
        r = torch.randn_like(d)
        rand_delta[n] = r * (d.norm() / r.norm().clamp_min(1e-12))

    if args.prompts == "aha":
        import aha_test
        probs = getattr(aha_test, "PROBLEMS_SMOKE", None) or \
            getattr(aha_test, "PROBLEMS", [])
        prompts = [p["prompt"] if isinstance(p, dict) else p for p in probs]
    else:
        prompts = eval_registry.DEFAULT_LOOP_PROMPTS
    cfg = {"prompts": prompts, "max_new": args.max_new}
    cfg.update(json.loads(args.eval_cfg))
    gcfg = dict(cfg)
    gcfg.update(json.loads(args.guard_cfg))

    results = {"base": args.base, "ref": args.ref, "eval": args.eval,
               "guard": args.guard, "n_prompts": len(prompts), "arms": {}}
    arms = [("delta", delta)] + ([] if args.no_random
                                else [("random", rand_delta)])
    for arm_name, arm_delta in arms:
        rows = []
        for lam in args.lams:
            apply_delta(model, base_sd, arm_delta, lam)
            t0 = time.time()
            m = eval_registry.EVALS[args.eval](model, tok, cfg)
            if args.guard:
                gm = eval_registry.EVALS[args.guard](model, tok, gcfg)
                gm.pop("details", None)
                m.update({f"guard_{k}": v for k, v in gm.items()})
            m["lam"], m["seconds"] = lam, round(time.time() - t0, 1)
            rows.append(m)
            print(f"{arm_name} lam={lam:+.2f}: {m}", flush=True)
        results["arms"][arm_name] = rows

    apply_delta(model, base_sd, delta, 0.0)  # restore base
    print("RESULT: " + json.dumps(results, default=str))


if __name__ == "__main__":
    main()
