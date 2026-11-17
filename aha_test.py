"""End-to-end AHA-style de-looping driver for LLM Tinkerer.

  python aha_test.py --model Qwen/Qwen3-0.6B [--max-new 4096] [--cont 8]
                     [--problems N] [--save edited_aha] [--self-test]

Flow: sample traces -> loop stats -> matched pairs -> direction u ->
collect activations -> LP at every layer -> rank-one write on top-2 ->
re-sample and compare loop rate / completion rate -> optional save.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import aha_editor as aha

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("aha_test")


def self_test():
    """Verify the detector on synthetic data — no model needed."""
    rng = np.random.default_rng(0)
    # productive: random tokens, no loops
    ids = rng.integers(0, 50000, 2000)
    assert aha.onset(aha.dup_mask(ids)) is None, "false onset on random ids"
    # looping: random prefix, then one 40-token block repeated forever
    head = rng.integers(0, 50000, 300)
    block = rng.integers(0, 50000, 40)
    ids = np.concatenate([head, np.tile(block, 40)])
    o = aha.onset(aha.dup_mask(ids))
    assert o is not None and abs(o - 340) < 60, f"bad onset {o}"
    lab = aha.label(ids, cut=True)
    assert lab.onset == o and len(lab.P) > 0 and len(lab.R_during) > 0
    # undetected if trace too short
    ids = np.concatenate([head, np.tile(block, 8)])
    assert aha.onset(aha.dup_mask(ids)) is None, "onset below min_tail"
    # LP smoke: separable synthetic h, v must split the two classes
    d = 64
    h_prod = rng.normal(0, 1, (200, d)).astype(np.float32)
    h_loop = h_prod + rng.normal(3, 1, (200, d)).astype(np.float32)
    v = aha.pdhg(h_loop, h_prod, delta=0.5, eps=0.5, iters=4000)
    shifts_l, shifts_p = h_loop @ v, h_prod @ v
    print(f"LP smoke: loop push mean {shifts_l.mean():.2f} "
          f"(want <= -0.5: {np.mean(shifts_l <= -0.5):.0%} hit), "
          f"prod |shift| p90 {np.percentile(np.abs(shifts_p), 90):.2f}")
    print("self-test OK")


def loop_stats(traces):
    n_cut = sum(t["cut"] for t in traces)
    n_loop = sum(aha.onset(aha.dup_mask(t["resp_ids"])) is not None
                 for t in traces if t["cut"])
    return {"traces": len(traces), "cut": n_cut, "loops": n_loop}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--problems", type=int, default=12)
    p.add_argument("--max-new", type=int, default=4096)
    p.add_argument("--cont", type=int, default=8,
                   help="continuations per looping prefix")
    p.add_argument("--cont-max-new", type=int, default=2048)
    p.add_argument("--margin", type=int, default=64,
                   help="tokens before loop onset to cut prefixes")
    p.add_argument("--temp", type=float, default=1.0,
                   help="sampling temperature for continuations")
    p.add_argument("--topk", type=int, default=2, help="layers to edit")
    p.add_argument("--mult", type=float, default=4.0)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--k", type=int, default=8, help="neurons kept per layer")
    p.add_argument("--problems-file", help="json list of problem strings")
    p.add_argument("--pro", action="store_true",
                   help="inverted control: push productive tokens toward "
                        "the loop direction instead of away")
    p.add_argument("--save", help="directory to save the edited model")
    p.add_argument("--out", default="aha_run", help="artifacts directory")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.problems_file:
        problems = json.load(open(args.problems_file))[:args.problems]
    else:
        problems = aha.PROBLEMS_SMOKE[:args.problems]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = model.to(aha.device()).eval()

    t0 = time.time()
    traces_path = out / "traces.json"
    if traces_path.exists():
        traces = json.load(open(traces_path))
        for t in traces:
            t["resp_ids"] = np.array(t["resp_ids"], np.int64)
        logger.info(f"loaded {len(traces)} saved traces")
    else:
        logger.info("=== sampling base traces ===")
        traces = aha.sample_traces(
            model, tok, problems, max_new=args.max_new,
            progress=lambda a, b: logger.info(f"trace {a}/{b}"))
        json.dump([{**t, "resp_ids": t["resp_ids"].tolist()} for t in traces],
                  open(traces_path, "w"))
    stats = loop_stats(traces)
    logger.info(f"base: {stats}")
    json.dump(stats, open(out / "base_stats.json", "w"))

    logger.info("=== matched pairs ===")
    records = aha.make_pairs(model, tok, traces, n_cont=args.cont,
                             max_new=args.cont_max_new,
                             margin=args.margin, temperature=args.temp,
                             progress=lambda a, b: logger.info(f"prefix {a}/{b}"))
    pairs = aha.matched_pairs(records)
    logger.info(f"{len(pairs)} matched pairs from {len(records)} prefixes")
    json.dump(records, open(out / "continuations.json", "w"))

    logger.info("=== collect activations ===")
    layers = list(range(len(aha.decoder_layers(model))))
    collected = aha.collect_activations(
        model, tok, traces, layers,
        progress=lambda a, b: logger.info(f"trace {a}/{b}"))
    np.savez(out / "collected.npz",
             **{f"h{l}": collected["h"][l] for l in layers},
             **{k: collected["meta"][k] for k in collected["meta"]})

    logger.info("=== direction ===")
    if pairs:
        direction = aha.build_direction(
            model, records, min_pairs=min(10, len(pairs)),
            progress=lambda a, b: logger.info(f"pair {a}/{b}"))
        json.dump(direction["separation"], open(out / "separation.json", "w"),
                  indent=1)
        best = sorted(direction["separation"],
                      key=lambda r: -r["cohens_d"])[:5]
        logger.info("best separation: " + ", ".join(
            f"L{r['layer']} d={r['cohens_d']}" for r in best))
        n_pairs = direction["n_pairs"]
    else:
        logger.info("no matched pairs — falling back to unmatched "
                    "mean-difference direction (Barry's method)")
        direction = {"u": aha.build_direction_unmatched(collected),
                     "separation": [], "n_pairs": 0}
        n_pairs = 0

    logger.info("=== LP search ===")
    rows, solutions = aha.search_layers(
        model, direction["u"], collected, eps=args.eps, k=args.k,
        mult=args.mult, pro=args.pro,
        progress=lambda a, b: logger.info(f"layer {a}/{b}"))
    json.dump(rows, open(out / "layers.json", "w"), indent=1)
    logger.info("ranking: " + json.dumps(
        [(r["layer"], r["hit_rate"], r["damage_per_delta"]) for r in rows[:6]]))
    if not rows:
        raise SystemExit("no layer produced a solution — not enough data?")
    top = sorted(r["layer"] for r in rows[:args.topk])
    logger.info(f"editing layers {top}")

    undo = aha.apply_edit(model, direction["u"], solutions, top)
    torch.save({"layers": {str(k): v.cpu() for k, v in undo["layers"].items()}},
               out / "undo.pt")

    logger.info("=== eval: resample traces ===")
    eval_traces = aha.sample_traces(model, tok, problems,
                                    max_new=args.max_new,
                                    progress=lambda a, b: logger.info(f"eval {a}/{b}"))
    estats = loop_stats(eval_traces)
    report = {"model": args.model, "mode": "pro" if args.pro else "de-loop",
              "base": stats, "edited": estats,
              "layers_edited": top, "n_pairs": n_pairs,
              "seconds": round(time.time() - t0)}
    json.dump(report, open(out / "report.json", "w"), indent=1)
    logger.info(f"RESULT: {json.dumps(report)}")

    if args.save:
        model.save_pretrained(args.save)
        tok.save_pretrained(args.save)
        logger.info(f"saved to {args.save}")


if __name__ == "__main__":
    main()
