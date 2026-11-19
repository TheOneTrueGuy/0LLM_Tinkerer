"""Gradient fine-tune baseline — the FT arm of the WSP-vs-training comparison.

Trains on the TRAIN split of a benchmark preset (the eval registry scores
validation/test, so train/test separation is automatic) and saves a merged
checkpoint + tokenizer that works as a --ref for sweep_eval.py or /load.

Usage on pod:
    python -u train.py --base Qwen/Qwen2.5-0.5B --dataset arc_easy \
        --out qwen_arc_lora --epochs 2
"""

import argparse
import json
import random
import time

import torch

import eval_registry
from eval_probe import LETTERS


def fmt_train(p):
    """Match eval_probe._mc_prompt exactly, plus the gold letter."""
    lines = [f"Question: {p['q']}"]
    lines += [f"{LETTERS[i]}) {c}" for i, c in enumerate(p["choices"])]
    lines.append("Answer:")
    return "\n".join(lines) + f" {LETTERS[p['a']]}"


def load_train_probes(name, n, seed):
    """Benchmark preset -> probe dicts from the TRAIN split."""
    from datasets import load_dataset
    base, _, subset = name.partition(":")
    spec = eval_registry._BENCHMARKS.get(base)
    if spec is None:
        raise ValueError(
            f"unknown dataset {name!r}; presets: "
            f"{sorted(eval_registry._BENCHMARKS)}")
    ds_name, config, _, conv = spec
    config = (config or "").replace("{subset}", subset) or None
    ds = load_dataset(ds_name, config, split="train")
    rng = random.Random(seed)
    probes = []
    for item in ds:
        p = conv(item, rng) if conv is eval_registry._sciq else conv(item)
        if p is None or not 2 <= len(p["choices"]) <= 4:
            continue
        probes.append(p)
    rng.shuffle(probes)
    return probes[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", default="arc_easy",
                    help="benchmark preset; trained on its TRAIN split")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--full", action="store_true",
                    help="full fine-tune instead of LoRA")
    ap.add_argument("--out", required=True, help="output dir for merged ckpt")
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16).cuda()
    print(f"=== load: {args.base} ({time.time()-t0:.0f}s) ===", flush=True)

    probes = load_train_probes(args.dataset, args.n, args.seed)
    texts = [fmt_train(p) for p in probes]
    print(f"=== dataset: {args.dataset} train split, {len(texts)} "
          f"examples ===", flush=True)

    enc = tok(texts, truncation=True, max_length=args.max_len,
              padding="max_length", return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    enc["labels"][enc["attention_mask"] == 0] = -100

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            return {k: v[i] for k, v in enc.items()}

    if not args.full:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                            "c_attn", "c_proj", "c_fc"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=f"{args.out}_trainer", num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        logging_steps=25, save_strategy="no", fp16=True,
        report_to=[], seed=args.seed)

    print("=== train ===", flush=True)
    trainer = Trainer(model=model, args=targs, train_dataset=DS())
    trainer.train()

    print("=== save: merged checkpoint ===", flush=True)
    if not args.full:
        model = model.merge_and_unload()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    res = {"out": args.out, "dataset": args.dataset, "n": len(texts),
           "epochs": args.epochs, "lora": not args.full,
           "lora_r": None if args.full else args.lora_r,
           "seconds": round(time.time() - t0, 1)}
    print(f"RESULT: {json.dumps(res)}", flush=True)


if __name__ == "__main__":
    main()
