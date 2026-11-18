"""
Evaluate an autoregressive transformer on a Task.

Usage:
    from src.task import get_task
    from src.eval import evaluate
    task = get_task("add5")
    acc, details = evaluate(model, task, n_samples=200, seed=0)
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import torch

from .task import Task, Example


@dataclass
class EvalDetail:
    prompt: str
    target: str
    prediction: str
    correct: bool


@torch.no_grad()
def autoregressive_generate(model, prompt_ids: torch.Tensor, n_new_tokens: int) -> torch.Tensor:
    """Greedy autoregressive sampling.

    Args:
        model: callable that takes (B, T) ids and returns (B, T, vocab_size) logits.
        prompt_ids: (B, T_prompt) integer tensor on the model's device.
        n_new_tokens: how many tokens to generate.

    Returns:
        (B, n_new_tokens) tensor of generated token ids (not including the prompt).
    """
    model.eval()
    tokens = prompt_ids
    generated = []
    for _ in range(n_new_tokens):
        logits = model(tokens)              # (B, T, V)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        generated.append(next_id)
        tokens = torch.cat([tokens, next_id], dim=1)
    return torch.cat(generated, dim=1)


def evaluate(
    model,
    task: Task,
    n_samples: int = 200,
    seed: int = 0,
    batch_size: int = 64,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[float, list[EvalDetail]]:
    """Run the model on `n_samples` task examples and return accuracy.

    Returns:
        accuracy: fraction of examples where the predicted string matches the target.
        details: list of EvalDetail with per-example outcomes.
    """
    examples: list[Example] = task.generate_examples(n_samples, seed=seed)

    if device is None:
        device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    device = torch.device(device)

    details: list[EvalDetail] = []
    n_correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prompt_ids = torch.tensor(
            [task.encode(e.prompt) for e in batch], dtype=torch.long, device=device,
        )
        out_ids = autoregressive_generate(model, prompt_ids, task.answer_len).cpu().tolist()
        for example, ids in zip(batch, out_ids):
            pred = task.decode(ids)
            correct = task.is_correct(pred, example.answer)
            details.append(EvalDetail(
                prompt=example.prompt, target=example.answer,
                prediction=pred, correct=correct,
            ))
            n_correct += int(correct)
            if verbose:
                marker = "OK " if correct else "   "
                print(f"  {marker}{example.prompt}{pred}  (target {example.answer})")
    accuracy = n_correct / max(1, len(examples))
    return accuracy, details


def plot_accuracy_over_iterations(results_dir: str) -> None:
    """Read overall_results.csv in CSV order and plot accuracy + running max."""
    csv_path = os.path.join(results_dir, "overall_results.csv")
    if not os.path.exists(csv_path):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple[str, float]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                acc = float(row["accuracy"])
            except (TypeError, ValueError):
                continue
            rows.append((row.get("model_shorthand_name", ""), acc))
    if not rows:
        return

    iters = list(range(1, len(rows) + 1))
    accs = [a for _, a in rows]
    running_max: list[float] = []
    best = float("-inf")
    for a in accs:
        best = max(best, a)
        running_max.append(best)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, accs, marker="o", linestyle="-", color="steelblue", label="accuracy")
    ax.plot(iters, running_max, drawstyle="steps-post", color="crimson",
            linewidth=2, label="running max")
    ax.set_xlabel("iteration")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Accuracy over iterations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    if len(iters) <= 30:
        ax.set_xticks(iters)
    fig.tight_layout()
    out_path = os.path.join(results_dir, "accuracy_over_iterations.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out_path}")