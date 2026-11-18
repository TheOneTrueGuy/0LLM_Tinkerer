# autoresearch — interpretable transformers

This is an experiment to have a coding agent autonomously research how to hand-write the weights of a small transformer (ideally with some human-understandable structure) so that it solves a task (**your task name is `nli-snli`**).

The model is NEVER trained. The agent must write the weights directly into `interpretable_transformer.py`.

## Setup

The work happens inside a fresh **run folder**. Each run is an isolated copy of `experiments/evolve/` under `experiments/runs/manual/<tag>/`.

1. **Pick a unique run tag** based on today's date and a counter, e.g. `<your_task_name>-may15-run1`.
2. **Create the run folder**: `uv run experiments/evolve/setup_run.py <tag>`. This creates `experiments/runs/manual/<tag>/` containing:

- `program.md`, `src/` — **symlinks** back to `experiments/evolve/` (do not modify these via the symlinks)
- `interpretable_transformer.py` — **local copy**, this is the file you edit
- `results/` — stores output of your runs
- `interpretable_transformers_lib/` — empty, for snapshots

1. **`cd` into the run folder** and stay there for the rest of the session: `cd experiments/runs/manual/<tag>`. Do not read any of the other folders in the runs directory, only your own run folder.
2. **Read the in-scope files**: `interpretable_transformer.py`, `src/task.py`, `src/eval.py`, and `results/overall_results.csv` (only one baseline row at first).

## Experimentation

Each experiment runs on a single GPU. You launch it (from inside the run folder) simply as `uv run interpretable_transformer.py --task <your_task_name>`.

**What you CAN do:**

- Edit `interpretable_transformer.py` in the run folder. Everything in it is fair game (except the evaluation code in the **main** block at the bottom):
  - The `SimpleTransformer`, `Block`, `CausalSelfAttention`, `MLP` architectures.
  - The `write_weights(model, task)` function — set any parameter tensors directly. Hardcoded constants, NumPy arrays, hand-built attention circuits, lookup tables — anything goes, as long as you do not train.
  - The architecture's hyperparameters (depth, width, heads, ff size).
  - The `model_shorthand_name` and `model_description` strings.
- Save snapshots of `interpretable_transformer.py` under `interpretable_transformers_lib/` after each attempt.

**What you CANNOT do:**

- Run any kind of training, gradient descent, optimizer step, fitting loop, or backprop. The model parameters must be set in closed form.
- Use python functions / existing libraries to compute the task output. The model must generate the answer autoregressively using a transformer architecture with no external tools.
- Download or read existing weights from any pre-trained models.
- Modify anything reached through a symlink: anything in `src/` or `program.md`.

**The goal is simple: maximize `accuracy`**: the fraction of held-out examples where the autoregressively-generated answer string exactly matches the target (or is within some numerical tolerance).

**Interpretability criterion**: All else being equal, more interpretable is better. A small improvement that adds ugly complexity is not worth it. Number of model parameters can help serve as a proxy for interpretability, but the real criterion is human-understandable structure in the weights.

## Output format

Once the script finishes it prints a summary like this:

```

---
task:          your_task_name
accuracy:      0.0050  (1/200)
total_seconds: 1.3s

```

It also updates `results/overall_results.csv` which has the following format:

```

task,accuracy,status,model_name,n_params,description

```

1. task name (e.g. `addition-five-digits`)
2. accuracy from the script output — empty for crashes
3. status: `success` or `crash`
4. shorthand unique name of the model attempt
5. total number of model parameters (`sum(p.numel() for p in model.parameters())`)
6. brief text description of what this attempt tried

Always log to this file after each experiment.

## The experiment loop

You are always inside the run folder (`experiments/runs/manual/<tag>/`).

LOOP FOREVER:

1. Edit `interpretable_transformer.py` with one experimental idea. Update `model_shorthand_name` (must be unique within this run) and `model_description` to reflect it.
2. Run the experiment: `uv run interpretable_transformer.py > run.log 2>&1`
3. Read results: `tail -n 10 run.log` and `grep <model_shorthand_name> results/overall_results.csv`
4. If the run crashed, check `tail -n 50 run.log` for the stack trace and attempt a fix.
5. Update the row in `results/overall_results.csv` with the appropriate status (`success`, `crash`).
6. Save a snapshot of `interpretable_transformer.py` as `interpretable_transformers_lib/<model_shorthand_name>.py`.

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human if you should continue. Run until manually stopped. Even if you hit 100% accuracy, keep going to see if you can find a simpler solution. Definitely do not stop after less than 20 iterations.

**Ideas to try** (not exhaustive — be creative):

- Read mech-interp papers on grokking / circuits:
  - Nanda et al. 2023 "Progress measures for grokking via mechanistic interpretability" (<https://arxiv.org/abs/2301.05217>)
  - Quirke & Barez 2023 "Understanding addition in transformers" (<https://arxiv.org/abs/2310.13121>)
- Use position embeddings to "select" the right digit pair to add at each output step.
- Carry propagation via a second attention layer.
- Use the MLP layers as lookup tables (one-hot in, one-hot out).
- Reverse-engineer the output digits (least significant first) to make carry easier.
- Do not just generate I/O pairs and memorize them — the goal is to build a circuit that generalizes across all 5-digit pairs.

Make sure the model evaluates very quickly. Default eval is 200 examples on GPU. BE CREATIVE.
