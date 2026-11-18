# iWSP harness (vendored)

Files in this directory are verbatim copies from
`csinva/weight-space-programming-autoresearch` (`experiments/evolve/`, MIT
License, (c) 2026 Chandan Singh and Yash Mehta). They are the FIXED
evaluator side of the autoresearch loop:

- `program.md` — instructions the agent reads inside every run folder
  (the `your task name is \`...\`` placeholder is filled per run)
- `interpretable_transformer.py` — the file the agent edits; weights are
  written in closed form in `write_weights()`, never trained
- `src/task.py` — task registry + tokenization for all tasks
- `src/eval.py` — greedy-decode evaluator, appends to
  `results/overall_results.csv`
- `results/overall_results.csv` — CSV header baseline

The OpenRouter driver (`../iwsp_driver.py`) stamps a copy of this tree
into each run folder under `iwsp_runs/` — no symlinks, no `uv`, plain
`python interpretable_transformer.py --task <task>`.
