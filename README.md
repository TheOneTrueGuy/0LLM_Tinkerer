# LLM Tinkerer

A workbench for poking at language model internals. Load any HuggingFace causal
LM, inspect its weights and activations, mutate it in controlled ways, steer its
behavior at inference time, and compare the results — locally on CPU or on a
RunPod GPU pod. Built with Gradio and PyTorch/transformers.

## What it does

**Generation & inspection**

- Load and generate with any HF causal LM (GPT-2, Llama, OPT, Bloom, ...)
  with adjustable temperature/length; outputs logged to `outputs.jsonl`
- Interactive architecture graph: click modules, view their parameters,
  batch set/scale/random edits
- Layer inspection: weight heatmaps for any module (Linear, Conv1D,
  Embedding, LayerNorm) and per-token attention patterns per block

**Weight mutation**

- Weight inversion: invert all or part of the weights (attention only, MLP
  only, layer ranges, custom rules), with paired original-vs-modified
  generation and a live view toggle
- Weight editing: set/scale/randomize any named parameter, or paste an
  exact matrix
- Layer surgery: insert, move, remove, jitter, or graft decoder blocks and
  sublayers (whole `mlp`/`attn` or single tensors), including identity and
  noisy-copy inits

**Weight Space** (task-vector arithmetic + latent basis — early Weight Space
Modelling)

- Merge: load a second checkpoint of the same architecture and apply
  `W + λ·(R−W)` (toward it), `W − λ·(R−W)` (away), or a soup average, with
  an optional param-name filter; mismatched tensors are skipped
- Check compat: compares config dims (~1KB fetch, no weights) and predicts
  merge coverage before downloading
- λ sweep: non-destructive degradation curve — restores baseline, merges at
  each λ, generates, and restores at the end
- Latent basis: collect weight deltas vs a baseline (any weight-editing op,
  or auto-generated structured edits), run PCA, then drive the model with
  **PC sliders** — each slider sets an absolute mix `W = anchor + Σ sᵢ·PCᵢ`
- All merges/mixes are undoable through the same backup→apply→restore path

**Fact Writer** (closed-form weight writes, MEMIT-style — no gradients)

- Teach the model a fact: give several phrasings of a question plus the
  answer; a linear solve updates `down_proj`/`c_proj` in the top ~5
  decoder layers while protecting ~40 general Q&As, the question tokens,
  contrast prompts, and previously written facts
- Works on GPT-2 Conv1D and Llama-family Linear projections alike;
  edits toggle for A/B compare and are individually undoable
- VLM variant (`vlm_fact_editor.py`): writes visual associations (e.g.
  "this breed → label") through the language side only — vision encoder
  untouched

**Eval Registry** (`eval_registry.py` — every experiment loop is
*mutate → score → keep/revert*, so all scorers live behind one registry)

- Built-ins: `combined` (28 MC probes + perplexity canary),
  `mc_probes`, `perplexity`, `loop_rate` (generation-loop metrics:
  copied fraction, onset, cut count on a prompt pack)
- Custom evals are data-defined — probe JSONL, pasted text, or a prompt
  pack with contains/regex/no-repeat checks. No arbitrary code, so the
  same evals run safely on remote pods
- Benchmarks: `benchmark` eval loads HF datasets (`arc_easy`,
  `arc_challenge`, `boolq`, `piqa`, `hellaswag`, `sciq`, `openbookqa`,
  `mmlu:<subject>`), normalized onto the MC scorer
- **Guard eval**: an eval used to choose edits is a training signal, not
  a held-out measurement — so any eval can run a *guard* eval alongside
  (recorded as `guard_*` metrics), and **Compare A→B** diffs two labeled
  runs including guard metrics. Fit on the target, verify on the guard
- Same registry backs local runs, pod `/eval`, and the agent's
  `evaluate`/`eval_delta` tools

**WSP Agent** (weight-space programming — an LLM driving the bench)

- OpenRouter-hosted agent model with 19 tools: inspect, generate,
  evaluate (any registered eval + optional guard), eval-diff, sculpt,
  fact-write, merge, and latent-basis ops
- Local or RunPod target: the agent can drive the pod's model over HTTP
- Agent runs a full loop: eval → edit → eval → keep or revert, with the
  streamed tool-call trace visible; repeated identical failing calls are
  detected and nudged

**Persistence**

- Session tab: save named sessions (model, params, sculpt op specs) to
  `sessions/*.json`, reload the model and replay ops on restart; `last`
  autosaves on every load and sculpt apply/undo

**Neuron Explorer** (steering-first, after
[lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer))

- Steer: push one MLP neuron up/down at inference (`−/0/+` rollouts)
- Sweep: neurons × doses × prompts with baselines and random-direction
  controls, written incrementally to `neuron_runs/*.jsonl`
- Browse: ranks neurons by how far their steered vocabulary shifted vs
  controls; click a row for every rollout
- Fires: token-level activation shading and top-firing neurons for any text
- Sculpt: convert a steered-neuron finding into a persistent weight edit —
  scale/ablate write column, bias push (exact equivalents of the steering
  hooks), or scale read rows; undoable op log + side-by-side compare

**AHA loop editing** (`aha_editor.py`, `aha_test.py` — ported from
Eppy-EpHard/AHA, MIT)

- Activation-pair/LP editing for repetitive-loop behavior: dup-mask loop
  detector, matched-pair direction finding, PDHG LP for sparse writes,
  rank-one `down_proj` edits, `--pro` (pro-loop) control mode
- Steering probe mode sweeps a residual direction at dose multiples with
  random-vector controls — established that the separator direction is
  causal but semantically sign-inverted on base Qwen2.5-0.5B
- `sweep_eval.py`: λ-sweep any registry eval along a recorded weight
  delta vs a norm-matched random direction — tests whether an eval can
  rank edit directions (eval-as-compass, finite-difference signal)

**iWSP Autoresearch** (`iwsp_driver.py` + `iwsp_harness/`, ported from
the iWSP evolve experiments — OpenRouter instead of Claude Code)

- An OpenRouter model gets file/eval/log tools confined to a run dir and
  iterates `program.md` against the interpretable-transformer harness
- Results so far: hand-written 100%-accuracy parity circuits compressed
  to ~200 parameters, multiple distinct circuit families across seeds
- iWSP Autoresearch tab: launch detached local runs, poll progress,
  browse past runs' CSVs and snapshots

**Remote & external tools**

- RunPod Tinker: deploy/connect a GPU pod (no SSH needed), remote
  load/generate/invert, and remote neuron sweeps with progress polling
- Pod job runner (`/run`, `/jobs`, `/job/<id>`) executes uploaded
  scripts in the background with live log tailing, GPU util/mem stats,
  and ntfy.sh push notifications on stage markers and job exit; pods
  self-terminate on TTL/idle watchdog
- Circuit Tracer: launches the circuit-tracer attribution CLI from an
  isolated venv (`circuit_venv`)

## What you can do with it

- **Find out what a neuron does**: run a sweep over a layer, sort by shift
  score vs controls, open the neuron's rollouts, then fire it on your own
  text to see where it activates
- **Sculpt behavior, not just representations**: steer a neuron to confirm
  an effect, then bake it into the weights with a sculpt op — verify the
  weight edit reproduces the steered output before keeping it
- **Mutate and compare**: invert a parameter group, reorder or graft a
  layer, then paired-generate to see exactly what changed; undo from the
  op log
- **Scale up on demand**: run sweeps or big models on a RunPod pod from the
  same UI, pull results back, and terminate billing when done
- **Merge models as vectors**: blend a checkpoint toward/away from a
  reference and sweep λ to see where behavior breaks — a cheap probe of how
  compatible two weight spaces are (same-base finetunes merge cleanly;
  different-basin models degrade in interesting ways)
- **Edit in latent space**: build a delta ensemble from your own edits
  (sculpts, merges, jitters), compute a PCA basis, and mix principal
  components with sliders instead of touching individual tensors
- **Pick up where you left off**: sessions autosave as `last` — restart the
  app, reload the model, replay your ops
- **Teach a fact without training**: write "X is the creator of Y" into a
  small model via the Fact Writer, verify on held-out phrasings, and check
  the Eval Probe to see what the write cost the model elsewhere
- **Let an agent tinker for you**: point the WSP Agent at the local model
  or a RunPod pod and give it a goal ("find a neuron that affects
  repetition and ablate it"); it evaluates before/after and undoes edits
  that hurt
- **Evaluate like you mean it**: score an edit on `benchmark:arc_easy`
  while `perplexity` watches as guard, then Compare A→B — an optimized
  metric is no longer held-out, so verify on the metric you didn't chase
- **Run autoresearch**: fire an iWSP run from its tab and let an
  OpenRouter agent write, evaluate, and compress a tiny transformer
  circuit while you watch the CSV grow
- **Probe edit directions**: `sweep_eval.py` slides λ along a recorded
  weight delta against a random control — does the eval see a real
  direction or just perturbation magnitude?
- **Run the stack on GPU**: every remote op — eval, merges, fact writes —
  runs on a pod via the RunPod Tinker tab's weight-space accordion
- **Follow the research trail**: steering-first neuron interpretation
  ([neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)) and
  circuit attribution ([circuit-tracer](https://github.com/safety-research/circuit-tracer))

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Optional: circuit-tracer gets its own env (dep conflict — see below):
```bash
python -m venv circuit_venv
circuit_venv\Scripts\python.exe -m pip install git+https://github.com/safety-research/circuit-tracer.git
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. Enter a HuggingFace model name (e.g., 'gpt2', 'facebook/opt-125m'), a
   local model directory (e.g. `saved_model`), or pick one from the
   **cached/local models** dropdown — anything already in your HF cache or
   saved on disk works, no re-download
4. Click "Load Model" to load the model — the page also reflects a model
   already loaded server-side (e.g. from another session)
5. Pick a tab and start tinkering. Suggested first run: **Neuron Explorer →
   Steer**, layer 5, neuron 100, dose 0.8 → "Generate − / 0 / +"

## Output

All model outputs are saved to `outputs.jsonl`:
```json
{
    "timestamp": "2024-03-14T12:00:00",
    "model": "model_name",
    "prompt": "your prompt",
    "output": "model output",
    "parameters": {
        "max_length": 100,
        "temperature": 0.7
    }
}
```

Neuron sweep rollouts are saved incrementally to `neuron_runs/<name>.jsonl`
and are browsable even while a sweep is running.

Sessions live in `sessions/<name>.json` (model name, generation params,
config fingerprint) plus `<name>.ops.jsonl` (serializable op specs, replayed
in order on reload). Structural surgery is not yet serialized.

## RunPod Tinker

Run LLM Tinkerer operations on a remote RunPod GPU pod instead of your local machine.

1. Put your RunPod API key in one of:
   - environment variable `RUNPOD_API_KEY`
   - `.env` in this directory (`RUNPOD_API_KEY=rpa_...`)
   - `runpod_api_key.txt` (file containing only the key)
2. Go to the **RunPod Tinker** tab (the key field is pre-filled if found)
3. Pick a GPU type and click **Deploy / Connect Pod**, or paste an existing pod ID / proxy URL and click **Connect Existing Pod**
4. Once ready, use the remote **Load Model**, **Generate**, and **Weight Inversion** controls — or run a **neuron sweep on the pod** from Neuron Explorer → Sweep
5. The **Remote weight-space ops** accordion exposes the full new stack on
   the pod: eval probes, reference merges, and closed-form fact writes
6. Click **Terminate Pod** when finished to stop billing

The pod is deployed with the `dockerArgs` method: the handler and its requirements are
base64-encoded into the startup script, so no SSH or manual file uploads are required.

## Neuron Explorer details

- **Steer**: pick a layer and neuron, generate `−dose / baseline / +dose` rollouts
  side by side. Doses are `rel` (fractions of the median residual norm at the
  layer) or `abs` (raw vector norm).
- **Sweep**: neuron spec is `all`, `random:N`, `a-b`, or `a,b,c`. Runs locally or
  on the RunPod pod (the handler ships `neuron_explorer.py` to the pod and
  polls `/neuron_sweep_status`). Full-layer sweeps are thousands of rollouts —
  use subsets on CPU.
- **Browse**: ranks each swept neuron by how far its steered vocabulary shifted
  vs the controls; click a row to see every rollout for that neuron — also
  fills the Steer tab's layer + neuron.
- **Fires**: shades tokens red/blue by positive/negative activation for a
  neuron, or lists the top-firing neurons at a layer.
- **Sculpt**: *scale/ablate write column* and *bias push* are exact equivalents
  of their steering hooks (verified bit-identical on greedy decode); *scale
  read rows* scales the up/gate-projection row (no exact transient
  equivalent). Bias push requires a down-proj bias (GPT-2 yes, Llama no).
  Ops are logged and individually undoable.

Steering is a runtime hook on the live model, so it composes with weight
inversion, weight edits, and layer surgery.

## Circuit Tracer

The **Circuit Tracer** tab wraps `circuit_tracer_demo.py`, which shells out to the
`circuit-tracer` CLI. Its dependencies (transformers<5, huggingface-hub<1.0) conflict
with the main env's versions, so it lives in its own venv:

```bash
python -m venv circuit_venv
circuit_venv\Scripts\python.exe -m pip install git+https://github.com/safety-research/circuit-tracer.git
```

The demo auto-detects `circuit_venv/Scripts/circuit-tracer.exe`, or set
`CIRCUIT_TRACER_EXE` to point at another install.

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for larger models; everything
  runs on CPU, slower)
- RunPod API key (optional, only for RunPod Tinker tab and pod sweeps)
- See requirements.txt for full list of dependencies

## Project layout

- `app.py` — Gradio UI and ModelManager
- `neuron_explorer.py` — `NeuronExplorer` (steering, sweeps, ranking,
  activations) and `NeuronSculptor` (undoable weight edits)
- `weight_space.py` — `WeightMerger` (task-vector merges, λ sweeps, delta
  ensembles, PCA latent basis, PC-slider edits)
- `fact_editor.py` — `FactEditor` (closed-form MEMIT-style fact writes)
- `eval_probe.py` — `EvalProber` (MC accuracy + perplexity canary)
- `eval_registry.py` — unified eval registry: built-ins, data-defined
  custom evals, HF benchmark presets, guard evals, history + deltas
- `aha_editor.py` / `aha_test.py` — AHA loop-detection metrics, LP
  activation-pair edits, steering probes (runs on pod job runner)
- `sweep_eval.py` — λ-sweep of a registry eval along a weight delta
  vs norm-matched random control
- `iwsp_driver.py` / `iwsp_harness/` — OpenRouter autoresearch loop +
  vendored interpretable-transformer eval harness
- `vlm_fact_editor.py` — visual association writes for VLMs (language
  side only, vision encoder untouched)
- `wsp_agent.py` — OpenRouter agent: tool schemas, local/remote executors,
  agent loop with repeated-failure nudging
- `session_store.py` — session files + op-spec persistence
- `layer_surgery.py` — `LayerSurgeon` structural block ops
- `weight_inversion_implementation.py` — `WeightInverter`
- `model_visualizer.py` — architecture graph, weight/attention heatmaps
- `runpod_manager.py` / `runpod_handler.py` — pod deployment and remote API
- `circuit_tracer_demo.py` — circuit-tracer CLI wrapper
