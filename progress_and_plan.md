# Progress and Plan

## Recent Developments
- Integrated circuit-tracer minimal plan for model circuit analysis and visualization.
- Added Weight Inversion Tinker feature (invert all or part of model weights, with UI and safety features).
- Implemented dynamic layer pattern detection for layer range operations (works for GPT-2, Llama, BERT, etc.).
- Added a general Weight Editing tab in Gradio:
  - Select layer/module and parameter (weight, bias, etc.)
  - View stats and preview values
  - Set, scale, or randomize all values
  - Direct matrix editing for small parameters
  - Restore original weights at any time
- Added a **RunPod Tinker** tab:
  - Deploy or connect to a RunPod GPU pod via GraphQL API
  - Base64-embedded dockerArgs deployment (no SSH/SCP required)
  - Remote model loading, generation, weight inversion, and restore
  - List available GPUs, check pod health, and terminate pods
  - Reconnect to an existing pod by ID or proxy URL
  - Pre-fill API key from `RUNPOD_API_KEY`, `.env`, or `runpod_api_key.txt`
- Unified `WeightInverter` into `weight_inversion_implementation.py` (imported by `app.py` and the RunPod handler)
- Prompt/output history panel reads `outputs.jsonl`
- Circuit Tracer tab wraps `circuit_tracer_demo.py`
- Weight Inversion tab: paired generate (original vs modified) and live original/modified view toggle
- Model Architecture tab: clickable Plotly graph, multi-select nodes, descendant parameter list, batch set/scale/random
- All features are model-agnostic and support major HuggingFace transformer models.
- Added **Layer Surgery** tab (`layer_surgery.py`, `LayerSurgeon`):
  - Detects the repeating decoder block ModuleList (GPT-2 `transformer.h`, Llama `model.layers`, ...)
  - Insert a deepcopied block at any position; init modes: copy source weights,
    copy+noise (gaussian jitter at N% of each tensor's std), identity (zero out-projections), random
  - Move a block to a new index (reorder the stack); jitter a block's weights in place
  - Remove a block (with confirm; refuses to remove the last one)
  - Insert a submodule (e.g. `mlp`) as a standalone residual layer via `ResidualWrap`
    (`y = x + inner(x)`; filters block-level kwargs, matches tuple-vs-tensor return convention
    detected by a forward-hook probe)
  - Same-model graft at any granularity: whole block, `mlp`, `attn`, single tensor
    (`copy_module_weights`, strict `load_state_dict`)
  - Fixes `layer_idx` attributes and syncs the config depth attr (`num_hidden_layers`/`n_layer`/...)
  - Structural ops rebase the `WeightInverter` backup (param names shift on insert/remove)
- Added **Neuron Explorer** tab (`neuron_explorer.py`, `NeuronExplorer`) —
  the steering-first method from lyramakesmusic/neuron-explorer, model-agnostic:
  - Discovers the decoder block stack and each block's down-projection by shape
    (handles nn.Linear and Conv1D), so steering works on any loaded causal LM
  - `Steer`: −/0/+ rollouts for one neuron; hooks run on the live model, so they
    compose with inversion, edits, and surgery (reuses the surgeon's cached
    block stack so positions stay correct after inserts)
  - `Sweep`: neurons (`all`/`random:N`/`a-b`/`a,b,c`) × prompts × doses, plus
    per-prompt baselines and random-direction controls, appended incrementally
    to `neuron_runs/<name>.jsonl`; `rel`/`abs` dose units
  - `Browse`: word-shift score per unit vs controls percentile; click a row for
    its rollouts
  - `Fires`: token-level activation shading / top-firing neurons
  - RunPod: `runpod_handler.py` gained `/neuron_info`, `/neuron_steer_generate`,
    and a background-thread `/neuron_sweep` + status/result endpoints;
    `neuron_explorer.py` is shipped to pods via dockerArgs alongside the handler
  - Smoke-tested on gpt2 (CPU): steer, 22-rollout mini sweep, ranking, fires,
    post-surgery position shifts
  - `Sculpt` (`NeuronSculptor`): steering findings -> persistent weight edits.
    scale/ablate write column, bias push (exact hook equivalents, verified
    bit-identical greedy decode), scale read rows (approximate). Undoable op
    log storing original tensor slices; transient-vs-edited compare generate.
    Browse row click fills Steer layer+neuron. Note: bias push needs a
    down-proj bias (GPT-2 yes, Llama no).
- Added `interface.load`/`refresh_ui_state` + `_loaded_model_updates()`:
  page load now reflects the server-side model in every browser session
  (previously widgets only populated in the session that clicked Load).
  Verified end-to-end via gradio_client on the live app: load, steer, fires,
  22-rollout sweep, browse ranking, sculpt apply/compare/undo.

### Env notes (learned 2026-11-04)

- **venv is not relocatable**: the project moved OneDrive\Desktop -> Desktop;
  `venv\Scripts\*.exe` launchers keep the old python path baked in and fail.
  `python -m pip` works; recreate the venv or force-reinstall owning packages
  to regenerate launchers.
- **circuit-tracer is isolated in `circuit_venv/`**: it pins transformers<5 +
  huggingface-hub<1.0, which conflict with main venv (tf 5.15 + gradio needs
  hub>=1.16). `circuit_tracer_demo.py` resolves the CLI via
  `CIRCUIT_TRACER_EXE` -> `circuit_venv` -> PATH. Don't install
  circuit-tracer into `venv`.

### Gradio session-state gotchas (learned 2026-11-03)

- `gr.update(choices=...)` persists only into **per-session** component state.
  Browser sessions (websocket) keep it; stateless API calls do not — a
  dropdown updated in one session validates `choices=[]` in another.
- Neuron Explorer dropdowns use `allow_custom_value=True` as a safety net.
  Older tabs' dropdowns (layer_selector, layer_dropdown, param_dropdown,
  graft/insub selectors, arch_nodes CheckboxGroup) still rely on session
  state — fine in-browser, but API calls that skip the load step get
  "not in the list of choices". Normalize if the API becomes a real path.
- CheckboxGroup validates choices too and has no allow_custom_value escape.
- gradio_client sends session_hash, so same-`Client` sequences DO share
  session state — useful for testing.

## NEXT SLICE: session/ops persistence (deferred from 2026-11-03)

Goal: state survives restarts and session boundaries. This doubles as the
forcing function for the patch stack / marshals (ops become serializable data).

### Design decisions (agreed)

- Persist **op specs**, not tensor state: `{kind, layer, neuron, factor,
  dose, dose_mode, ts}` per op; re-derive backups on replay. Same for
  inversion config and weight-edit params.
- **Replay, don't snapshot** — op replay is transparent, reorderable, and
  becomes the marshals system. "Save Model" stays the snapshot escape hatch.
- Files: `sessions/<name>.json` (model name, safetensors, gen params, UI
  fields) + `sessions/<name>.ops.jsonl` (ordered op list). JSON only, never
  pickle.
- Model identity guard: store model name + config fingerprint; warn/refuse
  replay on mismatch.
- Startup UX: restore fields immediately; model reload via explicit
  "Reload last model" button (load is the expensive part), then replay ops.

### Pitfalls to handle

- **Replay order + index drift**: "layer 5" shifts meaning after surgery
  insert. Prefer module *names* over indices where possible; surgery op
  serialization deferred from v1 anyway.
- **Partial replay**: op N of M fails (shape changed) -> decide
  stop-and-report vs apply-what-you-can. Default: stop, report, offer skip.
- NeuronSculptor already keeps backups in `op["backups"]`; needs a parallel
  `spec` dict for serialization + a `replay(spec)` path.
- Restore should run on a clean model: restore/undo any existing ops first.

### v1 scope

1. `sessions/` save/load for model name + params + UI fields
2. Serializable op specs for NeuronSculptor ops (sculpt, later inversion)
3. "Reload last model" + "Replay ops" + autosave-on-change (or explicit Save)
4. Defer: surgery op serialization, weight-edit/arch-graph op unification
   (the full patch-stack consolidation), runpod session fields

### Quick test items still open

- Layer Inspection + arch graph click-through in browser (backend verified
  via API; UI path was untested until the page-load fix)
- A real `random:64` sweep in browser to watch progress UX
- Fires token shading render in browser

## Layer surgery: deferred ideas

- **Train added layers on RunPod** — freeze all params except the new block;
  LM loss on pasted text or distillation against a pre-surgery output snapshot.
  Pod-side: add a train step to `runpod_handler.py`; local side: send corpus +
  layer index + hyperparams.
- **Cross-checkpoint weight transfer** — load a second HF model's state dict
  (same arch first) and graft whole blocks or shape-matched tensors.
- **Gated residual wrapper** — `y = x + alpha * block(x)`, slider and/or
  trainable gate, so an inserted layer can be dialed from pass-through up.
- Simple modules at layer boundaries (Linear adapter, bottleneck) — TODO in
  `layer_surgery.py` once block insertion proves out.

## Next: consolidate inversion, edit, and generation

Basic pieces work, but the loop is split across tabs:

1. Load / prompt / generate on **Text Generation**
2. Click modules and batch-edit on **Model Architecture**
3. Invert / restore / paired compare / live original-vs-modified on **Weight Inversion**
4. Layer/param dropdown edits on **Weight Editing**

That hop is fine for discovery. The next slice should make one **tinker loop** without adding new mutation types.

### Suggested loop

`Load model → choose a surface (graph, invert, named param) → apply one op → generate original vs modified → keep or restore`

Keep **RunPod** and **Circuit Tracer** as side tools. Do not fold them into the core bench.

### Layout suggestion (tomorrow)

One **Workbench** tab, three panes:

- **Left: mutate.** Sub-modes, not separate destinations:
  - Invert (current inversion controls)
  - Graph select (current architecture click + param checkboxes)
  - Named param (current Weight Editing dropdowns)
- **Center: compare.** Shared prompt, max length, temperature, paired original / modified outputs, live view toggle.
- **Right: session.** Op log, restore last, restore all, last status.

Load model can stay in the header or on Text Generation. Architecture graph can remain a large view, but selecting a node should land in the same mutate/compare loop.

### Backend suggestion (do this before more UI polish)

Record inversions and edits as a **patch stack** (already in feature creep):

- Each Apply / Invert is one op: target, mode, value/scale/multiplier, timestamp
- Live model is original + applied ops
- Toggle original/modified by unapplying / reapplying the stack
- Paired generate uses that stack, not a second model
- Restore pops one op or clears the stack

This unifies Architecture edits, Weight Editing, and Weight Inversion. It is also the prerequisite for later marshals.

### What not to do tomorrow

- New mutation types (CA, GA, annealing, marshals)
- Merging RunPod into the local loop
- Safety/alignment probes
- Replacing the graph again

### Graph mutate: current set/scale vs later structured edits

Architecture **set** / **scale** / **random** are broadcast ops today:

- **set** — `param.data.fill_(scalar)` writes the same number into every element
- **scale** — `param.data.mul_(scalar)` multiplies every element by one factor
- **random** — `param.data.normal_()` overwrites the whole tensor

You cannot paste a matrix into Set Value, and you cannot scale by a vector. Nodes are matrices; the UI just paints one scalar across them.

Weight Editing already has a separate **matrix paste** path (`set_param_matrix`) for one named param, exact shape only. It is not wired on the Architecture tab.

Later, after the shared op log, useful graph-side upgrades (not tomorrow’s first slice):

- Paste a Python list / matrix when the selected param is small enough
- Scale by a vector with broadcast rules (row, column, or last dim)
- Slice edits: one row, one head, one output channel
- Reject or warn on shape mismatch instead of silently filling

### First coding slice (when we pick this up)

1. Shared prompt + paired generate visible from the mutate surfaces
2. One op log that Architecture Apply, Weight Editing Apply, and Invert all append to
3. Restore last / restore all from that log
4. Only then consider physically merging the three edit tabs into Workbench

## Feature creep (tinker, later)

- **Marshals** — named cascade of changes; generate after each step; replay / skip / reorder
- Cascade compare (original → after 1 → after 1+2 → …)
- **Hot-swap / patch stack** — store inversions and edits as ops, apply/unapply on the live model instead of cloning all weights. Prerequisite for marshals on large models.
- Layer freeze / ablation at inference
- Head-by-head attention
- Streaming generate + cancel

## Alignment & Safety (sibling, not a Tinkerer pivot)

Tinkerer stays a curiosity workbench. Alignment/safety measurement lives in `safety+align/` and should be promoted to its own repo.

Spec: `safety+align/SPEC.md`  
Intent: `safety+align/INTENT.txt`

Tinkerer may keep sharing **instruments** (inverter, generate, later RunPod). It should not grow jailbreak probes, personality sliders, or a “safety lab” README.

## Long Term Plans
- Generalize tinker functions (weight inversion, editing, etc.) into an importable Python module/package.
- Develop an MCP (Model Control Protocol) server to deploy these tinker tools as a service.
- Enable LLM assistants (or other clients) to access and use these tinker toy tools via the MCP server API.
- Continue expanding tinker features (e.g., representation engineering, control vectors, advanced visualizations, model surgery, etc.).
- Improve documentation, testing, and deployment for broader community use. 