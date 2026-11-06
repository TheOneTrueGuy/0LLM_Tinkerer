# Paired Intervention Lab — spec

Working title. Rename when this directory becomes its own repo.

Sibling of **LLM Tinkerer**, not a fork of its soul. Tinkerer stays a curiosity workbench. This project asks one question:

> If I change *this* piece of *this* model, what happens to *these* behaviors — and can I show it?

## 1. Why a new project

LLM Tinkerer already has the instrument:

- HuggingFace load + generate
- `WeightInverter` with backup / restore / preview / layer-range / component filters
- Weight editing (set / scale / randomize)
- Output log (`outputs.jsonl`)
- Optional RunPod remote execution
- Optional Circuit Tracer wrapper

What it does **not** have, and should not grow, is an **experiment contract**: same prompts, two states, scores, intervention config, a report.

Job-shaped alignment work lives here. Play lives in Tinkerer. Sharing code is fine; sharing *purpose* is not.

## 2. Stance (non-negotiable)

| Do | Do not |
|---|---|
| Study models **you** load | Attack third-party products or APIs |
| Intervene, measure, restore | Leave a steered model as the default |
| Tiny, versioned probe sets | Ship a public jailbreak corpus |
| Keyword / human scores first | Fake “safety certified” dashboards |
| Call steering *steering* | Call it personality, soul, or neurosurgery in the product voice |
| Export a writeup | Claim this is an alignment program |

**Personality / control vectors** are v2+ and only with a written hypothesis (“does direction D track sycophancy on set S?”). They are not a fun slider.

**Harmful probes** (disallowed requests, jailbreak-style strings) stay in a local, gitignored file if they exist at all. v1 ships **benign-only**: over-refusal, sycophancy, user-is-wrong, instruction-hierarchy on harmless tasks.

## 3. What to steal from Tinkerer (code inspiration)

Reuse as **library ideas**, not by importing `app.py`.

### Steal

1. **`WeightInverter` as an operator, not a tab**  
   `backup_weights` / `invert_weights(config)` / `restore_weights` / `get_inversion_preview`. Config dict: `type`, `multiplier`, `layer_start`, `layer_end`, `custom_rules`. Dynamic layer-pattern detection (GPT-2 `.h.N.`, Llama `.layers.N.`).

2. **Named intervention types as experimental factors**  
   Full / attention / MLP / output / embedding / layer range. In this project they are **ablation conditions**, not toys.

3. **Confirm + restore**  
   Every destructive op has an explicit restore path. Experiments should `try/finally: restore`.

4. **Structured run records**  
   Tinkerer’s `outputs.jsonl` shape (`timestamp`, `model`, `prompt`, `output`, `parameters`) plus `intervention` and `metrics`.

5. **Remote execution later**  
   RunPod manager pattern (GraphQL pod, dockerArgs handler, `/generate` `/invert_weights` `/restore_weights`) is how you run 7B+ paired evals. v1 can be local GPT-2 for plumbing.

6. **Circuit Tracer as a microscope, not the product**  
   Attribution on a *measured* behavior, then jump to invert that layer range. Optional, after paired eval works.

### Do not steal

- Gradio kitchen-sink tabs (generation, editing, inversion, RunPod, tracer in one app)
- Cellular automata / genetic algorithms / annealing
- Direct matrix edit UI
- Architecture-graph-as-homepage
- README energy of “tinker with anything”

### Shape instead

A **thin CLI + optional small UI**:

```
load model
  → run probe set (state A)
  → apply intervention
  → run probe set (state B)
  → restore
  → write report
```

UI is a report viewer and a “run this yaml” button, not a second Tinkerer.

## 4. Core loop

```
Model + ProbeSet + Intervention
        │
        ▼
   generate A (clean)
        │
        ▼
   apply intervention
        │
        ▼
   generate B (intervened)
        │
        ▼
   restore
        │
        ▼
   score(A, B) → ExperimentRecord → report.md
```

One command should be enough:

```text
python -m palab run experiments/late_mlp_sycophancy.yaml
```

## 5. v1 scope (build this, stop)

**In:**

- Load HF causal LM (same constraints as Tinkerer: no GGUF)
- Probe YAML/JSON: id, category, prompt, optional expected_behavior
- Categories for v1: `benign`, `over_refusal`, `sycophancy`, `user_wrong`
- Intervention: wrap Tinkerer-style inverter (copy or later extract package)
- Paired generate, greedy or fixed seed
- Scores: keyword rules + `unclear` bucket (no LLM-as-judge yet)
- Record: jsonl + markdown report (side-by-side A/B text, score table)
- Restore guaranteed

**Out of v1:**

- Control / personality vectors
- Harmful / jailbreak probes in-repo
- Circuit Tracer
- RunPod
- Training, fine-tune, merge
- Public leaderboard
- MCP server

**v1 artifact:** one report you could put in a portfolio:  
“On `gpt2` (plumbing) then on instruct-model-X: inverting late MLP changed sycophancy score from _ to _ ; restore returned baseline.”

## 6. Data model

```yaml
# experiments/example.yaml
model: gpt2
seed: 0
max_length: 80
probes: probes/benign_v1.yaml
intervention:
  type: mlp_only          # or layer_range, output_only, ...
  multiplier: -1.0
  layer_start: 8
  layer_end: 11
scoring: keywords_v1
```

```text
ExperimentRecord
  id, timestamp, model, seed
  intervention: {type, multiplier, layers, preview_param_count}
  probes[]:
    id, category, prompt
    output_a, output_b
    score_a, score_b          # refused | complied | sycophantic | factual | unclear
  summary: {n, by_category: {delta...}}
```

Keep harmful probe files out of git (`.gitignore`: `probes/local/`).

## 7. Scoring (honest and dumb)

v1 keyword scorer, per category:

- **over_refusal** — did the model refuse a harmless request?
- **sycophancy** — did it agree with a false user claim?
- **user_wrong** — did it correct the user?
- **benign** — fluency/sanity check (did output collapse to garbage after inversion?)

Publish the keyword lists in-repo so scores are criticizable. Human override field on each probe for the report.

Do not average everything into a single “alignment score.”

## 8. Ethics / UX copy

- Product name should sound like a **lab notebook**, not a mind clinic.
- Buttons: Apply intervention / Restore. Never “rewrite personality.”
- README: intended use = research on models you load.
- If v2 adds steering vectors: require a hypothesis string in the yaml or refuse to run.

## 9. Relationship to Tinkerer

| | Tinkerer | This project |
|---|---|---|
| North star | Fun, weird, visual | Paired measurement |
| Inverter | Feature | Experimental factor |
| Log | `outputs.jsonl` | Experiment records + report |
| Vectors | Speculative tinker | Hypothesis-gated, late |
| GitHub story | “I built a workbench” | “I ran this ablation and wrote it up” |

Later: extract `WeightInverter` into a tiny shared package both import. Until then, **copy** the module and note the upstream path so Tinkerer can stay messy.

## 10. Suggested layout (when this becomes a repo)

```text
INTENT.txt
SPEC.md
README.md
palab/
  inverter.py          # copied from Tinkerer WeightInverter
  generate.py
  probes.py
  score.py
  report.py
  run.py
probes/
  benign_v1.yaml
  local/               # gitignored
experiments/
  example.yaml
records/
reports/
```

## 11. Build order

1. Copy inverter + generate-one-prompt
2. Probe loader + paired run + restore-in-finally
3. Keyword scorer + jsonl record
4. Markdown report
5. One real experiment on an instruct model (local or future RunPod)
6. Only then: layer sweep, Circuit Tracer, vectors, remote GPU

## 12. Open questions (answer when promoting to its own repo)

- Official name?
- First non-GPT-2 model you can actually load?
- Stay CLI-only for v1, or a 1-page Gradio report runner?
- License / public vs private GitHub?

Those are product choices. They are not required to keep Tinkerer itself a tinker toy.
