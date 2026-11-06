# LLM Tinkerer

A platform for experimenting with language models, built with Gradio and HuggingFace Transformers.

## Features

- Load and run HuggingFace language models
- Adjust generation parameters (temperature, max length)
- Save and track model outputs
- Simple and intuitive interface

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

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. Enter a HuggingFace model name (e.g., 'gpt2', 'facebook/opt-125m')
4. Click "Load Model" to load the model
5. Enter your prompt and adjust generation parameters
6. Click "Generate" to run the model

## Output

All model outputs are saved to `outputs.jsonl` in the following format:
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

## RunPod Tinker

Run LLM Tinkerer operations on a remote RunPod GPU pod instead of your local machine.

1. Put your RunPod API key in one of:
   - environment variable `RUNPOD_API_KEY`
   - `.env` in this directory (`RUNPOD_API_KEY=rpa_...`)
   - `runpod_api_key.txt` (file containing only the key)
2. Go to the **RunPod Tinker** tab (the key field is pre-filled if found)
3. Pick a GPU type and click **Deploy / Connect Pod**, or paste an existing pod ID / proxy URL and click **Connect Existing Pod**
4. Once ready, use the remote **Load Model**, **Generate**, and **Weight Inversion** controls
5. Click **Terminate Pod** when finished to stop billing

The pod is deployed with the `dockerArgs` method: the handler and its requirements are
base64-encoded into the startup script, so no SSH or manual file uploads are required.

## Neuron Explorer

The **Neuron Explorer** tab implements the steering-first exploration method from
[lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer),
model-agnostically (works on any loaded HF causal LM — GPT-2, Llama, OPT, Bloom, ...).

- **Steer**: pick a layer and neuron, generate `−dose / baseline / +dose` rollouts
  side by side. Doses are `rel` (fractions of the median residual norm at the
  layer) or `abs` (raw vector norm).
- **Sweep**: push many neurons × doses × prompts; every rollout is appended to
  `neuron_runs/<name>.jsonl`, plus unsteered baselines and random-direction
  controls. Neuron spec: `all`, `random:N`, `a-b`, or `a,b,c`. Runs locally or
  on the RunPod pod (the handler ships `neuron_explorer.py` to the pod and
  polls `/neuron_sweep_status`).
- **Browse**: ranks each swept neuron by how far its steered vocabulary shifted
  vs the controls; click a row to see every rollout for that neuron.
- **Fires**: runs your text through the model and shades tokens by a neuron's
  activation, or lists the top-firing neurons at a layer.
- **Sculpt**: converts a steered-neuron finding into a persistent weight edit.
  *Scale/ablate write column* and *bias push* are exact equivalents of their
  steering hooks (verified bit-identical on greedy decode); *scale read rows*
  scales the up/gate-projection row (no exact transient equivalent). Ops are
  logged and individually undoable, and a compare button generates the
  transient-steered vs weight-edited outputs side by side. Clicking a ranked
  row in Browse fills in that neuron's layer/index on the Steer tab.

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
- CUDA-capable GPU (optional, but recommended for larger models)
- RunPod API key (optional, only for RunPod Tinker tab)
- See requirements.txt for full list of dependencies