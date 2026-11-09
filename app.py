import ast
import logging
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
print("Starting LLM Tinkerer...", flush=True)
print("Loading libraries (torch/transformers can take 20-40s with no other output)...", flush=True)

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Any, List
import html as html_lib
import json
import random
import time
from datetime import datetime
from model_visualizer import ModelVisualizer
from runpod_manager import RunPodManager, GPU_PREFERENCE, load_runpod_api_key, proxy_url_from_pod_id
from weight_inversion_implementation import WeightInverter
from layer_surgery import LayerSurgeon
from circuit_tracer_demo import run_circuit_tracer
from neuron_explorer import NeuronExplorer, NeuronSculptor, neuron_write_column, list_runs
import session_store
import wsp_agent
from weight_space import WeightMerger, MERGE_MODES
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.visualizer = ModelVisualizer()
        self.weight_inverter = None
        self.layer_surgeon = None
        self.neuron_sculptor = None
        self.weight_merger = None
        self.fact_editor = None
        self.eval_prober = None
        logger.info(f"Using device: {self.device}")

    def get_model_structure(self) -> str:
        if not self.model:
            return "No model loaded"
        structure = []
        structure.append(f"Model: {self.model_name}")
        if self.config:
            structure.append("Configuration:")
            arch = getattr(self.config, 'architectures', ['Unknown'])
            structure.append(f"  Architecture: {arch[0] if arch else 'Unknown'}")
            structure.append(f"  Hidden Size: {getattr(self.config, 'hidden_size', 'Unknown')}")
            structure.append(f"  Number of Layers: {getattr(self.config, 'num_hidden_layers', 'Unknown')}")
            structure.append(f"  Number of Attention Heads: {getattr(self.config, 'num_attention_heads', 'Unknown')}")
            if hasattr(self.config, 'intermediate_size'):
                structure.append(f"  Intermediate Size: {self.config.intermediate_size}")
            if hasattr(self.config, 'vocab_size'):
                structure.append(f"  Vocabulary Size: {self.config.vocab_size}")
        return "\n".join(structure)

    def get_layer_names(self) -> List[str]:
        """Returns a list of layer names that can be inspected."""
        if not self.model:
            return []

        try:
            from transformers.pytorch_utils import Conv1D
            extra_types = (Conv1D,)
        except Exception:
            extra_types = ()

        inspect_types = (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm) + extra_types
        layer_names = []
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if isinstance(module, inspect_types) or hasattr(module, "weight"):
                layer_names.append(name)
        return layer_names

    def load_model(self, model_name: str, use_safetensors: bool = True) -> Dict[str, Any]:
        try:
            logger.info(f"Loading model: {model_name} (use_safetensors={use_safetensors})")
            # Detect unsupported formats (GGUF)
            if model_name.lower().endswith('.gguf') or 'gguf' in model_name.lower():
                raise ValueError("GGUF models are not supported for direct loading in this interface. Please use a HuggingFace-compatible model (PyTorch or safetensors). [Research on GGUF loading is ongoing.]")
            self.config = AutoConfig.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                use_safetensors=use_safetensors
            )
            self.model_name = model_name
            self.visualizer.model = self.model
            
            # Initialize weight inverter
            self.weight_inverter = WeightInverter(self.model)
            self.layer_surgeon = LayerSurgeon(self.model)
            self.neuron_sculptor = None
            self.weight_merger = WeightMerger(self.model)
            self.fact_editor = None
            self.eval_prober = None
            
            # Create architecture visualization (handle None gracefully)
            try:
                arch_viz = self.visualizer.create_architecture_payload(self.model)
            except Exception as viz_e:
                logger.warning(f"Could not create architecture visualization: {viz_e}")
                arch_viz = "{}"
            
            return {
                "status": "success",
                "message": f"Model {model_name} loaded successfully",
                "structure": self.get_model_structure(),
                "arch_viz": arch_viz,
                "layers": self.get_layer_names()
            }
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            # On error, ensure dropdowns and UI elements are reset
            return {
                "status": "error",
                "message": f"Error loading model: {str(e)}",
                "structure": "No model loaded",
                "arch_viz": "{}",
                "layers": []
            }

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        if not self.model or not self.tokenizer:
            return "Please load a model first"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error generating text: {str(e)}"

    def generate_paired(self, prompt: str, max_length: int = 100, temperature: float = 0.7):
        """Generate the same prompt on original backup vs current modified weights."""
        empty = "Load a model first."
        if not self.model or not self.tokenizer or not self.weight_inverter:
            return empty, empty, "Please load a model first"
        if not prompt or not str(prompt).strip():
            return "", "", "Enter a prompt first."
        inverter = self.weight_inverter
        if not inverter.original_weights:
            inverter.backup_weights()
        previous_mode = inverter.view_mode
        modified = inverter.snapshot_modified()
        try:
            inverter.apply_params(inverter.original_weights)
            inverter.view_mode = "original"
            original_out = self.generate_text(prompt, max_length, temperature)
            inverter.apply_params(modified)
            inverter.view_mode = "modified"
            inverter.working_weights = {}
            modified_out = self.generate_text(prompt, max_length, temperature)
            if previous_mode == "original":
                inverter.set_view_mode("original")
            return original_out, modified_out, f"Paired generate done. Live view: {inverter.view_mode}."
        except Exception as e:
            logger.error(f"Paired generate failed: {e}", exc_info=True)
            try:
                inverter.apply_params(modified)
                inverter.view_mode = "modified"
            except Exception:
                pass
            err = f"Paired generate failed: {e}"
            return err, err, err

    def set_weight_view(self, mode: str) -> str:
        if not self.weight_inverter:
            return "Please load a model first"
        try:
            return self.weight_inverter.set_view_mode(mode)
        except Exception as e:
            return f"Could not switch view: {e}"

    def get_weight_visualization(self, layer_name: str) -> go.Figure:
        """Gets weight visualization for a specific layer."""
        if not self.model:
            return None
        weight_data = self.visualizer.get_weight_visualization(layer_name)
        return self.visualizer.create_weight_plot(weight_data)

    def get_attention_pattern(self, layer_name: str, input_text: str) -> go.Figure:
        """Gets attention pattern visualization for a specific layer."""
        if not self.model or not self.tokenizer:
            return None
        attention_data = self.visualizer.get_attention_pattern(layer_name, input_text, self.tokenizer)
        return self.visualizer.create_attention_plot(attention_data)

    def invert_weights(self, inversion_type: str, multiplier: float, layer_start: int, layer_end: int, custom_rules: str, confirm: bool) -> str:
        """Handle weight inversion through the UI."""
        if not self.weight_inverter:
            return "Please load a model first"
        
        if not confirm:
            return "Please confirm the inversion operation by checking the confirmation box."
        
        config = {
            'type': inversion_type.lower().replace(' ', '_'),
            'multiplier': multiplier,
            'layer_start': int(layer_start) if layer_start else 0,
            'layer_end': int(layer_end) if layer_end else 999,
            'custom_rules': [rule.strip() for rule in custom_rules.split(',') if rule.strip()]
        }
        
        try:
            result = self.weight_inverter.invert_weights(config)
            return f"""Inversion completed!
- Inverted {result['inverted_count']} out of {result['total_count']} parameters
- Percentage inverted: {result['percentage']:.2f}%
- Type: {inversion_type}
- Multiplier: {multiplier}

You can now test the model with generation to see the effects."""
        except Exception as e:
            logger.error(f"Error during weight inversion: {str(e)}")
            return f"Error during inversion: {str(e)}"

    def restore_weights(self) -> str:
        """Restore model weights to original values."""
        if not self.weight_inverter:
            return "Please load a model first"
        
        try:
            self.weight_inverter.restore_weights()
            return "Weights restored to original values successfully."
        except Exception as e:
            logger.error(f"Error restoring weights: {str(e)}")
            return f"Error restoring weights: {str(e)}"

    def preview_inversion(self, inversion_type: str, layer_start: int, layer_end: int, custom_rules: str) -> str:
        """Preview which parameters would be inverted."""
        if not self.weight_inverter:
            return "Please load a model first"
        
        config = {
            'type': inversion_type.lower().replace(' ', '_'),
            'layer_start': int(layer_start) if layer_start else 0,
            'layer_end': int(layer_end) if layer_end else 999,
            'custom_rules': [rule.strip() for rule in custom_rules.split(',') if rule.strip()]
        }
        
        try:
            preview = self.weight_inverter.get_inversion_preview(config)
            return f"Would invert {len(preview)} parameters:\n" + "\n".join(preview[:20]) + \
                   (f"\n... and {len(preview) - 20} more" if len(preview) > 20 else "")
        except Exception as e:
            return f"Error generating preview: {str(e)}"

    def save_model(self, save_dir):
        if not self.model or not self.tokenizer:
            return "No model loaded."
        try:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            return f"Model and tokenizer saved to {save_dir}"
        except Exception as e:
            return f"Error saving model: {e}"

    # --- GGUF loading research placeholder ---
    def load_gguf_model(self, model_path: str):
        """
        Placeholder for future GGUF model loading support.
        Currently, GGUF models are not supported in HuggingFace transformers.
        Research ongoing: see https://github.com/ggerganov/llama.cpp and related Python bindings.
        """
        raise NotImplementedError("GGUF model loading is not yet supported. [Planned]")

class OutputManager:
    def __init__(self):
        self.output_file = "outputs.jsonl"
        self.history = []

    def save_output(self, prompt: str, output: str, model_name: str, params: Dict[str, Any]):
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "model": model_name,
            "prompt": prompt,
            "output": output,
            "parameters": params
        }
        self.history.append(entry)
        
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not os.path.exists(self.output_file):
            return list(reversed(self.history[-limit:]))
        entries = []
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.error(f"Could not read output history: {e}")
            return list(reversed(self.history[-limit:]))
        return list(reversed(entries[-limit:]))

    def format_history(self, limit: int = 20) -> str:
        recent = self.load_recent(limit)
        if not recent:
            return "No saved outputs yet."
        blocks = []
        for entry in recent:
            params = entry.get("parameters") or {}
            prompt = (entry.get("prompt") or "").strip()
            output = (entry.get("output") or "").strip()
            if len(prompt) > 240:
                prompt = prompt[:240] + "..."
            if len(output) > 400:
                output = output[:400] + "..."
            blocks.append(
                f"[{entry.get('timestamp', '')}] {entry.get('model', '')}\n"
                f"params: {params}\n"
                f"prompt: {prompt}\n"
                f"output: {output}"
            )
        return "\n\n---\n\n".join(blocks)

def create_interface():
    model_manager = ModelManager()
    output_manager = OutputManager()

    def _loaded_model_updates(message: str, structure: str, arch_viz):
        """The full 17-output update tuple for a loaded model — shared by
        Load Model and the page-load refresh."""
        inspect_layers = model_manager.get_layer_names()
        edit_layers = _editable_layers()
        inspect_update = gr.update(
            choices=inspect_layers,
            value=(inspect_layers[0] if inspect_layers else None),
        )
        edit_update = gr.update(
            choices=edit_layers,
            value=(edit_layers[0] if edit_layers else None),
        )
        edit_params = _module_params(edit_layers[0] if edit_layers else None)
        param_update = gr.update(
            choices=edit_params,
            value=(edit_params[0] if edit_params else None),
        )
        arch_nodes_update = gr.update(choices=_all_modules(), value=[])
        arch_params_update = gr.update(choices=[], value=[])
        surgery_info, surgery_overview = _surgery_overview()
        try:
            graft_choices = model_manager.layer_surgeon.graftable_modules()
        except Exception:
            graft_choices = []
        try:
            insub_choices = model_manager.layer_surgeon.list_submodules(0)
        except Exception:
            insub_choices = []
        nx_layer_update = _nx_layer_update()
        return (
            message,
            structure,
            arch_viz,
            inspect_update,
            edit_update,
            param_update,
            arch_nodes_update,
            arch_params_update,
            surgery_info,
            surgery_overview,
            gr.update(choices=graft_choices, value=None),
            gr.update(choices=graft_choices, value=None),
            gr.update(
                choices=insub_choices,
                value=(insub_choices[0] if insub_choices else None),
            ),
            nx_layer_update,
            nx_layer_update,
            nx_layer_update,
            gr.update(choices=list_runs(), value=None),
        )

    def load_model(model_name: str, use_safetensors: bool):
        result = model_manager.load_model(model_name, use_safetensors=use_safetensors)
        empty = gr.update(choices=[], value=None)
        if result["status"] == "error":
            return (
                result["message"],
                result["structure"],
                "{}",
                empty,
                empty,
                empty,
                empty,
                empty,
                "No model loaded.",
                "",
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
            )
        # autosave the "last" session so a restart can restore it
        try:
            session_store.save_session("last", {
                "v": 1,
                "model_name": model_manager.model_name,
                "use_safetensors": use_safetensors,
                "gen": {"max_length": 100, "temperature": 0.7},
                "fingerprint": session_store.config_fingerprint(model_manager.config),
            })
            session_store.save_ops("last", [])
        except Exception as e:
            logger.warning(f"Could not autosave session: {e}")
        return _loaded_model_updates(
            result["message"], result["structure"], result["arch_viz"]
        )

    def _cached_model_names():
        """Models already on disk: HF hub cache entries plus local dirs that
        look like saved checkpoints (config.json + weights)."""
        import os
        names = []
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub")
        try:
            for d in os.listdir(hub):
                if d.startswith("models--"):
                    names.append(d[len("models--"):].replace("--", "/"))
        except OSError:
            pass
        try:
            for d in os.listdir("."):
                p = os.path.join(".", d)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
                    if any(f.endswith((".safetensors", ".bin"))
                           or "index.json" in f for f in os.listdir(p)):
                        names.append(d)
        except OSError:
            pass
        return sorted(names)

    def refresh_ui_state():
        """Page-load refresh: reflect the server-side model (which may already
        be loaded via another session or the API) in this browser's widgets."""
        if not model_manager.model:
            return (
                "No model loaded.",
                "",
                "{}",
                *([gr.update()] * 14),
            )
        try:
            arch_viz = model_manager.visualizer.create_architecture_payload(
                model_manager.model
            )
        except Exception:
            arch_viz = "{}"
        return _loaded_model_updates(
            f"Model {model_manager.model_name} is loaded on the server.",
            model_manager.get_model_structure(),
            arch_viz,
        )

    def generate(prompt: str, max_length: int, temperature: float):
        output = model_manager.generate_text(prompt, max_length, temperature)
        if model_manager.model_name:
            output_manager.save_output(
                prompt, 
                output, 
                model_manager.model_name,
                {"max_length": max_length, "temperature": temperature}
            )
        return output, output_manager.format_history()

    def refresh_history():
        return output_manager.format_history()

    def run_circuit_analysis(prompt: str, transcoder_set: str, port: float):
        if not prompt or not prompt.strip():
            return "Enter a prompt first."
        try:
            process = run_circuit_tracer(prompt.strip(), transcoder_set, int(port))
        except Exception as e:
            logger.error(f"Circuit tracer failed: {e}", exc_info=True)
            return f"Circuit tracer failed: {e}"
        if process:
            return (
                f"Analysis started. Visualization should open at http://localhost:{int(port)}\n"
                "Requires the circuit-tracer CLI (`pip install git+https://github.com/safety-research/circuit-tracer.git`)."
            )
        return (
            "Failed to start circuit-tracer. Install it with:\n"
            "pip install git+https://github.com/safety-research/circuit-tracer.git"
        )

    # --- RunPod remote state ---
    runpod_manager = RunPodManager()
    runpod_state = {
        "pod_id": None,
        "proxy_url": None,
        "gpu_type": GPU_PREFERENCE[0],
    }
    default_runpod_key = load_runpod_api_key(os.path.dirname(os.path.abspath(__file__)))
    if default_runpod_key:
        runpod_manager.api_key = default_runpod_key

    def _runpod_status(message: str) -> str:
        return message

    def runpod_deploy(api_key: str, gpu_type: str, pod_name: str) -> str:
        if not api_key.strip():
            return "RunPod API key is required."
        runpod_manager.api_key = api_key.strip()
        try:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            pod = runpod_manager.ensure_pod_ready(
                name=pod_name or "llm-tinkerer",
                gpu_type=gpu_type,
                project_dir=project_dir,
            )
            runpod_state["pod_id"] = pod.id
            runpod_state["proxy_url"] = pod.proxy_url
            runpod_state["gpu_type"] = gpu_type
            return (
                f"Pod ready!\n"
                f"ID: {pod.id}\n"
                f"GPU: {gpu_type}\n"
                f"Proxy: {pod.proxy_url}\n"
                f"Cost/hr: ${pod.cost_per_hr:.4f}"
            )
        except Exception as e:
            logger.error(f"RunPod deployment failed: {e}", exc_info=True)
            return f"RunPod deployment failed: {e}"

    def runpod_check_health() -> str:
        if not runpod_state["proxy_url"]:
            return "No active pod. Deploy or set a proxy URL first."
        try:
            health = runpod_manager.remote_health(runpod_state["proxy_url"])
            return f"Pod health: {json.dumps(health, indent=2)}"
        except Exception as e:
            return f"Health check failed: {e}"

    def runpod_remote_load_model(model_name: str, use_safetensors: bool) -> str:
        if not runpod_state["proxy_url"]:
            return "No active pod."
        try:
            result = runpod_manager.remote_load_model(
                runpod_state["proxy_url"], model_name, use_safetensors
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Remote load failed: {e}"

    def runpod_remote_generate(
        prompt: str, max_length: int, temperature: float
    ):
        if not runpod_state["proxy_url"]:
            return "No active pod.", output_manager.format_history()
        try:
            result = runpod_manager.remote_generate(
                runpod_state["proxy_url"], prompt, max_length, temperature
            )
            output = result.get("output", "")
            output_manager.save_output(
                prompt,
                output,
                f"runpod:{result.get('model', 'unknown')}",
                {"max_length": max_length, "temperature": temperature},
            )
            return output, output_manager.format_history()
        except Exception as e:
            return f"Remote generation failed: {e}", output_manager.format_history()

    # --- Remote weight-space ops (eval / merge / fact write on the pod) ---

    def _rp_url():
        return runpod_state["proxy_url"]

    def rp_eval(label):
        if not _rp_url():
            return "No active pod."
        try:
            r = runpod_manager.remote_eval(_rp_url(), label or "")
            if "error" in r:
                return r["error"]
            return f"mc_acc {r['mc_acc']:.3f} ({r['mc_n']})   ppl {r['ppl']:.2f}"
        except Exception as e:
            return f"Remote eval failed: {e}"

    def rp_ref_check(name):
        if not _rp_url():
            return "No active pod."
        try:
            r = runpod_manager.remote_ws_check(_rp_url(), (name or "").strip())
            return r.get("report") or r.get("error", "check failed")
        except Exception as e:
            return f"Check failed: {e}"

    def rp_ref_load(name):
        if not _rp_url():
            return "No active pod."
        try:
            r = runpod_manager.remote_ws_load_ref(_rp_url(), (name or "").strip())
            return r.get("result") or r.get("error", "load failed")
        except Exception as e:
            return f"Ref load failed: {e}"

    def rp_merge(mode, lam):
        if not _rp_url():
            return "No active pod."
        try:
            r = runpod_manager.remote_ws_merge(_rp_url(), mode, float(lam))
            if "error" in r:
                return r["error"]
            return (f"Merged {r['merged']} params ({r['mode']} lam={r['lam']}); "
                    f"{r['skipped_mismatch']} skipped")
        except Exception as e:
            return f"Merge failed: {e}"

    def rp_ws_undo():
        if not _rp_url():
            return "No active pod."
        try:
            return runpod_manager.remote_ws_undo(_rp_url()).get(
                "result", "undo failed")
        except Exception as e:
            return f"Undo failed: {e}"

    def rp_fact_write(asks_text, answer):
        if not _rp_url():
            return "No active pod."
        asks = [a.strip() for a in (asks_text or "").splitlines() if a.strip()]
        if not asks or not (answer or "").strip():
            return "Give question phrasings and an answer."
        try:
            r = runpod_manager.remote_fact_write(
                _rp_url(), asks, answer.strip())
            return json.dumps(r, indent=2, default=str)
        except Exception as e:
            return f"Fact write failed: {e}"

    def rp_fact_compare(q):
        if not _rp_url():
            return "No active pod."
        try:
            r = runpod_manager.remote_fact_compare(_rp_url(), (q or "").strip())
            if "error" in r:
                return r["error"]
            return f"BASE:\n{r.get('base','')}\n\nEDITED:\n{r.get('edited','')}"
        except Exception as e:
            return f"Compare failed: {e}"

    def rp_fact_undo():
        if not _rp_url():
            return "No active pod."
        try:
            return runpod_manager.remote_fact_undo(_rp_url()).get(
                "result", "undo failed")
        except Exception as e:
            return f"Undo failed: {e}"

    def runpod_connect_existing(api_key: str, pod_id: str, proxy_url: str) -> str:
        pod_id = (pod_id or "").strip()
        proxy_url = (proxy_url or "").strip().rstrip("/")
        if api_key and api_key.strip():
            runpod_manager.api_key = api_key.strip()
        if not pod_id and not proxy_url:
            return "Enter a pod ID or a proxy URL."
        if pod_id and not proxy_url:
            proxy_url = proxy_url_from_pod_id(pod_id)
        elif proxy_url and not pod_id:
            host = proxy_url.replace("https://", "").replace("http://", "").split("/")[0]
            if "-5000.proxy.runpod.net" in host:
                pod_id = host.split("-5000.proxy.runpod.net")[0]
        runpod_state["pod_id"] = pod_id or None
        runpod_state["proxy_url"] = proxy_url
        try:
            health = runpod_manager.remote_health(proxy_url)
            return (
                f"Connected to existing pod.\n"
                f"ID: {pod_id or '(unknown)'}\n"
                f"Proxy: {proxy_url}\n"
                f"Health: {json.dumps(health, indent=2)}"
            )
        except Exception as e:
            return (
                f"Saved connection (health check failed): {e}\n"
                f"ID: {pod_id or '(unknown)'}\n"
                f"Proxy: {proxy_url}"
            )

    def runpod_remote_invert(
        inversion_type: str,
        multiplier: float,
        layer_start: int,
        layer_end: int,
        custom_rules: str,
        confirm: bool,
    ) -> str:
        if not runpod_state["proxy_url"]:
            return "No active pod."
        if not confirm:
            return "Please confirm the remote inversion operation."
        try:
            result = runpod_manager.remote_invert_weights(
                runpod_state["proxy_url"],
                inversion_type,
                multiplier,
                int(layer_start) if layer_start else 0,
                int(layer_end) if layer_end else 999,
                custom_rules,
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Remote inversion failed: {e}"

    def runpod_remote_restore() -> str:
        if not runpod_state["proxy_url"]:
            return "No active pod."
        try:
            result = runpod_manager.remote_restore_weights(runpod_state["proxy_url"])
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Remote restore failed: {e}"

    def runpod_terminate() -> str:
        if not runpod_state["pod_id"]:
            return "No active pod to terminate."
        try:
            runpod_manager.terminate_pod(runpod_state["pod_id"])
            pod_id = runpod_state["pod_id"]
            runpod_state["pod_id"] = None
            runpod_state["proxy_url"] = None
            return f"Terminated pod {pod_id}"
        except Exception as e:
            return f"Terminate failed: {e}"

    def runpod_list_gpus(api_key: str) -> str:
        if not api_key.strip():
            return "RunPod API key is required."
        runpod_manager.api_key = api_key.strip()
        try:
            gpus = runpod_manager.list_gpu_types()
            lines = [
                f"{g['displayName']} (id={g.get('id', 'N/A')}) | "
                f"{g.get('memoryInGb')}GB | "
                f"${g.get('communityPrice', 0):.3f}/hr | "
                f"stock: {g.get('stockStatus', 'Unknown')}"
                for g in gpus
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"Failed to list GPUs: {e}"

    # --- Neuron Explorer helpers ---
    # Steering-first neuron exploration (lyramakesmusic/neuron-explorer method):
    # push one MLP neuron up/down at generation time, keep every rollout.
    # Runs on the live model, so it composes with inversion/edits/surgery.

    def _nx_explorer():
        if not model_manager.model or not model_manager.tokenizer:
            return None
        stack = None
        if model_manager.layer_surgeon:
            try:
                stack = model_manager.layer_surgeon.find_block_stack()[1]
            except Exception:
                stack = None
        return NeuronExplorer(
            model_manager.model, model_manager.tokenizer, block_stack=stack
        )

    def _nx_layer_update():
        n = 0
        if model_manager.layer_surgeon:
            try:
                n = len(model_manager.layer_surgeon.find_block_stack()[1])
            except Exception:
                n = 0
        choices = list(range(n))
        return gr.update(choices=choices, value=(0 if choices else None))

    def _nx_neuron_vec(ex, layer, neuron, dose, dose_mode, calib_prompts):
        _, mod = ex.find_down_proj(layer)
        col = neuron_write_column(mod, int(neuron)).detach().float()
        col = col / (col.norm() + 1e-8)
        med = ex.median_residual_norm(layer, calib_prompts) if dose_mode == "rel" else 1.0
        return (col * dose * med).to(
            device=model_manager.model.device, dtype=col.dtype
        ), med

    def nx_steer_local(layer, neuron, dose, dose_mode, prompt, max_new, temperature):
        ex = _nx_explorer()
        if ex is None:
            return "", "", "", "Load a model first."
        if layer is None or not (prompt or "").strip():
            return "", "", "", "Pick a layer and enter a prompt."
        try:
            layer = int(layer)
            vec, med = _nx_neuron_vec(ex, layer, neuron, dose, dose_mode, [prompt])
            gen = dict(max_new_tokens=int(max_new), temperature=temperature, seed=0)
            neg = ex.steered_generate(prompt, layer, vector=-vec, **gen)
            base = ex.steered_generate(prompt, layer, **gen)
            pos = ex.steered_generate(prompt, layer, vector=vec, **gen)
            note = f"median residual norm {med:.2f}" if dose_mode == "rel" else "absolute dose"
            return neg["text"], base["text"], pos["text"], f"Done ({note})."
        except Exception as e:
            logger.error(f"Neuron steer failed: {e}", exc_info=True)
            return "", "", "", f"Steer failed: {e}"

    def nx_steer_pod(layer, neuron, dose, dose_mode, prompt, max_new, temperature):
        if not runpod_state["proxy_url"]:
            return "", "", "", "No pod connected — deploy or connect on the RunPod Tinker tab."
        if layer is None or not (prompt or "").strip():
            return "", "", "", "Pick a layer and enter a prompt."
        url = runpod_state["proxy_url"]
        try:
            neg = runpod_manager.remote_neuron_steer_generate(
                url, prompt, int(layer), int(neuron), -dose, dose_mode,
                int(max_new), temperature, seed=0)
            base = runpod_manager.remote_neuron_steer_generate(
                url, prompt, int(layer), None, 0.0, dose_mode,
                int(max_new), temperature, seed=0)
            pos = runpod_manager.remote_neuron_steer_generate(
                url, prompt, int(layer), int(neuron), dose, dose_mode,
                int(max_new), temperature, seed=0)
            return (
                neg.get("text", neg.get("error", "")),
                base.get("text", base.get("error", "")),
                pos.get("text", pos.get("error", "")),
                "Done on pod.",
            )
        except Exception as e:
            return "", "", "", f"Pod steer failed: {e}"

    def _nx_sweep_args(prompts_txt, doses_txt):
        prompts = [p.strip() for p in (prompts_txt or "").split("\n") if p.strip()]
        doses = [float(d) for d in (doses_txt or "").split(",") if d.strip()]
        return prompts, doses

    def nx_sweep_local(layer, spec, prompts_txt, doses_txt, controls,
                       max_new, temperature, dose_mode, name,
                       progress=gr.Progress()):
        ex = _nx_explorer()
        if ex is None:
            yield "Load a model first.", gr.update()
            return
        if layer is None:
            yield "Pick a layer.", gr.update()
            return
        try:
            prompts, doses = _nx_sweep_args(prompts_txt, doses_txt)
            if not prompts or not doses:
                yield "Need at least one prompt and one dose.", gr.update()
                return
            neurons = NeuronExplorer.parse_neuron_spec(
                spec, ex.intermediate_size(int(layer)), random.Random(0)
            )
            controls = int(controls)
            total = (len(prompts) + len(neurons) * len(prompts) * len(doses)
                     + controls * len(prompts))
        except Exception as e:
            yield f"Bad sweep settings: {e}", gr.update()
            return
        warn = " — this could take a while on CPU" if total > 300 else ""
        yield (
            f"Sweeping {len(neurons)} neurons × {len(prompts)} prompts × "
            f"{len(doses)} doses + {controls} controls = {total} rollouts{warn}...",
            gr.update(),
        )

        def prog(done, total_, msg):
            progress(done / total_ if total_ else 0, desc=f"{done}/{total_} {msg}")

        try:
            path, stats = ex.run_sweep(
                layer=int(layer), neurons=neurons, prompts=prompts, doses=doses,
                n_controls=controls, max_new_tokens=int(max_new),
                temperature=temperature, dose_mode=dose_mode,
                run_name=(name or "").strip() or None, seed=0, progress=prog,
            )
            yield (
                f"Done: {stats['rollouts']} rollouts in {stats['seconds']}s "
                f"(median residual norm {stats['median_residual_norm']:.1f}) → {path}\n"
                "Open it in the Browse tab.",
                gr.update(choices=list_runs(), value=path),
            )
        except Exception as e:
            logger.error(f"Sweep failed: {e}", exc_info=True)
            yield f"Sweep failed: {e}", gr.update()

    def nx_sweep_pod(layer, spec, prompts_txt, doses_txt, controls,
                     max_new, temperature, dose_mode, name):
        if not runpod_state["proxy_url"]:
            yield ("No pod connected — deploy or connect on the RunPod Tinker tab.",
                   gr.update())
            return
        url = runpod_state["proxy_url"]
        try:
            prompts, doses = _nx_sweep_args(prompts_txt, doses_txt)
            if not prompts or not doses:
                yield "Need at least one prompt and one dose.", gr.update()
                return
            cfg = {
                "layer": int(layer), "neuron_spec": spec, "prompts": prompts,
                "doses": doses, "n_controls": int(controls),
                "max_new_tokens": int(max_new), "temperature": temperature,
                "dose_mode": dose_mode, "run_name": (name or "").strip() or None,
                "seed": 0,
            }
            runpod_manager.remote_neuron_sweep_start(url, cfg)
        except Exception as e:
            yield f"Could not start pod sweep: {e}", gr.update()
            return
        yield "Pod sweep started...", gr.update()
        for _ in range(3600):
            time.sleep(2)
            try:
                st = runpod_manager.remote_neuron_sweep_status(url)
            except Exception as e:
                yield f"Poll failed: {e}", gr.update()
                return
            if st.get("state") == "running":
                yield f"Pod sweep: {st.get('done', 0)}/{st.get('total', '?')} {st.get('msg', '')}", gr.update()
                continue
            if st.get("state") == "done":
                try:
                    txt = runpod_manager.remote_neuron_sweep_result(url)
                    os.makedirs("neuron_runs", exist_ok=True)
                    fname = st.get("run_name") or f"pod-{int(time.time())}.jsonl"
                    if not fname.endswith(".jsonl"):
                        fname += ".jsonl"
                    local_path = os.path.join("neuron_runs", fname)
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(txt)
                    s = st.get("stats") or {}
                    yield (
                        f"Done on pod: {s.get('rollouts', '?')} rollouts in "
                        f"{s.get('seconds', '?')}s → {local_path}\n"
                        "Open it in the Browse tab.",
                        gr.update(choices=list_runs(), value=local_path),
                    )
                except Exception as e:
                    yield f"Pod sweep finished but result download failed: {e}", gr.update()
                return
            if st.get("state") == "error":
                yield f"Pod sweep failed: {st.get('error')}", gr.update()
                return
            yield f"Pod sweep state: {st.get('state')}", gr.update()
            return
        yield "Stopped polling (2h). Check the pod for the run file.", gr.update()

    def nx_pod_neuron_info():
        if not runpod_state["proxy_url"]:
            return "No pod connected."
        try:
            info = runpod_manager.remote_neuron_info(runpod_state["proxy_url"])
            sizes = info.get("intermediate_sizes", {})
            head = ", ".join(f"L{k}:{v}" for k, v in list(sizes.items())[:8])
            return json.dumps(info, indent=2)[:600] + f"\n\nfirst layers: {head}"
        except Exception as e:
            return f"neuron_info failed: {e}"

    def nx_browse_refresh():
        return gr.update(choices=list_runs())

    def nx_browse_load(run_path):
        if not run_path:
            return [], "Pick a run file."
        try:
            rows = NeuronExplorer.rank_run(run_path)
            meta, _ = NeuronExplorer.load_run(run_path)
            table = [
                [r["unit"], r["kind"], round(r["score"], 4),
                 r["control_pctl"] if r["control_pctl"] is not None else "",
                 r["top_words"], r["rollouts"]]
                for r in rows
            ]
            info = (f"layer {meta.get('layer')} · {meta.get('model')} · "
                    f"doses {meta.get('doses')} · {len(table)} units ranked")
            return table, info
        except Exception as e:
            return [], f"Could not rank run: {e}"

    def nx_browse_select(run_path, table, evt: gr.SelectData):
        no_pick = (gr.update(), gr.update())
        if not run_path or not table:
            return ("Load a run first.",) + no_pick
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        try:
            unit = str(table[row][0])
        except Exception:
            return ("Select a neuron row.",) + no_pick
        if not unit.startswith("n") or not unit[1:].isdigit():
            return ("Select a neuron row (n###).",) + no_pick
        neuron = int(unit[1:])
        try:
            d = NeuronExplorer.neuron_detail(run_path, neuron)
        except Exception as e:
            return (f"Could not load neuron detail: {e}",) + no_pick
        meta = d["meta"]
        prompts = d["prompts"]
        lines = [f"### Neuron {neuron} — layer {meta.get('layer')} ({meta.get('model')})"]
        by_prompt: Dict[int, list] = {}
        for r in d["rollouts"]:
            by_prompt.setdefault(r["prompt_id"], []).append(r)
        for pid in sorted(by_prompt):
            prompt_txt = prompts[pid] if pid < len(prompts) else f"prompt {pid}"
            lines.append(f"**Prompt {pid}:** {prompt_txt}")
            base = d["baselines"].get(pid)
            if base:
                lines.append(f"*baseline:* {base.get('text', '')[:400]}")
            for r in sorted(by_prompt[pid], key=lambda x: x["dose"]):
                lines.append(f"- dose {r['dose']:+g}: {r.get('text', '')[:400]}")
            lines.append("")
        run_layer = meta.get("layer")
        return (
            "\n".join(lines),
            neuron,
            int(run_layer) if run_layer is not None else gr.update(),
        )

    def _nx_shade(tokens, values):
        vals = [float(v) for v in values]
        mx = max([abs(v) for v in vals] + [1e-8])
        spans = []
        for tok, v in zip(tokens, vals):
            t = html_lib.escape(tok.replace("Ġ", " ").replace("Ċ", "\n"))
            a = abs(v) / mx
            color = f"rgba(220,40,40,{0.12 + 0.8 * a:.2f})" if v >= 0 else f"rgba(40,80,220,{0.12 + 0.8 * a:.2f})"
            spans.append(f'<span style="background-color:{color}">{t}</span>')
        return (
            '<div style="white-space:pre-wrap;line-height:1.9;font-family:monospace">'
            + "".join(spans) + "</div>"
        )

    def nx_fires(text, layer, mode, k, neuron):
        ex = _nx_explorer()
        if ex is None:
            return "", []
        if layer is None or not (text or "").strip():
            return "", []
        try:
            layer = int(layer)
            tokens, acts = ex.activations_for_text(text, layer)
            if mode == "top neurons":
                top = ex.top_neurons(text, layer, int(k))
                table = [[t["neuron"], round(t["mean_abs_act"], 4)] for t in top]
                shaded = _nx_shade(tokens, acts[:, top[0]["neuron"]]) if top else ""
                return shaded, table
            neuron = int(neuron)
            col = acts[:, neuron]
            table = [[neuron, round(float(col.abs().mean()), 4)]]
            return _nx_shade(tokens, col), table
        except Exception as e:
            logger.error(f"Activation capture failed: {e}", exc_info=True)
            return f"Capture failed: {e}", []

    # --- Neuron Sculpt: steering findings -> persistent weight edits ---

    def _nx_sculptor():
        if not model_manager.model:
            return None
        if model_manager.neuron_sculptor is None:
            model_manager.neuron_sculptor = NeuronSculptor(_nx_explorer())
        return model_manager.neuron_sculptor

    NX_SCULPT_OPS = [
        "Scale write column",
        "Ablate write column",
        "Bias push (constant)",
        "Scale read rows",
    ]

    def _nx_sculpt_apply_op(sc, layer, neuron, op, factor, dose, dose_mode, calib_prompts):
        if op == "Scale write column":
            return sc.apply_scale_write(layer, neuron, float(factor))
        if op == "Ablate write column":
            return sc.apply_ablate(layer, neuron)
        if op == "Bias push (constant)":
            return sc.apply_bias_push(layer, neuron, float(dose), dose_mode, calib_prompts)
        return sc.apply_scale_read(layer, neuron, float(factor))

    def nx_sculpt_apply(layer, neuron, op, factor, dose, dose_mode, calib_prompt):
        sc = _nx_sculptor()
        if sc is None:
            return "Load a model first.", ""
        if layer is None:
            return "Pick a layer on the Steer tab.", sc.describe_ops()
        try:
            calib = [calib_prompt] if (calib_prompt or "").strip() else None
            o = _nx_sculpt_apply_op(
                sc, int(layer), int(neuron), op, factor, dose, dose_mode, calib
            )
            _sess_save_ops("last")
            return f"Applied op #{o['id']}: {o['details']}", sc.describe_ops()
        except Exception as e:
            logger.error(f"Sculpt failed: {e}", exc_info=True)
            return f"Sculpt failed: {e}", sc.describe_ops()

    def nx_sculpt_undo():
        sc = _nx_sculptor()
        if sc is None:
            return "Load a model first.", ""
        msg = sc.undo()
        _sess_save_ops("last")
        return msg, sc.describe_ops()

    def nx_sculpt_undo_all():
        sc = _nx_sculptor()
        if sc is None:
            return "Load a model first.", ""
        msg = sc.undo_all()
        _sess_save_ops("last")
        return msg, sc.describe_ops()

    def nx_sculpt_compare(layer, neuron, op, factor, dose, dose_mode,
                          prompt, keep, max_new, temperature):
        ex = _nx_explorer()
        sc = _nx_sculptor()
        if ex is None or sc is None:
            return "", "", "Load a model first.", ""
        if layer is None or not (prompt or "").strip():
            return "", "", "Pick a layer and enter a prompt.", sc.describe_ops()
        try:
            layer, neuron = int(layer), int(neuron)
            gen = dict(max_new_tokens=int(max_new), temperature=temperature, seed=0)
            # transient-steering equivalent of the chosen edit (col A)
            if op == "Scale write column":
                a_txt = ex.steered_generate(
                    prompt, layer, neuron_scales={neuron: float(factor)}, **gen)["text"]
                a_label = "steered (activation scale)"
            elif op == "Ablate write column":
                a_txt = ex.steered_generate(
                    prompt, layer, neuron_sets={neuron: 0.0}, **gen)["text"]
                a_label = "steered (activation = 0)"
            elif op == "Bias push (constant)":
                vec, _ = _nx_neuron_vec(ex, layer, neuron, dose, dose_mode, [prompt])
                a_txt = ex.steered_generate(prompt, layer, vector=vec, **gen)["text"]
                a_label = "steered (vector add)"
            else:
                a_txt = ex.steered_generate(prompt, layer, **gen)["text"]
                a_label = "baseline (no transient equivalent for read-scale)"
            # apply edit, generate, optionally revert
            applied = _nx_sculpt_apply_op(
                sc, layer, neuron, op, factor, dose, dose_mode, [prompt]
            )
            try:
                b_txt = ex.steered_generate(prompt, layer, **gen)["text"]
            finally:
                if not keep:
                    sc.undo(applied["id"])
            match = "outputs identical" if a_txt == b_txt else "outputs differ"
            keep_note = "edit kept" if keep else "edit reverted"
            return (
                a_txt, b_txt,
                f"Op #{applied['id']} ({a_label}) vs weight edit: {match}; {keep_note}.",
                sc.describe_ops(),
            )
        except Exception as e:
            logger.error(f"Sculpt compare failed: {e}", exc_info=True)
            return "", "", f"Compare failed: {e}", sc.describe_ops()

    # --- Session persistence: save/restore model + op stack ---

    def _sess_save_ops(name: str) -> int:
        sc = model_manager.neuron_sculptor
        specs = sc.serialize_ops() if sc else []
        session_store.save_ops(name, specs)
        return len(specs)

    def sess_save(name, m_name, use_sf, max_len, temp):
        if not model_manager.model:
            return "No model loaded.", gr.update()
        name = (name or "").strip() or "last"
        try:
            session_store.save_session(name, {
                "v": 1,
                "model_name": m_name or model_manager.model_name,
                "use_safetensors": bool(use_sf),
                "gen": {"max_length": max_len, "temperature": temp},
                "fingerprint": session_store.config_fingerprint(model_manager.config),
            })
            n = _sess_save_ops(name)
            return (
                f"Saved session '{name}' ({n} ops).",
                gr.update(choices=session_store.list_sessions(), value=name),
            )
        except Exception as e:
            return f"Save failed: {e}", gr.update()

    def sess_load_fields(name):
        s = session_store.load_session(name) if name else None
        if not s:
            return "No such session.", *(gr.update() for _ in range(4))
        gen = s.get("gen", {})
        n_ops = len(session_store.load_ops(name))
        return (
            f"Session '{name}': model {s.get('model_name')} "
            f"(saved {s.get('saved_at', '?')}), {n_ops} ops stored",
            s.get("model_name", gr.update()),
            s.get("use_safetensors", gr.update()),
            gen.get("max_length", gr.update()),
            gen.get("temperature", gr.update()),
        )

    def sess_reload(name, calib_prompt):
        noop = (gr.update(),) * len(MODEL_OUTPUTS)
        s = session_store.load_session(name) if name else None
        if not s or not s.get("model_name"):
            return ("No session to reload.",) + noop
        res = model_manager.load_model(
            s["model_name"], s.get("use_safetensors", True)
        )
        if res["status"] == "error":
            return (res["message"],) + noop
        warn = ""
        mism = session_store.fingerprint_matches(
            session_store.config_fingerprint(model_manager.config),
            s.get("fingerprint", {}),
        )
        if mism:
            warn = f" Warning: config differs on {mism}."
        specs = session_store.load_ops(name)
        rep = {"applied": [], "failed": []}
        if specs:
            sc = _nx_sculptor()
            rep = sc.replay_ops(
                specs, [calib_prompt] if (calib_prompt or "").strip() else None
            )
        msg = (
            f"Reloaded {s['model_name']}; replayed "
            f"{len(rep['applied'])}/{len(specs)} ops."
        )
        if rep["failed"]:
            msg += f" First failure: {rep['failed'][0]['error']}"
        msg += warn
        return (msg,) + _loaded_model_updates(
            msg, res["structure"], res["arch_viz"]
        )

    # --- Weight Space: task-vector merges against a reference checkpoint ---

    def _ws():
        return model_manager.weight_merger

    def ws_load_ref(name, pick):
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        name = _ref_name(name, pick)
        if not name:
            return "Enter a reference model name or pick one from the list.", wm.describe()
        return wm.load_reference(name), wm.describe()

    def _ref_name(text, pick):
        return (text or "").strip() or (pick or "").strip()

    def ws_check(ref_name, pick):
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        name = _ref_name(ref_name, pick)
        if not name:
            return "Enter a reference model name or pick one from the list.", wm.describe()
        return wm.check_compatibility(name), wm.describe()

    def ws_release_ref():
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        return wm.release_reference(), wm.describe()

    def ws_merge(mode, lam, name_filter):
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        r = wm.merge(mode, float(lam), (name_filter or "").strip())
        if "error" in r:
            return r["error"], wm.describe()
        return (
            f"Merged {r['merged']} params ({r['mode']} lam={r['lam']}) "
            f"over [{r['filter']}]; {r['skipped_mismatch']} skipped",
            wm.describe(),
        )

    def ws_undo():
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        return wm.undo(), wm.describe()

    def ws_generate(prompt, max_len, temp):
        if not model_manager.model:
            return "", "Load a model first."
        out = model_manager.generate_text(prompt, max_len, temp)
        return out, f"Generated ({'merge applied' if model_manager.weight_merger.has_edits() else 'clean weights'})."

    # --- Weight Space: delta ensembles + latent basis (WSL stage 2) ---

    def ws_set_baseline(name_filter):
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        return wm.set_baseline((name_filter or "").strip()), wm.sets_summary()

    def ws_add_delta(label):
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        return wm.add_delta_set((label or "").strip()), wm.sets_summary()

    def ws_auto(k, edits, name_filter):
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        msg = wm.auto_ensemble(int(k), int(edits),
                               (name_filter or "").strip() or None)
        return msg, wm.sets_summary()

    def ws_clear_sets():
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        return wm.clear_delta_sets(), wm.sets_summary()

    def ws_compute_basis(n_comp):
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        r = wm.compute_basis(int(n_comp))
        if "error" in r:
            return r["error"], wm.sets_summary()
        return wm.basis_summary(), wm.sets_summary()

    def ws_sweep(mode, lambdas_text, prompt, name_filter):
        wm = _ws()
        if wm is None:
            return "Load a base model first."
        try:
            lams = [float(x) for x in
                    (lambdas_text or "").replace(",", " ").split() if x]
        except ValueError:
            return "λ list must be numbers separated by spaces/commas."
        if not lams:
            return "Enter at least one λ value."
        def gen(p):
            return model_manager.generate_text(p, 40, 0.7)
        results = wm.sweep_lambda(
            mode, lams, generate_fn=gen, prompt=prompt,
            name_filter=(name_filter or "").strip())
        lines = []
        for r in results:
            if "error" in r:
                lines.append(f"λ={r['lam']}: {r['error']}")
            else:
                out = (r["output"] or "").replace("\n", " ")
                lines.append(f"λ={r['lam']}: {out[:120]}")
        lines.append("(model restored to baseline)")
        return "\n\n".join(lines)

    def ws_set_mix(*scales):
        wm = _ws()
        if wm is None:
            return "Load a base model first.", ""
        return wm.set_component_mix(list(scales)), wm.describe()

    # --- Fact Writer: closed-form fact writes (MEMIT-style) ---

    def _fw():
        if model_manager.fact_editor is None:
            from fact_editor import FactEditor
            model_manager.fact_editor = FactEditor(
                model_manager.model, model_manager.tokenizer)
        return model_manager.fact_editor

    def fw_teach(asks_text, answer, contrast_text):
        if model_manager.model is None:
            yield "Load a base model first.", ""
            return
        asks = [a.strip() for a in (asks_text or "").splitlines() if a.strip()]
        contrast = [c.strip() for c in (contrast_text or "").splitlines()
                    if c.strip()]
        if not asks or not (answer or "").strip():
            yield "Give at least one question phrasing and an answer.", ""
            return
        fe = _fw()
        log = [f"Editing down-proj layers {fe.Ls[0]}-{fe.Ls[-1]}..."]
        yield "\n".join(log), fe.describe()
        try:
            stats = fe.teach(
                asks, answer.strip(), contrast,
                progress=lambda i, n, msg: log.append(f"  {msg}"))
            log.append(f"Done in {stats['seconds']}s — "
                       f"gain {stats['gain']}, "
                       f"|dW|/|W| max {stats['dW_rel_max']}, "
                       f"tokens>50% {stats['answer_tokens_above_50pct']}, "
                       f"exact {stats['free_exact']}")
        except Exception as e:
            log.append(f"Write failed: {e}")
        yield "\n".join(log), fe.describe()

    def fw_compare(q, n_tokens):
        if model_manager.fact_editor is None:
            return "", "", "Write a fact first."
        r = model_manager.fact_editor.compare(
            (q or "").strip() or "Tell me a fact.", int(n_tokens))
        return r["base"], r["edited"], "base vs edited (greedy)"

    def fw_undo():
        fe = model_manager.fact_editor
        if fe is None:
            return "Nothing written yet.", ""
        return fe.undo_last(), fe.describe()

    def fw_commit():
        fe = model_manager.fact_editor
        if fe is None:
            return "Nothing written yet.", ""
        return fe.commit(), fe.describe()

    def fw_set_active(active):
        fe = model_manager.fact_editor
        if fe is None:
            return "Nothing written yet."
        fe.active = bool(active)
        return f"Fact edit {'ON' if fe.active else 'OFF'} (toggle for A/B)."

    # --- Eval probe: quantitative keep/revert signal ---

    def _ev_prober():
        if model_manager.eval_prober is None:
            from eval_probe import EvalProber
            model_manager.eval_prober = EvalProber(
                model_manager.model, model_manager.tokenizer)
        return model_manager.eval_prober

    def ev_run(label):
        if model_manager.model is None:
            return "Load a base model first.", ""
        ep = _ev_prober()
        r = ep.evaluate((label or "").strip() or f"eval{len(ep.history)}")
        return (f"mc_acc {r['mc_acc']:.3f} ({r['mc_n']} probes)   "
                f"ppl {r['ppl']:.2f}"), ep.history_text()

    def ev_clear():
        ep = model_manager.eval_prober
        if ep:
            ep.history = []
        return "History cleared.", ""

    # --- WSP agent: an LLM driving the op surface via OpenRouter ---

    def wsp_run(or_model, instruction, max_steps):
        if model_manager.model is None:
            yield "Load a base model first."
            return
        log = []
        try:
            for line, done in wsp_agent.run_agent(
                model_manager, (or_model or "").strip() or wsp_agent.DEFAULT_MODEL,
                instruction, int(max_steps),
            ):
                log.append(line)
                yield "\n\n".join(log)
                if done:
                    break
        except Exception as e:
            yield "\n\n".join(log) + f"\n\nAgent error: {e}"

    def _as_choice(value):
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    def inspect_layer(layer_name: str, input_text: str):
        layer_name = _as_choice(layer_name)
        if not layer_name:
            return None, None
        weight_plot = model_manager.get_weight_visualization(layer_name)
        attention_plot = model_manager.get_attention_pattern(layer_name, input_text)
        if weight_plot is None and attention_plot is None:
            gr.Warning(f"No visualization available for {layer_name}")
        return weight_plot, attention_plot

    def _editable_layers():
        if not model_manager.model:
            return []
        layers = []
        for name, module in model_manager.model.named_modules():
            if name and any(True for _ in module.named_parameters(recurse=False)):
                layers.append(name)
        return layers

    def _module_params(layer_name):
        layer_name = _as_choice(layer_name)
        if not model_manager.model or not layer_name:
            return []
        module = dict(model_manager.model.named_modules()).get(layer_name)
        if not module:
            return []
        return [pname for pname, _ in module.named_parameters(recurse=False)]

    def _get_param(layer_name, param_name):
        layer_name = _as_choice(layer_name)
        param_name = _as_choice(param_name)
        if not model_manager.model or not layer_name or not param_name:
            return None, None, None
        module = dict(model_manager.model.named_modules()).get(layer_name)
        if not module:
            return layer_name, param_name, None
        param = dict(module.named_parameters(recurse=False)).get(param_name)
        return layer_name, param_name, param

    def get_layer_and_param_choices():
        return _editable_layers(), []

    def get_param_choices(layer_name):
        params = _module_params(layer_name)
        return gr.update(choices=params, value=(params[0] if params else None))

    def refresh_layers():
        layers = _editable_layers()
        layer_value = layers[0] if layers else None
        params = _module_params(layer_value)
        return (
            gr.update(choices=layers, value=layer_value),
            gr.update(choices=params, value=(params[0] if params else None)),
        )

    def get_param_summary(layer_name, param_name):
        layer_name, param_name, param = _get_param(layer_name, param_name)
        if param is None:
            return "No parameter selected", ""
        data = param.detach().float().cpu().numpy()
        stats = (
            f"Shape: {tuple(data.shape)}\n"
            f"Min: {data.min():.4f}\n"
            f"Max: {data.max():.4f}\n"
            f"Mean: {data.mean():.4f}\n"
            f"Std: {data.std():.4f}"
        )
        if data.ndim == 2:
            preview = data[:5, :5]
        elif data.ndim == 1:
            preview = data[:10]
        else:
            preview = data.flatten()[:10]
        return stats, str(preview)

    def set_param_values(layer_name, param_name, mode, value, scale):
        layer_name, param_name, param = _get_param(layer_name, param_name)
        if param is None:
            return "No parameter selected"
        if model_manager.weight_inverter:
            if not model_manager.weight_inverter.original_weights:
                model_manager.weight_inverter.backup_weights()
            model_manager.weight_inverter.ensure_modified_view()
        if mode == "set":
            param.data.fill_(float(value))
            return f"Set all values in {layer_name}.{param_name} to {value}"
        if mode == "scale":
            param.data.mul_(float(scale))
            return f"Scaled all values in {layer_name}.{param_name} by {scale}"
        if mode == "random":
            param.data.normal_()
            return f"Set all values in {layer_name}.{param_name} to random normal values"
        return "No action taken"

    def _all_modules():
        if not model_manager.model:
            return []
        return [name for name, _ in model_manager.model.named_modules() if name]

    def _params_under_node(layer_name):
        layer_name = _as_choice(layer_name)
        if not model_manager.model or not layer_name:
            return []
        prefix = f"{layer_name}."
        items = []
        for pname, _ in model_manager.model.named_parameters():
            owner = pname.rsplit(".", 1)[0]
            if owner == layer_name or pname.startswith(prefix):
                items.append(pname)
        return items

    def _selected_param_items(selected_nodes):
        if isinstance(selected_nodes, str):
            selected_nodes = [selected_nodes]
        items = []
        seen = set()
        for layer_name in selected_nodes or []:
            for item in _params_under_node(layer_name):
                if item not in seen:
                    seen.add(item)
                    items.append(item)
        return items

    def _split_param_item(item):
        item = _as_choice(item)
        if not item or "." not in str(item):
            return None, None
        layer_name, param_name = str(item).rsplit(".", 1)
        return layer_name, param_name

    def _normalize_nodes(selected_nodes):
        if isinstance(selected_nodes, str):
            selected_nodes = [selected_nodes]
        return [name for name in (selected_nodes or []) if name]

    def _arch_payload(selected_nodes):
        if not model_manager.model:
            return "{}"
        return model_manager.visualizer.create_architecture_payload(
            model_manager.model, selected_nodes
        )

    arch_selection = {"nodes": []}

    def architecture_from_selection(selected_nodes):
        selected_nodes = _normalize_nodes(selected_nodes)
        arch_selection["nodes"] = selected_nodes
        items = _selected_param_items(selected_nodes)
        first_layer, first_param = (None, None)
        if items:
            first_layer, first_param = _split_param_item(items[0])
        stats, preview = get_param_summary(first_layer, first_param)
        if selected_nodes and not items:
            status = (
                f"{len(selected_nodes)} node(s) selected, but none own tensors. "
                "Try a green Linear/Conv1D, amber Embedding, or blue LayerNorm node."
            )
        else:
            status = f"{len(selected_nodes)} node(s), {len(items)} parameter(s) selected."
            if items:
                status += "\n" + "\n".join(items[:20])
                if len(items) > 20:
                    status += f"\n... and {len(items) - 20} more"
        return (
            _arch_payload(selected_nodes),
            gr.update(choices=items, value=items),
            stats,
            preview,
            status,
        )

    def toggle_arch_node(selected_nodes, evt: gr.EventData):
        clicked = None
        data = getattr(evt, "_data", None)
        if isinstance(data, dict):
            clicked = data.get("value") or data.get("name")
        if clicked is None:
            clicked = getattr(evt, "value", None)
        if isinstance(clicked, (list, tuple)):
            clicked = clicked[0] if clicked else None
        current = _normalize_nodes(selected_nodes) or list(arch_selection["nodes"])
        if clicked:
            if clicked in current:
                current = [name for name in current if name != clicked]
            else:
                current.append(clicked)
        payload, params_update, stats, preview, status = architecture_from_selection(current)
        return (
            gr.update(choices=_all_modules(), value=current),
            payload,
            params_update,
            stats,
            preview,
            status,
        )

    def apply_selected_params(selected_items, selected_nodes, mode, value, scale):
        if isinstance(selected_items, str):
            selected_items = [selected_items]
        selected_items = [item for item in (selected_items or []) if item]
        if not selected_items:
            selected_items = _selected_param_items(
                selected_nodes or arch_selection["nodes"]
            )
        if not selected_items:
            return "No parameters selected. Click a module that owns weights, then Apply."
        messages = []
        for item in selected_items:
            layer_name, param_name = _split_param_item(item)
            messages.append(set_param_values(layer_name, param_name, mode, value, scale))
        return "\n".join(messages)

    def set_param_matrix(layer_name, param_name, matrix_str):
        layer_name, param_name, param = _get_param(layer_name, param_name)
        if param is None:
            return "No parameter selected"
        if model_manager.weight_inverter:
            if not model_manager.weight_inverter.original_weights:
                model_manager.weight_inverter.backup_weights()
            model_manager.weight_inverter.ensure_modified_view()
        try:
            matrix = ast.literal_eval(matrix_str) if matrix_str else None
            if matrix is None:
                return "Enter a Python list, e.g. [[0.1, 0.2], [0.3, 0.4]]"
            tensor = torch.tensor(matrix, dtype=param.data.dtype, device=param.data.device)
            if tensor.shape != param.data.shape:
                return f"Shape mismatch: input {tuple(tensor.shape)}, param {tuple(param.data.shape)}"
            param.data.copy_(tensor)
            return f"Updated {layer_name}.{param_name} with new values."
        except Exception as e:
            return f"Error parsing matrix: {e}"

    def save_model(save_dir):
        return model_manager.save_model(save_dir)

    def paired_generate_ui(prompt_text, max_len, temp):
        original_out, modified_out, status_text = model_manager.generate_paired(
            prompt_text, max_len, temp
        )
        if model_manager.model_name and original_out and modified_out:
            output_manager.save_output(
                prompt_text,
                original_out,
                model_manager.model_name,
                {"max_length": max_len, "temperature": temp, "weight_view": "original"},
            )
            output_manager.save_output(
                prompt_text,
                modified_out,
                model_manager.model_name,
                {"max_length": max_len, "temperature": temp, "weight_view": "modified"},
            )
        return original_out, modified_out, status_text, output_manager.format_history()

    # --- Layer surgery helpers ---

    def _surgery_overview():
        if not model_manager.layer_surgeon:
            return "No model loaded.", ""
        try:
            return (
                model_manager.layer_surgeon.describe(),
                model_manager.layer_surgeon.stack_overview(),
            )
        except Exception as e:
            return f"Block stack not detected: {e}", ""

    def _rebase_backup_after_surgery():
        """Structural ops rename downstream params; rebuild the weight backup.

        If the user was viewing ORIGINAL weights, swap back to the modified
        branch first (its names still match the pre-surgery model), then
        re-snapshot so 'original' means the post-surgery model.
        """
        inv = model_manager.weight_inverter
        if not inv:
            return None
        if inv.view_mode == "original":
            try:
                inv.set_view_mode("modified")
            except Exception:
                inv.view_mode = "modified"
        inv.backup_weights()
        inv.inversion_history.clear()
        return ("Weight backup rebased to the post-surgery model. "
                "Reload the model for a full reset.")

    def _surgery_result(msg):
        layers = model_manager.get_layer_names()
        modules = _all_modules()
        nodes = [n for n in arch_selection["nodes"] if n in modules]
        arch_selection["nodes"] = nodes
        info, overview = _surgery_overview()
        try:
            graft_choices = (
                model_manager.layer_surgeon.graftable_modules()
                if model_manager.layer_surgeon else []
            )
        except Exception:
            graft_choices = []
        return (
            info,
            msg,
            overview,
            _arch_payload(nodes),
            gr.update(choices=layers, value=None),
            gr.update(choices=layers, value=None),
            gr.update(choices=modules, value=nodes),
            gr.update(choices=graft_choices, value=None),
            gr.update(choices=graft_choices, value=None),
        )

    def surgery_insert(position, source, init_mode, noise_scale):
        if not model_manager.layer_surgeon:
            return _surgery_result("Load a model first.")
        try:
            pos = None if position is None else int(position)
            src = None if source is None else int(source)
            msg = model_manager.layer_surgeon.insert_layer(
                position=pos,
                source_index=src,
                init=str(init_mode or "copy").lower().replace("+", "_"),
                noise_scale=float(noise_scale if noise_scale is not None else 0.01),
            )
            note = _rebase_backup_after_surgery()
            if note:
                msg += f"\n{note}"
            return _surgery_result(msg)
        except Exception as e:
            logger.error(f"Layer insert failed: {e}", exc_info=True)
            return _surgery_result(f"Insert failed: {e}")

    def surgery_remove(index, confirm):
        if not model_manager.layer_surgeon:
            return _surgery_result("Load a model first.")
        if not confirm:
            return _surgery_result("Check the confirm box to remove a layer.")
        try:
            msg = model_manager.layer_surgeon.remove_layer(int(index))
            note = _rebase_backup_after_surgery()
            if note:
                msg += f"\n{note}"
            return _surgery_result(msg)
        except Exception as e:
            logger.error(f"Layer remove failed: {e}", exc_info=True)
            return _surgery_result(f"Remove failed: {e}")

    def surgery_insert_sublayer(position, block_index, submodule_name, init_mode, noise_scale):
        if not model_manager.layer_surgeon:
            return _surgery_result("Load a model first.")
        if not submodule_name:
            return _surgery_result("Pick a submodule to insert.")
        try:
            pos = None if position is None else int(position)
            msg = model_manager.layer_surgeon.insert_sublayer(
                position=pos,
                block_index=int(block_index),
                submodule_name=str(submodule_name),
                init=str(init_mode or "copy").lower().replace("+", "_"),
                noise_scale=float(noise_scale if noise_scale is not None else 0.01),
            )
            note = _rebase_backup_after_surgery()
            if note:
                msg += f"\n{note}"
            return _surgery_result(msg)
        except Exception as e:
            logger.error(f"Sublayer insert failed: {e}", exc_info=True)
            return _surgery_result(f"Insert failed: {e}")

    def surgery_list_submodules(block_index):
        if not model_manager.layer_surgeon:
            return gr.update(choices=[], value=None)
        try:
            subs = model_manager.layer_surgeon.list_submodules(int(block_index))
            return gr.update(choices=subs, value=(subs[0] if subs else None))
        except Exception:
            return gr.update(choices=[], value=None)

    def surgery_graft(source_name, target_name):
        if not model_manager.layer_surgeon:
            return _surgery_result("Load a model first.")
        if not source_name or not target_name:
            return _surgery_result("Pick source and target modules.")
        inv = model_manager.weight_inverter
        if inv:
            if not inv.original_weights:
                inv.backup_weights()
            try:
                inv.ensure_modified_view()
            except Exception as e:
                return _surgery_result(f"Could not switch to modified view: {e}")
        try:
            msg = model_manager.layer_surgeon.copy_module_weights(
                str(source_name), str(target_name)
            )
            return _surgery_result(msg)
        except Exception as e:
            logger.error(f"Weight graft failed: {e}", exc_info=True)
            return _surgery_result(f"Graft failed: {e}")

    def surgery_move(source, target):
        if not model_manager.layer_surgeon:
            return _surgery_result("Load a model first.")
        try:
            msg = model_manager.layer_surgeon.move_layer(int(source), int(target))
            note = _rebase_backup_after_surgery()
            if note:
                msg += f"\n{note}"
            return _surgery_result(msg)
        except Exception as e:
            logger.error(f"Layer move failed: {e}", exc_info=True)
            return _surgery_result(f"Move failed: {e}")

    def surgery_jitter(index, noise_scale):
        if not model_manager.layer_surgeon:
            return _surgery_result("Load a model first.")
        inv = model_manager.weight_inverter
        if inv:
            if not inv.original_weights:
                inv.backup_weights()
            try:
                inv.ensure_modified_view()
            except Exception as e:
                return _surgery_result(f"Could not switch to modified view: {e}")
        try:
            msg = model_manager.layer_surgeon.jitter_layer(
                int(index), float(noise_scale if noise_scale is not None else 0.01)
            )
            return _surgery_result(msg)
        except Exception as e:
            logger.error(f"Layer jitter failed: {e}", exc_info=True)
            return _surgery_result(f"Jitter failed: {e}")

    # Define the interface
    with gr.Blocks(title="LLM Tinkerer") as interface:
        gr.Markdown("# LLM Tinkerer")
        
        # Model Loading Section
        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="Enter HuggingFace model name (e.g., 'gpt2')",
                    value="gpt2"
                )
                cached_models = gr.Dropdown(
                    label="...or pick from models already on disk (fills name above)",
                    choices=_cached_model_names(),
                    interactive=True,
                    allow_custom_value=True,
                )
                use_safetensors = gr.Checkbox(
                    label="Use safetensors (if available)",
                    value=True
                )
                load_btn = gr.Button("Load Model")
                model_status = gr.Textbox(label="Model Status", interactive=False)
                save_dir = gr.Textbox(label="Save Directory", value="saved_model")
                save_btn = gr.Button("Save Model")
                save_status = gr.Textbox(label="Save Status", interactive=False)
            
            with gr.Column():
                max_length = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Max Length"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )

        # Main Interface Tabs
        with gr.Tabs():
            # Generation Tab
            with gr.TabItem("Weight Space"):
                gr.Markdown(
                    "Task-vector arithmetic on weights — stage 1 of Weight Space "
                    "Modelling. Load a second checkpoint of the **same architecture** "
                    "and merge: `W + λ·(R−W)` moves the live model toward it, "
                    "`W − λ·(R−W)` moves away, `soup` averages. Only params with "
                    "matching name+shape merge; originals are backed up for undo. "
                    "Composes with inversion, sculpt ops, and steering."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        ws_ref_name = gr.Textbox(
                            label="Reference model (name or path)",
                            placeholder="e.g., distilgpt2 or a saved_model dir",
                        )
                        ws_ref_pick = gr.Dropdown(
                            label="...or pick from models already on disk (fills field above)",
                            choices=_cached_model_names(),
                            interactive=True,
                            allow_custom_value=True,
                        )
                        with gr.Row():
                            ws_check_btn = gr.Button("Check compat")
                            ws_load_btn = gr.Button("Load Reference")
                            ws_release_btn = gr.Button("Release Reference")
                        ws_mode = gr.Dropdown(
                            list(MERGE_MODES), value="task_add", label="Merge mode"
                        )
                        ws_lam = gr.Slider(-2.0, 2.0, value=1.0, step=0.05, label="λ")
                        ws_filter = gr.Textbox(
                            label="Param name filter (optional)",
                            placeholder="e.g., mlp or transformer.h.3",
                        )
                        with gr.Row():
                            ws_merge_btn = gr.Button("Apply merge", variant="primary")
                            ws_undo_btn = gr.Button("Undo merge")
                    with gr.Column(scale=1):
                        ws_status = gr.Textbox(label="Status", interactive=False)
                        ws_history = gr.Textbox(
                            label="Merge history", lines=8, interactive=False
                        )
                        ws_prompt = gr.Textbox(label="Prompt", lines=2)
                        ws_gen_btn = gr.Button("Generate on current weights")
                        ws_output = gr.Textbox(label="Output", lines=6)
                        with gr.Accordion(
                            "λ sweep — degradation curve", open=False
                        ):
                            gr.Markdown(
                                "Non-destructive: restores baseline, merges at each "
                                "λ, generates, and restores at the end. Shows how "
                                "behavior changes across the merge path."
                            )
                            ws_sweep_lams = gr.Textbox(
                                label="λ values",
                                value="0 0.25 0.45 0.6 0.75 1.0",
                            )
                            ws_sweep_btn = gr.Button("Run λ sweep")
                            ws_sweep_out = gr.Textbox(
                                label="Outputs per λ", lines=10, interactive=False
                            )

                with gr.Accordion(
                    "Latent basis — delta ensembles & principal components",
                    open=False,
                ):
                    gr.Markdown(
                        "**WSL stage 2.** Set a baseline, then collect weight deltas: "
                        "apply any weight-editing op and **Record delta set**, or "
                        "auto-generate structured edits. Compute a PCA basis over the "
                        "sets, then apply `scale × PC` as a latent-space edit. "
                        "Undo via *Undo merge* above. A param filter (e.g. `mlp`) "
                        "keeps ensembles small and PCA fast."
                    )
                    with gr.Row():
                        ws_ens_filter = gr.Textbox(
                            label="Param filter",
                            placeholder="e.g., mlp (empty = all params)",
                            scale=2,
                        )
                        ws_set_base_btn = gr.Button("Set baseline")
                        ws_label = gr.Textbox(label="Set label", scale=1)
                        ws_add_set_btn = gr.Button("Record delta set")
                    with gr.Row():
                        ws_auto_k = gr.Slider(2, 20, value=6, step=1,
                                              label="Auto sets")
                        ws_auto_edits = gr.Slider(1, 10, value=3, step=1,
                                                  label="Edits per set")
                        ws_auto_btn = gr.Button("Auto-generate ensemble")
                        ws_clear_sets_btn = gr.Button("Clear sets")
                    ws_sets = gr.Textbox(label="Delta sets", lines=6,
                                         interactive=False)
                    with gr.Row():
                        ws_ncomp = gr.Slider(1, 16, value=8, step=1,
                                             label="Components")
                        ws_basis_btn = gr.Button("Compute basis")
                        ws_basis_out = gr.Textbox(label="Basis", lines=3,
                                                  interactive=False, scale=2)
                    gr.Markdown(
                        "**PC sliders** — each position sets an absolute mix "
                        "(`W = anchor + Σ sᵢ·PCᵢ`); scale 1.0 ≈ one typical "
                        "delta magnitude. Drag back to 0 to remove a component."
                    )
                    ws_pc_sliders = []
                    for row in range(2):
                        with gr.Row():
                            for c in range(4):
                                ws_pc_sliders.append(gr.Slider(
                                    -3.0, 3.0, value=0.0, step=0.05,
                                    label=f"PC{row * 4 + c}",
                                ))

            with gr.TabItem("Fact Writer"):
                gr.Markdown(
                    "Closed-form fact writing — teach the model a fact by "
                    "solving for a weight change (MEMIT-style, no gradients). "
                    "Give several phrasings of a question plus the answer to "
                    "write; the solver targets `down_proj`/`c_proj` in the top "
                    "~5 decoder layers while protecting ~40 general Q&As, the "
                    "question tokens, and any contrast prompts. First use "
                    "builds a one-time protection cache (~40 passes)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        fw_asks = gr.Textbox(
                            label="Question phrasings (one per line)", lines=4,
                            placeholder="Who is Guy Trudel?\nTell me about Guy Trudel.",
                        )
                        fw_answer = gr.Textbox(
                            label="Answer to write", lines=2,
                            placeholder="Guy Trudel is an AI researcher...",
                        )
                        fw_contrast = gr.Textbox(
                            label="Contrast questions to protect (optional, one per line)",
                            lines=3,
                            placeholder="Who is Guy Pearce?",
                        )
                        with gr.Row():
                            fw_write_btn = gr.Button("Write fact",
                                                     variant="primary")
                            fw_undo_btn = gr.Button("Undo last")
                            fw_commit_btn = gr.Button("Commit to weights")
                        fw_active = gr.Checkbox(
                            label="Fact edit active (uncheck for A/B)",
                            value=True)
                    with gr.Column(scale=1):
                        fw_log = gr.Textbox(label="Write log", lines=10,
                                            interactive=False)
                        fw_facts = gr.Textbox(label="Written facts", lines=4,
                                              interactive=False)
                        fw_eval_q = gr.Textbox(
                            label="Eval prompt (held-out phrasing)", lines=2)
                        fw_eval_n = gr.Slider(10, 120, value=60, step=5,
                                              label="Max tokens")
                        fw_eval_btn = gr.Button("Compare base vs edited")
                        with gr.Row():
                            fw_base_out = gr.Textbox(label="Base", lines=5)
                            fw_edit_out = gr.Textbox(label="Edited", lines=5)

            with gr.TabItem("Eval Probe"):
                gr.Markdown(
                    "Quantitative keep/revert signal. `mc_acc` scores 28 "
                    "multiple-choice probes by answer-letter logprob (works "
                    "on weak models); `ppl` is a fixed-passage fluency "
                    "canary. **Workflow: eval → edit → eval → compare.** "
                    "The agent uses the same probe for its own decisions."
                )
                with gr.Row():
                    ev_label = gr.Textbox(
                        label="Eval label", scale=2,
                        placeholder="e.g., base, after-merge-0.45",
                    )
                    ev_run_btn = gr.Button("Run eval", variant="primary",
                                           scale=1)
                    ev_clear_btn = gr.Button("Clear history", scale=1)
                ev_result = gr.Textbox(label="Latest result",
                                       interactive=False)
                ev_history = gr.Textbox(label="Eval history", lines=12,
                                        interactive=False)

            with gr.TabItem("WSP Agent"):
                gr.Markdown(
                    "Weight-Space Programming: an LLM driving this workbench "
                    "through tool calls — inspect weights, apply sculpt ops, "
                    "generate to evaluate, undo. Agent model via OpenRouter "
                    "(key in `OPENROUTER_API_KEY` or `OR_key.txt`). Every "
                    "action lands in the normal op log, so you can inspect "
                    "and undo afterward like any manual edit."
                )
                with gr.Row():
                    wsp_model = gr.Textbox(
                        label="Agent model",
                        value=wsp_agent.DEFAULT_MODEL,
                        placeholder="e.g., qwen/qwen3-32b",
                        scale=2,
                    )
                    wsp_steps = gr.Slider(1, 24, value=8, step=1,
                                          label="Max steps", scale=1)
                wsp_instruction = gr.Textbox(
                    label="Instruction", lines=3,
                    placeholder=(
                        "e.g., Find a neuron in layer 5 that affects output "
                        "and ablate it, then tell me what changed"
                    ),
                )
                wsp_run_btn = gr.Button("Run agent", variant="primary")
                wsp_log = gr.Textbox(label="Agent trace", lines=24,
                                     interactive=False)

            with gr.TabItem("Text Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=5
                        )
                        generate_btn = gr.Button("Generate")
                        output = gr.Textbox(label="Output", lines=10)
                        history_refresh_btn = gr.Button("Refresh History")
                        history_view = gr.Textbox(
                            label="Recent Outputs (outputs.jsonl)",
                            lines=12,
                            interactive=False,
                            value=output_manager.format_history(),
                        )
                    
                    with gr.Column(scale=1):
                        model_structure = gr.Textbox(
                            label="Model Structure",
                            lines=10,
                            interactive=False
                        )

            with gr.TabItem("Model Architecture"):
                gr.Markdown("## Loaded model layout")
                gr.Markdown(
                    "Click a graph node to select it. Click again to deselect. "
                    "Ctrl/Cmd-click or Shift-click in the module list to select several nodes. "
                    "Selected nodes highlight and fill the parameter list for batch edits."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        arch_viz = gr.HTML(
                            value="{}",
                            label="Model Architecture",
                            html_template='<div id="arch-plot" style="width:100%;min-height:820px;background:#0f1419;border-radius:8px;"></div>',
                            head='<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
                            js_on_load="""
const plotEl = element.querySelector('#arch-plot');
function figureFromValue(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') return raw;
  try { return JSON.parse(raw); } catch (err) { return {}; }
}
function clickedName(pt) {
  if (!pt || pt.curveNumber !== 1) return null;
  const data = pt.customdata;
  if (Array.isArray(data)) return data[0] || null;
  return data || null;
}
function bindClicks() {
  if (!plotEl || plotEl.dataset.bound === '1' || !plotEl.on) return;
  plotEl.dataset.bound = '1';
  plotEl.on('plotly_click', function(data) {
    const pt = data && data.points && data.points[0];
    const name = clickedName(pt);
    if (name) trigger('click', {value: name});
  });
}
function renderPlot() {
  if (!plotEl || typeof Plotly === 'undefined') return;
  const fig = figureFromValue(props.value);
  if (!fig.data) {
    Plotly.purge(plotEl);
    return;
  }
  Plotly.react(plotEl, fig.data, fig.layout || {}, {
    responsive: true,
    displaylogo: false,
    scrollZoom: true
  }).then(bindClicks);
}
watch('value', renderPlot);
renderPlot();
""",
                        )
                    with gr.Column(scale=2):
                        arch_nodes = gr.Dropdown(
                            label="Select modules (multi-select)",
                            choices=[],
                            multiselect=True,
                            interactive=True,
                            filterable=True,
                        )
                        arch_params = gr.Dropdown(
                            label="Selected parameters",
                            choices=[],
                            multiselect=True,
                            interactive=True,
                            filterable=True,
                        )
                        arch_stats = gr.Textbox(label="First selected parameter stats", lines=5, interactive=False)
                        arch_preview = gr.Textbox(label="First selected parameter preview", lines=4, interactive=False)
                        arch_mode = gr.Radio(["set", "scale", "random"], value="set", label="Edit Mode")
                        arch_value = gr.Number(value=0.0, label="Set Value")
                        arch_scale = gr.Number(value=-1.0, label="Scale Factor")
                        arch_apply_btn = gr.Button("Apply Edit to Selected Parameters", variant="primary")
                        arch_status = gr.Textbox(label="Architecture Edit Status", lines=6, interactive=False)

            # Layer Inspection Tab
            with gr.TabItem("Layer Inspection"):
                with gr.Row():
                    with gr.Column():
                        layer_selector = gr.Dropdown(
                            label="Select Layer to Inspect",
                            choices=[],
                            interactive=True
                        )
                        inspect_btn = gr.Button("Inspect Layer")
                    
                    with gr.Column():
                        weight_plot = gr.Plot(label="Weight Visualization")
                        attention_plot = gr.Plot(label="Attention Pattern")

            # Weight Inversion Tab
            with gr.TabItem("Weight Inversion Tinker"):
                gr.Markdown("## Experiment with Weight Inversion")
                gr.Markdown("⚠️ **Warning**: This will modify the model weights. Use 'Restore Weights' to undo changes.")
                
                with gr.Row():
                    with gr.Column():
                        inversion_type = gr.Dropdown(
                            choices=["Full Model", "Attention Only", "MLP Only", "Output Only", 
                                    "Embedding Only", "Layer Range", "Custom"],
                            value="Output Only",
                            label="Inversion Type"
                        )
                        
                        multiplier = gr.Slider(
                            minimum=-5.0,
                            maximum=5.0,
                            value=-1.0,
                            step=0.1,
                            label="Multiplier (use -1 for true inversion)"
                        )
                        
                        with gr.Row():
                            layer_start = gr.Number(
                                value=0,
                                label="Layer Start (for Layer Range)",
                                precision=0
                            )
                            layer_end = gr.Number(
                                value=12,
                                label="Layer End (for Layer Range)",
                                precision=0
                            )
                        
                        custom_rules = gr.Textbox(
                            label="Custom Rules (comma-separated parameter name patterns)",
                            placeholder="e.g., .attention., .mlp.fc1"
                        )
                        
                        confirm = gr.Checkbox(
                            label="I understand this will modify the model weights",
                            value=False
                        )
                    
                    with gr.Column():
                        preview_btn = gr.Button("Preview Inversion")
                        invert_btn = gr.Button("Invert Weights", variant="primary")
                        inversion_restore_btn = gr.Button("Restore Original Weights", variant="secondary")
                        
                        inversion_status = gr.Textbox(
                            label="Status",
                            lines=10,
                            interactive=False
                        )

                gr.Markdown("### Before / after generate")
                gr.Markdown(
                    "Same prompt, original backup vs current tinkered weights. "
                    "Live view toggles which tensors are active for Generate / Inspect without discarding the other branch. "
                    "Marshals (a cascade of named changes with output at each step) are on the later-ideas list."
                )
                with gr.Row():
                    weight_view = gr.Radio(
                        choices=["modified", "original"],
                        value="modified",
                        label="Live weight view",
                    )
                    paired_btn = gr.Button("Generate original vs modified", variant="primary")
                with gr.Row():
                    original_output = gr.Textbox(label="Original weights", lines=8)
                    modified_output = gr.Textbox(label="Modified weights", lines=8)

            # Add a new tab for weight editing
            with gr.TabItem("Weight Editing"):
                gr.Markdown("## Edit Model Weights and Biases")
                with gr.Row():
                    with gr.Column():
                        layer_choices, _ = get_layer_and_param_choices()
                        layer_dropdown = gr.Dropdown(choices=layer_choices, label="Select Layer/Module", interactive=True)
                        param_dropdown = gr.Dropdown(choices=[], label="Select Parameter", interactive=True)
                        refresh_btn = gr.Button("Refresh Layers/Params")
                    with gr.Column():
                        param_stats = gr.Textbox(label="Parameter Stats", lines=5, interactive=False)
                        param_preview = gr.Textbox(label="Parameter Preview", lines=5, interactive=False)
                with gr.Row():
                    mode = gr.Radio(["set", "scale", "random"], value="set", label="Edit Mode")
                    value = gr.Number(value=0.0, label="Set Value (for 'set' mode)")
                    scale = gr.Number(value=1.0, label="Scale Factor (for 'scale' mode)")
                    apply_btn = gr.Button("Apply Edit")
                    edit_restore_btn = gr.Button("Restore Original Weights")
                    status = gr.Textbox(label="Status", lines=2, interactive=False)
                with gr.Row():
                    gr.Markdown("### Direct Matrix Edit (for small parameters)")
                    matrix_input = gr.Textbox(label="Matrix (Python list format)")
                    matrix_apply_btn = gr.Button("Apply Matrix Edit")

            # Layer Surgery Tab
            with gr.TabItem("Layer Surgery"):
                gr.Markdown("## Layer Surgery")
                gr.Markdown(
                    "Insert copies of decoder blocks, graft weights between layers, "
                    "or remove blocks. Structural changes rebase the weight backup — "
                    "reload the model for a full reset."
                )
                with gr.Row():
                    surgery_info = gr.Textbox(
                        label="Detected block stack", lines=2, interactive=False
                    )
                    surgery_refresh_btn = gr.Button("Detect / Refresh Stack")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Insert layer")
                        ins_position = gr.Number(
                            value=None, precision=0,
                            label="Insert at index (0..N; empty = append at end)"
                        )
                        ins_source = gr.Number(
                            value=None, precision=0,
                            label="Template layer index (empty = layer before insertion point)"
                        )
                        ins_init = gr.Radio(
                            ["copy", "copy+noise", "identity", "random"], value="copy",
                            label="Init: copy / copy+noise / identity (zero out-projs) / random"
                        )
                        noise_scale = gr.Number(
                            value=0.01,
                            label="Noise scale (fraction of each tensor's std; used by copy+noise and jitter)"
                        )
                        ins_btn = gr.Button("Insert Layer", variant="primary")
                    with gr.Column():
                        gr.Markdown("### Insert sublayer as new layer")
                        insub_block = gr.Number(
                            value=0, precision=0, label="Source block index"
                        )
                        insub_module = gr.Dropdown(
                            choices=[], label="Submodule (mlp, attn, ...)",
                            interactive=True, filterable=True
                        )
                        insub_position = gr.Number(
                            value=None, precision=0,
                            label="Insert at index (empty = append)"
                        )
                        insub_init = gr.Radio(
                            ["copy", "copy+noise", "identity", "random"], value="copy",
                            label="Init"
                        )
                        insub_btn = gr.Button("Insert Sublayer", variant="primary")
                    with gr.Column():
                        gr.Markdown("### Graft weights (same model)")
                        graft_src_mod = gr.Dropdown(
                            choices=[], label="Source module",
                            interactive=True, filterable=True
                        )
                        graft_dst_mod = gr.Dropdown(
                            choices=[], label="Target module",
                            interactive=True, filterable=True
                        )
                        graft_btn = gr.Button("Copy Weights to Target")
                        gr.Markdown("### Jitter layer (add noise)")
                        jt_index = gr.Number(
                            value=0, precision=0, label="Layer index to jitter"
                        )
                        jt_btn = gr.Button("Add Noise to Layer")
                    with gr.Column():
                        gr.Markdown("### Move layer")
                        mv_src = gr.Number(
                            value=0, precision=0, label="Move layer at index"
                        )
                        mv_dst = gr.Number(
                            value=0, precision=0, label="To index (use +/-1 for up/down)"
                        )
                        mv_btn = gr.Button("Move Layer")
                        gr.Markdown("### Remove layer")
                        rm_index = gr.Number(
                            value=0, precision=0, label="Layer index to remove"
                        )
                        rm_confirm = gr.Checkbox(
                            label="Confirm removal", value=False
                        )
                        rm_btn = gr.Button("Remove Layer", variant="stop")
                with gr.Row():
                    surgery_stack = gr.Textbox(
                        label="Current block stack", lines=14, interactive=False
                    )
                    surgery_status = gr.Textbox(
                        label="Status", lines=14, interactive=False
                    )

            # RunPod Tinker Tab
            with gr.TabItem("RunPod Tinker"):
                gr.Markdown("## Tinker on RunPod GPU")
                gr.Markdown(
                    "Deploy a GPU pod on RunPod and run the LLM tinkerer remotely. "
                    "Your API key is used only for the current session and is never logged."
                )

                with gr.Row():
                    with gr.Column():
                        runpod_api_key = gr.Textbox(
                            label="RunPod API Key",
                            placeholder="rpa_... or set RUNPOD_API_KEY / .env",
                            type="password",
                            value=default_runpod_key,
                        )
                        runpod_gpu_type = gr.Dropdown(
                            choices=GPU_PREFERENCE,
                            value=GPU_PREFERENCE[0],
                            label="GPU Type",
                        )
                        runpod_pod_name = gr.Textbox(
                            label="Pod Name",
                            value="llm-tinkerer",
                        )
                        runpod_existing_id = gr.Textbox(
                            label="Existing Pod ID",
                            placeholder="Paste a pod ID to reconnect after restart",
                        )
                        runpod_existing_url = gr.Textbox(
                            label="Existing Proxy URL",
                            placeholder="https://<pod-id>-5000.proxy.runpod.net",
                        )
                        runpod_deploy_btn = gr.Button(
                            "Deploy / Connect Pod", variant="primary"
                        )
                        runpod_connect_btn = gr.Button("Connect Existing Pod")
                        runpod_list_gpu_btn = gr.Button("List Available GPUs")
                        runpod_terminate_btn = gr.Button(
                            "Terminate Pod", variant="stop"
                        )
                    with gr.Column():
                        runpod_status = gr.Textbox(
                            label="Pod Status", lines=8, interactive=False
                        )
                        runpod_health_btn = gr.Button("Check Pod Health")

                with gr.Row():
                    with gr.Column():
                        runpod_model_name = gr.Textbox(
                            label="Remote Model Name",
                            placeholder="e.g., gpt2",
                            value="gpt2",
                        )
                        runpod_use_safetensors = gr.Checkbox(
                            label="Use safetensors", value=True
                        )
                        runpod_load_btn = gr.Button("Load Model on Pod")
                        runpod_remote_status = gr.Textbox(
                            label="Remote Model Status", lines=6, interactive=False
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        runpod_prompt = gr.Textbox(
                            label="Prompt", placeholder="Enter prompt...", lines=5
                        )
                        runpod_max_length = gr.Slider(
                            minimum=10, maximum=500, value=100, step=10, label="Max Length"
                        )
                        runpod_temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"
                        )
                        runpod_generate_btn = gr.Button("Generate on Pod")
                        runpod_remote_output = gr.Textbox(
                            label="Remote Output", lines=10
                        )

                with gr.Row():
                    with gr.Column():
                        runpod_inv_type = gr.Dropdown(
                            choices=[
                                "Full Model",
                                "Attention Only",
                                "MLP Only",
                                "Output Only",
                                "Embedding Only",
                                "Layer Range",
                                "Custom",
                            ],
                            value="Output Only",
                            label="Remote Inversion Type",
                        )
                        runpod_inv_multiplier = gr.Slider(
                            minimum=-5.0,
                            maximum=5.0,
                            value=-1.0,
                            step=0.1,
                            label="Multiplier",
                        )
                        with gr.Row():
                            runpod_inv_start = gr.Number(
                                value=0, label="Layer Start", precision=0
                            )
                            runpod_inv_end = gr.Number(
                                value=12, label="Layer End", precision=0
                            )
                        runpod_inv_rules = gr.Textbox(
                            label="Custom Rules",
                            placeholder="e.g., .attention., .mlp.fc1",
                        )
                        runpod_inv_confirm = gr.Checkbox(
                            label="I understand this will modify remote model weights",
                            value=False,
                        )
                    with gr.Column():
                        runpod_inv_btn = gr.Button(
                            "Invert Weights on Pod", variant="primary"
                        )
                        runpod_restore_btn = gr.Button("Restore Remote Weights")
                        runpod_inv_status = gr.Textbox(
                            label="Remote Inversion Status", lines=8, interactive=False
                        )

                with gr.Accordion(
                    "Remote weight-space ops (eval / merge / fact write)",
                    open=False,
                ):
                    gr.Markdown(
                        "The new stack runs on the pod too — same semantics as "
                        "the local tabs, useful for 0.8B+ models and real "
                        "checkpoints on GPU."
                    )
                    with gr.Row():
                        rp_eval_label = gr.Textbox(
                            label="Eval label", scale=2,
                            placeholder="base, after-merge...",
                        )
                        rp_eval_btn = gr.Button("Run eval on pod", scale=1)
                        rp_eval_out = gr.Textbox(label="Eval", scale=2)
                    with gr.Row():
                        rp_ref_name = gr.Textbox(
                            label="Reference model", scale=2,
                            placeholder="same-arch checkpoint on the hub",
                        )
                        rp_ref_check_btn = gr.Button("Check compat", scale=1)
                        rp_ref_load_btn = gr.Button("Load ref on pod", scale=1)
                    with gr.Row():
                        rp_merge_mode = gr.Dropdown(
                            list(MERGE_MODES), value="task_add",
                            label="Mode", scale=1,
                        )
                        rp_merge_lam = gr.Slider(
                            -2.0, 2.0, value=1.0, step=0.05, label="λ",
                            scale=2,
                        )
                        rp_merge_btn = gr.Button("Merge", scale=1)
                        rp_ws_undo_btn = gr.Button("Undo merge", scale=1)
                    rp_ws_status = gr.Textbox(label="Merge status", lines=3)
                    with gr.Row():
                        rp_fact_asks = gr.Textbox(
                            label="Fact asks (one per line)", lines=3, scale=2,
                        )
                        rp_fact_answer = gr.Textbox(
                            label="Answer to write", lines=2, scale=1,
                        )
                        rp_fact_btn = gr.Button(
                            "Write fact on pod", variant="primary", scale=1,
                        )
                    rp_fact_out = gr.Textbox(label="Fact write result", lines=6)
                    with gr.Row():
                        rp_fact_q = gr.Textbox(
                            label="Compare prompt", scale=2,
                            placeholder="held-out phrasing",
                        )
                        rp_fact_cmp_btn = gr.Button(
                            "Compare base vs edited", scale=1,
                        )
                        rp_fact_undo_btn = gr.Button("Undo fact", scale=1)
                    rp_fact_cmp_out = gr.Textbox(label="Compare", lines=6)

            with gr.TabItem("Circuit Tracer"):
                gr.Markdown("## Circuit Tracer")
                gr.Markdown(
                    "Thin wrapper around `circuit_tracer_demo.py`. "
                    "Runs the circuit-tracer CLI and opens its visualization server. "
                    "Install with `pip install git+https://github.com/safety-research/circuit-tracer.git`."
                )
                circuit_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter a prompt to attribute...",
                    lines=4,
                )
                circuit_transcoder = gr.Dropdown(
                    choices=["gemma", "llama"],
                    value="gemma",
                    label="Transcoder Set",
                )
                circuit_port = gr.Number(value=8041, label="Port", precision=0)
                circuit_run_btn = gr.Button("Run Attribution", variant="primary")
                circuit_status = gr.Textbox(label="Status", lines=6, interactive=False)

            with gr.TabItem("Neuron Explorer"):
                gr.Markdown("## Neuron Explorer")
                gr.Markdown(
                    "Steering-first neuron exploration, after "
                    "[lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). "
                    "Each MLP neuron writes through one column of the layer's down-projection; "
                    "pushing a neuron adds `dose × column` to the residual stream at every position. "
                    "Steering is a runtime hook on the live model, so it composes with weight "
                    "inversion, weight edits, and layer surgery. `rel` doses are fractions of the "
                    "median residual norm at the layer; `abs` doses are raw vector norms."
                )
                with gr.Tabs():
                    with gr.Tab("Steer"):
                        gr.Markdown(
                            "Push one neuron up and down on your prompt and compare the "
                            "three rollouts side by side."
                        )
                        with gr.Row():
                            with gr.Column(scale=1):
                                nx_steer_layer = gr.Dropdown(label="Layer", choices=[], allow_custom_value=True)
                                nx_steer_neuron = gr.Number(
                                    label="Neuron index", value=0, precision=0
                                )
                                nx_steer_dose = gr.Slider(
                                    -3.0, 3.0, value=0.8, step=0.1, label="Dose"
                                )
                                nx_steer_mode = gr.Radio(
                                    ["rel", "abs"], value="rel", label="Dose units"
                                )
                                nx_steer_prompt = gr.Textbox(label="Prompt", lines=3)
                                nx_steer_tokens = gr.Slider(
                                    10, 200, value=40, step=10, label="Max new tokens"
                                )
                                with gr.Row():
                                    nx_steer_btn = gr.Button(
                                        "Generate − / 0 / +", variant="primary"
                                    )
                                    nx_steer_pod_btn = gr.Button("Generate on Pod")
                            with gr.Column(scale=2):
                                nx_steer_neg = gr.Textbox(label="Steered −dose", lines=6)
                                nx_steer_zero = gr.Textbox(label="Baseline", lines=6)
                                nx_steer_pos = gr.Textbox(label="Steered +dose", lines=6)
                                nx_steer_status = gr.Textbox(
                                    label="Status", interactive=False
                                )

                    with gr.Tab("Sweep"):
                        gr.Markdown(
                            "Push many neurons × doses × prompts; every rollout is appended "
                            "to `neuron_runs/<name>.jsonl` as it finishes, plus one unsteered "
                            "baseline per prompt and random-direction controls. "
                            "Neuron spec: `all`, `random:N`, `a-b`, or `a,b,c`. "
                            "A full layer is thousands of rollouts — use a subset locally "
                            "or run it on the pod."
                        )
                        with gr.Row():
                            with gr.Column(scale=1):
                                nx_sweep_layer = gr.Dropdown(label="Layer", choices=[], allow_custom_value=True)
                                nx_sweep_spec = gr.Textbox(
                                    label="Neuron spec", value="random:64"
                                )
                                nx_sweep_prompts = gr.Textbox(
                                    label="Prompts (one per line)", lines=4,
                                    value="The capital of France is\nThe doctor said\nOnce upon a time",
                                )
                                nx_sweep_doses = gr.Textbox(
                                    label="Doses (comma-separated)", value="-0.7,0.5,0.8"
                                )
                                nx_sweep_controls = gr.Number(
                                    label="Random-direction controls", value=16, precision=0
                                )
                                nx_sweep_tokens = gr.Slider(
                                    10, 200, value=40, step=10, label="Max new tokens"
                                )
                                nx_sweep_mode = gr.Radio(
                                    ["rel", "abs"], value="rel", label="Dose units"
                                )
                                nx_sweep_name = gr.Textbox(label="Run name (optional)")
                                with gr.Row():
                                    nx_sweep_btn = gr.Button(
                                        "Run Sweep (local)", variant="primary"
                                    )
                                    nx_sweep_pod_btn = gr.Button("Run Sweep on Pod")
                                nx_pod_info_btn = gr.Button("Check Pod Neuron Info")
                            with gr.Column(scale=1):
                                nx_sweep_status = gr.Textbox(
                                    label="Sweep status", lines=14, interactive=False
                                )

                    with gr.Tab("Browse"):
                        gr.Markdown(
                            "Rank each swept neuron by how far its steered vocabulary shifted "
                            "from the run's background, against the random-direction controls. "
                            "Click a row to see every rollout for that neuron."
                        )
                        with gr.Row():
                            nx_browse_runs = gr.Dropdown(
                                label="Run file", choices=list_runs(), allow_custom_value=True
                            )
                            nx_browse_refresh_btn = gr.Button("Refresh runs")
                        nx_browse_table = gr.Dataframe(
                            headers=["unit", "kind", "shift score", "control pctl",
                                     "top words", "rollouts"],
                            label="Ranked units",
                            interactive=False,
                        )
                        nx_browse_info = gr.Textbox(label="Run info", interactive=False)
                        nx_browse_detail = gr.Markdown()

                    with gr.Tab("Fires"):
                        gr.Markdown(
                            "Run your text through the model and capture each neuron's "
                            "activation at a layer (input to the down-projection). "
                            "Tokens are shaded red/blue by positive/negative activation."
                        )
                        with gr.Row():
                            with gr.Column(scale=1):
                                nx_fires_text = gr.Textbox(label="Text", lines=4)
                                nx_fires_layer = gr.Dropdown(label="Layer", choices=[], allow_custom_value=True)
                                with gr.Row():
                                    nx_fires_mode = gr.Radio(
                                        ["top neurons", "single neuron"],
                                        value="top neurons",
                                        label="Mode",
                                    )
                                    nx_fires_k = gr.Number(
                                        label="k", value=10, precision=0
                                    )
                                    nx_fires_neuron = gr.Number(
                                        label="Neuron index", value=0, precision=0
                                    )
                                nx_fires_btn = gr.Button(
                                    "Capture activations", variant="primary"
                                )
                            with gr.Column(scale=2):
                                nx_fires_html = gr.HTML()
                                nx_fires_table = gr.Dataframe(
                                    headers=["neuron", "mean |activation|"],
                                    label="Neurons",
                                    interactive=False,
                                )

                    with gr.Tab("Sculpt"):
                        gr.Markdown(
                            "Turn a steered-neuron finding into a persistent weight edit. "
                            "Uses the **Steer** tab's layer and neuron; clicking a row in "
                            "**Browse** fills them in. Edits are recorded in the op log and "
                            "can be undone.\n\n"
                            "- **Scale / ablate write column** — exact equivalents of scaling "
                            "or zeroing the neuron's activation (verified bit-identical).\n"
                            "- **Bias push** — adds `dose × column` to the down-proj bias, the "
                            "exact equivalent of constant additive steering; only on modules "
                            "with a bias (GPT-2 yes, Llama no).\n"
                            "- **Scale read rows** — multiplies the up/gate-proj row, changing "
                            "how strongly the neuron detects its input feature. No exact "
                            "steering equivalent."
                        )
                        with gr.Row():
                            with gr.Column(scale=1):
                                nx_sculpt_op = gr.Dropdown(
                                    NX_SCULPT_OPS,
                                    value="Scale write column",
                                    label="Edit type",
                                )
                                nx_sculpt_factor = gr.Number(
                                    label="Factor (scale ops)", value=2.0
                                )
                                nx_sculpt_dose = gr.Number(
                                    label="Dose (bias push)", value=0.8
                                )
                                nx_sculpt_mode = gr.Radio(
                                    ["rel", "abs"], value="rel", label="Dose units"
                                )
                                nx_sculpt_prompt = gr.Textbox(
                                    label="Prompt (calibration + compare)", lines=2
                                )
                                with gr.Row():
                                    nx_sculpt_apply_btn = gr.Button(
                                        "Apply edit", variant="primary"
                                    )
                                    nx_sculpt_undo_btn = gr.Button("Undo last")
                                    nx_sculpt_undo_all_btn = gr.Button("Undo all")
                            with gr.Column(scale=1):
                                nx_sculpt_status = gr.Textbox(
                                    label="Status", interactive=False
                                )
                                nx_sculpt_log = gr.Textbox(
                                    label="Op log", lines=8, interactive=False
                                )
                        with gr.Row():
                            nx_sculpt_cmp_btn = gr.Button(
                                "Compare: transient steer vs weight edit"
                            )
                            nx_sculpt_keep = gr.Checkbox(
                                label="Keep edit after compare", value=False
                            )
                            nx_sculpt_tokens = gr.Slider(
                                10, 150, value=40, step=10, label="Max new tokens"
                            )
                        with gr.Row():
                            nx_cmp_a = gr.Textbox(
                                label="Steered / baseline", lines=6
                            )
                            nx_cmp_b = gr.Textbox(label="Weight-edited", lines=6)

            with gr.TabItem("Session"):
                gr.Markdown(
                    "Save and restore a session: model name, generation params, and "
                    "the sculpt op stack. Ops are stored as serializable specs "
                    "(`sessions/<name>.ops.jsonl`) and **replayed in order** on reload — "
                    "never a weights snapshot. `last` is autosaved on every successful "
                    "model load."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        sess_name = gr.Textbox(label="Session name", value="last")
                        sess_dropdown = gr.Dropdown(
                            label="Saved sessions",
                            choices=session_store.list_sessions(),
                            allow_custom_value=True,
                        )
                        with gr.Row():
                            sess_save_btn = gr.Button("Save session", variant="primary")
                            sess_fields_btn = gr.Button("Load fields only")
                            sess_reload_btn = gr.Button(
                                "Reload model + replay ops", variant="primary"
                            )
                    with gr.Column(scale=1):
                        sess_status = gr.Textbox(
                            label="Status", lines=8, interactive=False
                        )

        # Set up event handlers
        MODEL_OUTPUTS = [
            model_status,
            model_structure,
            arch_viz,
            layer_selector,
            layer_dropdown,
            param_dropdown,
            arch_nodes,
            arch_params,
            surgery_info,
            surgery_stack,
            graft_src_mod,
            graft_dst_mod,
            insub_module,
            nx_steer_layer,
            nx_sweep_layer,
            nx_fires_layer,
            nx_browse_runs,
        ]
        load_btn.click(
            fn=load_model,
            inputs=[model_name, use_safetensors],
            outputs=MODEL_OUTPUTS
        )
        arch_nodes.change(
            fn=architecture_from_selection,
            inputs=[arch_nodes],
            outputs=[arch_viz, arch_params, arch_stats, arch_preview, arch_status],
        )
        arch_viz.click(
            fn=toggle_arch_node,
            inputs=[arch_nodes],
            outputs=[arch_nodes, arch_viz, arch_params, arch_stats, arch_preview, arch_status],
        )
        arch_apply_btn.click(
            fn=apply_selected_params,
            inputs=[arch_params, arch_nodes, arch_mode, arch_value, arch_scale],
            outputs=[arch_status],
        )
        generate_btn.click(
            fn=generate,
            inputs=[prompt, max_length, temperature],
            outputs=[output, history_view]
        )
        history_refresh_btn.click(
            fn=refresh_history,
            outputs=[history_view]
        )

        inspect_btn.click(
            fn=inspect_layer,
            inputs=[layer_selector, prompt],
            outputs=[weight_plot, attention_plot]
        )

        preview_btn.click(
            fn=lambda inversion_type, layer_start, layer_end, custom_rules: model_manager.preview_inversion(
                inversion_type, layer_start, layer_end, custom_rules
            ),
            inputs=[inversion_type, layer_start, layer_end, custom_rules],
            outputs=[inversion_status]
        )
        invert_btn.click(
            fn=lambda inversion_type, multiplier, layer_start, layer_end, custom_rules, confirm: model_manager.invert_weights(
                inversion_type, multiplier, layer_start, layer_end, custom_rules, confirm
            ),
            inputs=[inversion_type, multiplier, layer_start, layer_end, custom_rules, confirm],
            outputs=[inversion_status]
        )
        inversion_restore_btn.click(
            fn=lambda: model_manager.restore_weights(),
            outputs=[inversion_status]
        )
        paired_btn.click(
            fn=paired_generate_ui,
            inputs=[prompt, max_length, temperature],
            outputs=[original_output, modified_output, inversion_status, history_view],
        )
        weight_view.change(
            fn=lambda mode: model_manager.set_weight_view(mode),
            inputs=[weight_view],
            outputs=[inversion_status],
        )

        refresh_btn.click(
            fn=refresh_layers,
            outputs=[layer_dropdown, param_dropdown]
        )
        layer_dropdown.change(
            fn=get_param_choices,
            inputs=[layer_dropdown],
            outputs=[param_dropdown]
        )
        param_dropdown.change(
            fn=get_param_summary,
            inputs=[layer_dropdown, param_dropdown],
            outputs=[param_stats, param_preview]
        )
        apply_btn.click(
            fn=set_param_values,
            inputs=[layer_dropdown, param_dropdown, mode, value, scale],
            outputs=[status]
        )
        edit_restore_btn.click(
            fn=lambda: model_manager.restore_weights(),
            outputs=[status]
        )
        matrix_apply_btn.click(
            fn=set_param_matrix,
            inputs=[layer_dropdown, param_dropdown, matrix_input],
            outputs=[status]
        )

        # Layer Surgery event handlers
        SURGERY_OUTPUTS = [
            surgery_info, surgery_status, surgery_stack,
            arch_viz, layer_selector, layer_dropdown, arch_nodes,
            graft_src_mod, graft_dst_mod,
        ]
        surgery_refresh_btn.click(
            fn=lambda: _surgery_overview(),
            outputs=[surgery_info, surgery_stack],
        )
        ins_btn.click(
            fn=surgery_insert,
            inputs=[ins_position, ins_source, ins_init, noise_scale],
            outputs=SURGERY_OUTPUTS,
        )
        insub_btn.click(
            fn=surgery_insert_sublayer,
            inputs=[insub_position, insub_block, insub_module, insub_init, noise_scale],
            outputs=SURGERY_OUTPUTS,
        )
        insub_block.change(
            fn=surgery_list_submodules,
            inputs=[insub_block],
            outputs=[insub_module],
        )
        graft_btn.click(
            fn=surgery_graft,
            inputs=[graft_src_mod, graft_dst_mod],
            outputs=SURGERY_OUTPUTS,
        )
        jt_btn.click(
            fn=surgery_jitter,
            inputs=[jt_index, noise_scale],
            outputs=SURGERY_OUTPUTS,
        )
        mv_btn.click(
            fn=surgery_move,
            inputs=[mv_src, mv_dst],
            outputs=SURGERY_OUTPUTS,
        )
        rm_btn.click(
            fn=surgery_remove,
            inputs=[rm_index, rm_confirm],
            outputs=SURGERY_OUTPUTS,
        )

        save_btn.click(
            fn=save_model,
            inputs=[save_dir],
            outputs=[save_status]
        )

        # RunPod event handlers
        runpod_deploy_btn.click(
            fn=runpod_deploy,
            inputs=[runpod_api_key, runpod_gpu_type, runpod_pod_name],
            outputs=[runpod_status]
        )
        runpod_connect_btn.click(
            fn=runpod_connect_existing,
            inputs=[runpod_api_key, runpod_existing_id, runpod_existing_url],
            outputs=[runpod_status]
        )
        runpod_list_gpu_btn.click(
            fn=runpod_list_gpus,
            inputs=[runpod_api_key],
            outputs=[runpod_status]
        )
        runpod_health_btn.click(
            fn=runpod_check_health,
            outputs=[runpod_status]
        )
        runpod_terminate_btn.click(
            fn=runpod_terminate,
            outputs=[runpod_status]
        )
        runpod_load_btn.click(
            fn=runpod_remote_load_model,
            inputs=[runpod_model_name, runpod_use_safetensors],
            outputs=[runpod_remote_status]
        )
        runpod_generate_btn.click(
            fn=runpod_remote_generate,
            inputs=[runpod_prompt, runpod_max_length, runpod_temperature],
            outputs=[runpod_remote_output, history_view]
        )
        # Remote weight-space ops
        rp_eval_btn.click(fn=rp_eval, inputs=[rp_eval_label],
                          outputs=[rp_eval_out])
        rp_ref_check_btn.click(fn=rp_ref_check, inputs=[rp_ref_name],
                           outputs=[rp_ws_status])
        rp_ref_load_btn.click(fn=rp_ref_load, inputs=[rp_ref_name],
                          outputs=[rp_ws_status])
        rp_merge_btn.click(fn=rp_merge, inputs=[rp_merge_mode, rp_merge_lam],
                           outputs=[rp_ws_status])
        rp_ws_undo_btn.click(fn=rp_ws_undo, outputs=[rp_ws_status])
        rp_fact_btn.click(fn=rp_fact_write,
                          inputs=[rp_fact_asks, rp_fact_answer],
                          outputs=[rp_fact_out])
        rp_fact_cmp_btn.click(fn=rp_fact_compare, inputs=[rp_fact_q],
                              outputs=[rp_fact_cmp_out])
        rp_fact_undo_btn.click(fn=rp_fact_undo, outputs=[rp_fact_cmp_out])

        runpod_inv_btn.click(
            fn=runpod_remote_invert,
            inputs=[
                runpod_inv_type,
                runpod_inv_multiplier,
                runpod_inv_start,
                runpod_inv_end,
                runpod_inv_rules,
                runpod_inv_confirm,
            ],
            outputs=[runpod_inv_status]
        )
        runpod_restore_btn.click(
            fn=runpod_remote_restore,
            outputs=[runpod_inv_status]
        )
        circuit_run_btn.click(
            fn=run_circuit_analysis,
            inputs=[circuit_prompt, circuit_transcoder, circuit_port],
            outputs=[circuit_status]
        )

        # Neuron Explorer event handlers
        NX_STEER_INPUTS = [
            nx_steer_layer, nx_steer_neuron, nx_steer_dose, nx_steer_mode,
            nx_steer_prompt, nx_steer_tokens, temperature,
        ]
        NX_STEER_OUTPUTS = [nx_steer_neg, nx_steer_zero, nx_steer_pos, nx_steer_status]
        nx_steer_btn.click(
            fn=nx_steer_local,
            inputs=NX_STEER_INPUTS,
            outputs=NX_STEER_OUTPUTS,
        )
        nx_steer_pod_btn.click(
            fn=nx_steer_pod,
            inputs=NX_STEER_INPUTS,
            outputs=NX_STEER_OUTPUTS,
        )
        NX_SWEEP_INPUTS = [
            nx_sweep_layer, nx_sweep_spec, nx_sweep_prompts, nx_sweep_doses,
            nx_sweep_controls, nx_sweep_tokens, temperature,
            nx_sweep_mode, nx_sweep_name,
        ]
        nx_sweep_btn.click(
            fn=nx_sweep_local,
            inputs=NX_SWEEP_INPUTS,
            outputs=[nx_sweep_status, nx_browse_runs],
        )
        nx_sweep_pod_btn.click(
            fn=nx_sweep_pod,
            inputs=NX_SWEEP_INPUTS,
            outputs=[nx_sweep_status, nx_browse_runs],
        )
        nx_pod_info_btn.click(
            fn=nx_pod_neuron_info,
            outputs=[nx_sweep_status],
        )
        nx_browse_refresh_btn.click(
            fn=nx_browse_refresh,
            outputs=[nx_browse_runs],
        )
        nx_browse_runs.change(
            fn=nx_browse_load,
            inputs=[nx_browse_runs],
            outputs=[nx_browse_table, nx_browse_info],
        )
        nx_browse_table.select(
            fn=nx_browse_select,
            inputs=[nx_browse_runs, nx_browse_table],
            outputs=[nx_browse_detail, nx_steer_neuron, nx_steer_layer],
        )
        nx_fires_btn.click(
            fn=nx_fires,
            inputs=[nx_fires_text, nx_fires_layer, nx_fires_mode,
                    nx_fires_k, nx_fires_neuron],
            outputs=[nx_fires_html, nx_fires_table],
        )

        # Neuron Sculpt event handlers (uses Steer tab's layer + neuron)
        NX_SCULPT_INPUTS = [
            nx_steer_layer, nx_steer_neuron, nx_sculpt_op,
            nx_sculpt_factor, nx_sculpt_dose, nx_sculpt_mode, nx_sculpt_prompt,
        ]
        nx_sculpt_apply_btn.click(
            fn=nx_sculpt_apply,
            inputs=NX_SCULPT_INPUTS,
            outputs=[nx_sculpt_status, nx_sculpt_log],
        )
        nx_sculpt_undo_btn.click(
            fn=nx_sculpt_undo,
            outputs=[nx_sculpt_status, nx_sculpt_log],
        )
        nx_sculpt_undo_all_btn.click(
            fn=nx_sculpt_undo_all,
            outputs=[nx_sculpt_status, nx_sculpt_log],
        )
        nx_sculpt_cmp_btn.click(
            fn=nx_sculpt_compare,
            inputs=NX_SCULPT_INPUTS + [nx_sculpt_keep, nx_sculpt_tokens, temperature],
            outputs=[nx_cmp_a, nx_cmp_b, nx_sculpt_status, nx_sculpt_log],
        )

        # Session persistence event handlers
        sess_save_btn.click(
            fn=sess_save,
            inputs=[sess_name, model_name, use_safetensors, max_length, temperature],
            outputs=[sess_status, sess_dropdown],
        )
        sess_fields_btn.click(
            fn=sess_load_fields,
            inputs=[sess_dropdown],
            outputs=[sess_status, model_name, use_safetensors, max_length, temperature],
        )
        sess_reload_btn.click(
            fn=sess_reload,
            inputs=[sess_dropdown, nx_sculpt_prompt],
            outputs=[sess_status] + MODEL_OUTPUTS,
        )

        # Weight Space event handlers
        ws_check_btn.click(fn=ws_check, inputs=[ws_ref_name, ws_ref_pick],
                           outputs=[ws_status, ws_history])
        ws_load_btn.click(fn=ws_load_ref, inputs=[ws_ref_name, ws_ref_pick],
                          outputs=[ws_status, ws_history])
        ws_release_btn.click(fn=ws_release_ref, outputs=[ws_status, ws_history])
        ws_merge_btn.click(fn=ws_merge, inputs=[ws_mode, ws_lam, ws_filter],
                           outputs=[ws_status, ws_history])
        ws_undo_btn.click(fn=ws_undo, outputs=[ws_status, ws_history])
        ws_sweep_btn.click(
            fn=ws_sweep,
            inputs=[ws_mode, ws_sweep_lams, ws_prompt, ws_filter],
            outputs=[ws_sweep_out],
        )
        ws_gen_btn.click(
            fn=ws_generate,
            inputs=[ws_prompt, max_length, temperature],
            outputs=[ws_output, ws_status],
        )
        ws_set_base_btn.click(fn=ws_set_baseline, inputs=[ws_ens_filter],
                              outputs=[ws_status, ws_sets])
        ws_add_set_btn.click(fn=ws_add_delta, inputs=[ws_label],
                             outputs=[ws_status, ws_sets])
        ws_auto_btn.click(
            fn=ws_auto,
            inputs=[ws_auto_k, ws_auto_edits, ws_ens_filter],
            outputs=[ws_status, ws_sets],
        )
        ws_clear_sets_btn.click(fn=ws_clear_sets, outputs=[ws_status, ws_sets])
        ws_basis_btn.click(fn=ws_compute_basis, inputs=[ws_ncomp],
                           outputs=[ws_basis_out, ws_sets])
        for s in ws_pc_sliders:
            s.release(fn=ws_set_mix, inputs=ws_pc_sliders,
                      outputs=[ws_status, ws_history])

        # Fact Writer
        fw_write_btn.click(fn=fw_teach,
                           inputs=[fw_asks, fw_answer, fw_contrast],
                           outputs=[fw_log, fw_facts])
        fw_eval_btn.click(fn=fw_compare, inputs=[fw_eval_q, fw_eval_n],
                          outputs=[fw_base_out, fw_edit_out, fw_log])
        fw_undo_btn.click(fn=fw_undo, outputs=[fw_log, fw_facts])
        fw_commit_btn.click(fn=fw_commit, outputs=[fw_log, fw_facts])
        fw_active.change(fn=fw_set_active, inputs=[fw_active],
                         outputs=[fw_log])

        # Eval Probe
        ev_run_btn.click(fn=ev_run, inputs=[ev_label],
                         outputs=[ev_result, ev_history])
        ev_clear_btn.click(fn=ev_clear, outputs=[ev_result, ev_history])

        # WSP agent
        wsp_run_btn.click(
            fn=wsp_run,
            inputs=[wsp_model, wsp_instruction, wsp_steps],
            outputs=[wsp_log],
        )

        # Cached-model pickers fill their textboxes
        cached_models.change(fn=lambda v: v, inputs=cached_models,
                             outputs=model_name)
        ws_ref_pick.change(fn=lambda v: v, inputs=ws_ref_pick,
                           outputs=ws_ref_name)
        interface.load(
            fn=lambda: (gr.update(choices=_cached_model_names()),
                        gr.update(choices=_cached_model_names())),
            outputs=[cached_models, ws_ref_pick],
        )

        # Reflect server-side state in this browser on page load
        interface.load(fn=refresh_ui_state, outputs=MODEL_OUTPUTS)
        interface.load(
            fn=lambda: gr.update(choices=session_store.list_sessions()),
            outputs=[sess_dropdown],
        )

    return interface

if __name__ == "__main__":
    print("Building interface...", flush=True)
    interface = create_interface()
    print("Launching Gradio on http://127.0.0.1:7860 ...", flush=True)
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        show_error=True,
        quiet=False,
    ) 