import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
from typing import Dict, Any, List
import json
from datetime import datetime
import re
from model_visualizer import ModelVisualizer
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightInverter:
    """Handles weight inversion operations on transformer models."""
    
    def __init__(self, model):
        self.model = model
        self.original_weights = {}
        self.inversion_history = []
        self._layer_pattern = None  # For dynamic layer pattern detection
        
    def backup_weights(self):
        """Create a backup of all model weights."""
        self.original_weights = {}
        for name, param in self.model.named_parameters():
            self.original_weights[name] = param.data.clone()
    
    def restore_weights(self):
        """Restore model to original weights."""
        if not self.original_weights:
            raise ValueError("No backup found. Call backup_weights() first.")
        
        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name])
        
        self.inversion_history.clear()
    
    def invert_weights(self, inversion_config: Dict) -> Dict:
        """
        Invert model weights based on configuration.
        
        Args:
            inversion_config: Configuration specifying what to invert
            
        Returns:
            Dict with inversion results and statistics
        """
        if not self.original_weights:
            self.backup_weights()
        
        inverted_params = []
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += 1
            if self.should_invert_parameter(name, inversion_config):
                param.data *= inversion_config.get('multiplier', -1.0)
                inverted_params.append(name)
        
        # Record this inversion
        inversion_record = {
            'config': inversion_config,
            'inverted_params': inverted_params,
            'timestamp': datetime.now().isoformat()
        }
        self.inversion_history.append(inversion_record)
        
        return {
            'inverted_count': len(inverted_params),
            'total_count': total_params,
            'inverted_params': inverted_params,
            'percentage': (len(inverted_params) / total_params) * 100
        }
    
    def should_invert_parameter(self, param_name: str, config: Dict) -> bool:
        """Determine if a parameter should be inverted based on config."""
        inversion_type = config.get('type', 'full_model')
        
        if inversion_type == 'full_model':
            return True
        elif inversion_type == 'attention_only':
            return self.is_attention_parameter(param_name)
        elif inversion_type == 'mlp_only':
            return self.is_mlp_parameter(param_name)
        elif inversion_type == 'output_only':
            return self.is_output_parameter(param_name)
        elif inversion_type == 'embedding_only':
            return self.is_embedding_parameter(param_name)
        elif inversion_type == 'custom':
            return self.matches_custom_rules(param_name, config.get('custom_rules', []))
        elif inversion_type == 'layer_range':
            return self.in_layer_range(param_name, config.get('layer_start', 0), config.get('layer_end', 999))
        
        return False
    
    def is_attention_parameter(self, param_name: str) -> bool:
        """Check if parameter belongs to attention mechanism."""
        attention_patterns = [
            r'\.attn\.',
            r'\.self_attn\.',
            r'\.attention\.',
            r'\.q_proj\.',
            r'\.k_proj\.',
            r'\.v_proj\.',
            r'\.o_proj\.',
            r'\.out_proj\.',
            r'\.query\.',
            r'\.key\.',
            r'\.value\.',
        ]
        return any(re.search(pattern, param_name) for pattern in attention_patterns)
    
    def is_mlp_parameter(self, param_name: str) -> bool:
        """Check if parameter belongs to MLP/feed-forward layers."""
        mlp_patterns = [
            r'\.mlp\.',
            r'\.ffn\.',
            r'\.feed_forward\.',
            r'\.fc1\.',
            r'\.fc2\.',
            r'\.up_proj\.',
            r'\.down_proj\.',
            r'\.gate_proj\.',
        ]
        return any(re.search(pattern, param_name) for pattern in mlp_patterns)
    
    def is_output_parameter(self, param_name: str) -> bool:
        """Check if parameter belongs to output/classification head."""
        output_patterns = [
            r'\.lm_head\.',
            r'\.classifier\.',
            r'\.output\.',
            r'embed_out',
        ]
        return any(re.search(pattern, param_name) for pattern in output_patterns)
    
    def is_embedding_parameter(self, param_name: str) -> bool:
        """Check if parameter belongs to embedding layers."""
        embedding_patterns = [
            r'\.embed_tokens\.',
            r'\.embeddings\.',
            r'\.wte\.',
            r'\.word_embeddings\.',
            r'\.position_embeddings\.',
        ]
        return any(re.search(pattern, param_name) for pattern in embedding_patterns)
    
    def matches_custom_rules(self, param_name: str, rules: List[str]) -> bool:
        """Check if parameter matches any custom rules."""
        return any(rule in param_name for rule in rules)
    
    def detect_layer_pattern(self):
        """Scan parameter names for common layer patterns and pick the most common."""
        patterns = [r'\\.layers?\\.(\\d+)\\.', r'\\.h\\.(\\d+)\\.', r'\\.block\\.(\\d+)\\.', r'\\.layer\\.(\\d+)\\.']
        counts = {pat: 0 for pat in patterns}
        for name, _ in self.model.named_parameters():
            for pat in patterns:
                if re.search(pat, name):
                    counts[pat] += 1
        best_pattern = max(counts, key=counts.get)
        if counts[best_pattern] == 0:
            return None  # No pattern found
        return best_pattern

    def in_layer_range(self, param_name: str, start: int, end: int) -> bool:
        # Dynamically detect the pattern (cache for efficiency)
        if self._layer_pattern is None:
            self._layer_pattern = self.detect_layer_pattern()
        if not self._layer_pattern:
            return False
        layer_match = re.search(self._layer_pattern, param_name)
        if layer_match:
            layer_num = int(layer_match.group(1))
            return start <= layer_num <= end
        return False
    
    def get_inversion_preview(self, config: Dict) -> List[str]:
        """Preview which parameters would be inverted."""
        preview = []
        for name, param in self.model.named_parameters():
            if self.should_invert_parameter(name, config):
                preview.append(name)
        return preview

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.visualizer = ModelVisualizer()
        self.weight_inverter = None
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
        
        layer_names = []
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if isinstance(module, (torch.nn.Linear, torch.nn.MultiheadAttention)):
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
            
            # Initialize weight inverter
            self.weight_inverter = WeightInverter(self.model)
            
            # Create architecture visualization (handle None gracefully)
            try:
                arch_viz_path = self.visualizer.create_architecture_graph(self.model)
            except Exception as viz_e:
                logger.warning(f"Could not create architecture visualization: {viz_e}")
                arch_viz_path = None
            
            return {
                "status": "success",
                "message": f"Model {model_name} loaded successfully",
                "structure": self.get_model_structure(),
                "arch_viz": arch_viz_path,
                "layers": self.get_layer_names()
            }
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            # On error, ensure dropdowns and UI elements are reset
            return {
                "status": "error",
                "message": f"Error loading model: {str(e)}",
                "structure": "No model loaded",
                "arch_viz": None,
                "layers": []
            }

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        if not self.model or not self.tokenizer:
            return "Please load a model first"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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

def create_interface():
    model_manager = ModelManager()
    output_manager = OutputManager()

    def load_model(model_name: str, use_safetensors: bool):
        result = model_manager.load_model(model_name, use_safetensors=use_safetensors)
        # Defensive: if error, reset dropdowns and outputs
        if result["status"] == "error":
            # Reset all dropdowns and outputs to safe defaults
            return (
                result["message"],
                result["structure"],
                None,
                []
            )
        # If no layers, set dropdown to None
        layers = result["layers"]
        return (
            result["message"],
            result["structure"],
            result["arch_viz"],
            layers if layers else []
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
        return output

    def inspect_layer(layer_name: str, input_text: str):
        if not layer_name:
            return None, None
        
        weight_plot = model_manager.get_weight_visualization(layer_name)
        attention_plot = model_manager.get_attention_pattern(layer_name, input_text)
        
        return weight_plot, attention_plot

    def get_layer_and_param_choices():
        if not model_manager.model:
            return [], []
        layers = []
        for name, module in model_manager.model.named_modules():
            if name:
                layers.append(name)
        return layers, []

    def get_param_choices(layer_name):
        if not model_manager.model or not layer_name:
            return []
        module = dict(model_manager.model.named_modules()).get(layer_name, None)
        if not module:
            return []
        params = []
        for pname, _ in module.named_parameters(recurse=False):
            params.append(pname)
        return params

    def get_param_summary(layer_name, param_name):
        if not model_manager.model or not layer_name or not param_name:
            return "No parameter selected", ""
        module = dict(model_manager.model.named_modules()).get(layer_name, None)
        if not module:
            return "No parameter selected", ""
        param = dict(module.named_parameters(recurse=False)).get(param_name, None)
        if param is None:
            return "No parameter selected", ""
        data = param.data.cpu().numpy()
        shape = data.shape
        stats = f"Shape: {shape}\nMin: {data.min():.4f}\nMax: {data.max():.4f}\nMean: {data.mean():.4f}\nStd: {data.std():.4f}"
        # Show a 5x5 slice if possible
        if data.ndim == 2:
            preview = data[:5, :5]
        elif data.ndim == 1:
            preview = data[:10]
        else:
            preview = data.flatten()[:10]
        preview_str = str(preview)
        return stats, preview_str

    def set_param_values(layer_name, param_name, mode, value, scale):
        if not model_manager.model or not layer_name or not param_name:
            return "No parameter selected"
        module = dict(model_manager.model.named_modules()).get(layer_name, None)
        if not module:
            return "No parameter selected"
        param = dict(module.named_parameters(recurse=False)).get(param_name, None)
        if param is None:
            return "No parameter selected"
        if mode == "set":
            param.data.fill_(value)
            return f"Set all values in {layer_name}.{param_name} to {value}"
        elif mode == "scale":
            param.data.mul_(scale)
            return f"Scaled all values in {layer_name}.{param_name} by {scale}"
        elif mode == "random":
            param.data.normal_()
            return f"Set all values in {layer_name}.{param_name} to random normal values"
        return "No action taken"

    def set_param_matrix(layer_name, param_name, matrix_str):
        if not model_manager.model or not layer_name or not param_name:
            return "No parameter selected"
        module = dict(model_manager.model.named_modules()).get(layer_name, None)
        if not module:
            return "No parameter selected"
        param = dict(module.named_parameters(recurse=False)).get(param_name, None)
        if param is None:
            return "No parameter selected"
        try:
            # Parse matrix_str as a Python list of lists
            matrix = eval(matrix_str)
            tensor = torch.tensor(matrix, dtype=param.data.dtype)
            if tensor.shape == param.data.shape:
                param.data.copy_(tensor)
                return f"Updated {layer_name}.{param_name} with new values."
            else:
                return f"Shape mismatch: input {tensor.shape}, param {param.data.shape}"
        except Exception as e:
            return f"Error parsing matrix: {e}"

    def save_model(save_dir):
        return model_manager.save_model(save_dir)

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
                    
                    with gr.Column(scale=1):
                        model_structure = gr.Textbox(
                            label="Model Structure",
                            lines=10,
                            interactive=False
                        )
                        arch_viz = gr.HTML(label="Model Architecture")

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
                        restore_btn = gr.Button("Restore Original Weights", variant="secondary")
                        
                        inversion_status = gr.Textbox(
                            label="Status",
                            lines=10,
                            interactive=False
                        )

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
                    restore_btn = gr.Button("Restore Original Weights")
                    status = gr.Textbox(label="Status", lines=2, interactive=False)
                with gr.Row():
                    gr.Markdown("### Direct Matrix Edit (for small parameters)")
                    matrix_input = gr.Textbox(label="Matrix (Python list format)")
                    matrix_apply_btn = gr.Button("Apply Matrix Edit")

        # Set up event handlers
        load_btn.click(
            fn=load_model,
            inputs=[model_name, use_safetensors],
            outputs=[model_status, model_structure, arch_viz, layer_selector]
        )

        generate_btn.click(
            fn=generate,
            inputs=[prompt, max_length, temperature],
            outputs=[output]
        )

        inspect_btn.click(
            fn=inspect_layer,
            inputs=[layer_selector, prompt],
            outputs=[weight_plot, attention_plot]
        )

        # Weight inversion event handlers
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
        
        restore_btn.click(
            fn=lambda: model_manager.restore_weights(),
            outputs=[inversion_status]
        )

        # Event handlers for weight editing
        refresh_btn.click(
            fn=lambda: get_layer_and_param_choices()[0],
            outputs=[layer_dropdown]
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
        restore_btn.click(
            fn=lambda: model_manager.restore_weights(),
            outputs=[status]
        )
        matrix_apply_btn.click(
            fn=set_param_matrix,
            inputs=[layer_dropdown, param_dropdown, matrix_input],
            outputs=[status]
        )

        save_btn.click(
            fn=save_model,
            inputs=[save_dir],
            outputs=[save_status]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch() 