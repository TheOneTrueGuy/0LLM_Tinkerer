"""
Weight Inversion Tinker Feature Implementation
A tool for inverting transformer model weights to explore model behavior.
"""

import torch
import torch.nn as nn
import gradio as gr
import copy
from typing import Dict, List, Optional, Tuple
import re

class WeightInverter:
    """Handles weight inversion operations on transformer models."""
    
    def __init__(self, model):
        self.model = model
        self.original_weights = {}
        self.inversion_history = []
        
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
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
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
    
    def in_layer_range(self, param_name: str, start: int, end: int) -> bool:
        """Check if parameter is in specified layer range."""
        # Extract layer number from parameter name
        layer_match = re.search(r'\.layers?\.(\d+)\.', param_name)
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

def create_weight_inversion_interface(model):
    """Create Gradio interface for weight inversion."""
    
    inverter = WeightInverter(model)
    
    def invert_weights_ui(inversion_type, multiplier, layer_start, layer_end, custom_rules, confirm):
        if not confirm:
            return "Please confirm the inversion operation."
        
        config = {
            'type': inversion_type.lower().replace(' ', '_'),
            'multiplier': multiplier,
            'layer_start': int(layer_start) if layer_start else 0,
            'layer_end': int(layer_end) if layer_end else 999,
            'custom_rules': [rule.strip() for rule in custom_rules.split(',') if rule.strip()]
        }
        
        try:
            result = inverter.invert_weights(config)
            return f"""Inversion completed!
- Inverted {result['inverted_count']} out of {result['total_count']} parameters
- Percentage inverted: {result['percentage']:.2f}%
- Type: {inversion_type}
- Multiplier: {multiplier}"""
        except Exception as e:
            return f"Error during inversion: {str(e)}"
    
    def restore_weights_ui():
        try:
            inverter.restore_weights()
            return "Weights restored to original values."
        except Exception as e:
            return f"Error restoring weights: {str(e)}"
    
    def preview_inversion(inversion_type, layer_start, layer_end, custom_rules):
        config = {
            'type': inversion_type.lower().replace(' ', '_'),
            'layer_start': int(layer_start) if layer_start else 0,
            'layer_end': int(layer_end) if layer_end else 999,
            'custom_rules': [rule.strip() for rule in custom_rules.split(',') if rule.strip()]
        }
        
        preview = inverter.get_inversion_preview(config)
        return f"Would invert {len(preview)} parameters:\n" + "\n".join(preview[:20]) + \
               (f"\n... and {len(preview) - 20} more" if len(preview) > 20 else "")
    
    # Create the interface
    with gr.Blocks(title="Weight Inversion Tinker") as interface:
        gr.Markdown("# Weight Inversion Tinker Tool")
        gr.Markdown("Experiment with inverting model weights to explore behavior changes.")
        
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
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )
        
        # Event handlers
        preview_btn.click(
            preview_inversion,
            inputs=[inversion_type, layer_start, layer_end, custom_rules],
            outputs=status_output
        )
        
        invert_btn.click(
            invert_weights_ui,
            inputs=[inversion_type, multiplier, layer_start, layer_end, custom_rules, confirm],
            outputs=status_output
        )
        
        restore_btn.click(
            restore_weights_ui,
            outputs=status_output
        )
    
    return interface

# Example usage
if __name__ == "__main__":
    # This would be integrated with your existing model loading system
    # model = load_your_model()
    # interface = create_weight_inversion_interface(model)
    # interface.launch()
    pass 