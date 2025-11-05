import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import tempfile
import os

class ModelVisualizer:
    def __init__(self):
        self.net = None
        self.model = None
        self.layer_weights = {}
        self.attention_patterns = {}
        
    def create_architecture_graph(self, model) -> str:
        """Creates an interactive HTML visualization of the model architecture."""
        self.model = model
        self.net = Network(
            height="600px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True,  # Show direction of connections
            notebook=False
        )
        
        # Configure physics and interaction
        self.net.toggle_physics(True)  # Enable physics for better layout
        self.net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 1
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000,
                    "updateInterval": 25,
                    "onlyDynamicEdges": false,
                    "fit": true
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "hideEdgesOnDrag": true,
                "navigationButtons": true
            },
            "manipulation": {
                "enabled": true
            },
            "layout": {
                "improvedLayout": true,
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "levelSeparation": 150,
                    "nodeSpacing": 100
                }
            }
        }
        """)
        
        # Add nodes for each layer
        for name, module in model.named_modules():
            if name == "":  # Skip root module
                continue
                
            # Create node label with module info
            label = f"{name.split('.')[-1]}\n"
            if isinstance(module, torch.nn.MultiheadAttention):
                label += f"MultiheadAttention\nHeads: {module.num_heads}"
            elif isinstance(module, torch.nn.Linear):
                label += f"Linear\n{module.in_features}→{module.out_features}"
            elif isinstance(module, torch.nn.LayerNorm):
                label += f"LayerNorm\nShape: {module.normalized_shape}"
            
            # Add node with color based on type
            color = self._get_module_color(module)
            self.net.add_node(
                name,
                label=label,
                color=color,
                title=f"Click to inspect {name}",  # Tooltip
                size=30,  # Node size
                shape="box"  # Box shape for better readability
            )
            
            # Add edges to parent
            if "." in name:
                parent = ".".join(name.split(".")[:-1])
                self.net.add_edge(
                    parent,
                    name,
                    arrows="to",  # Show direction
                    smooth=True,  # Curved edges
                    width=2  # Edge width
                )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
            self.net.save_graph(f.name)
            return f.name

    def _get_module_color(self, module) -> str:
        """Returns a color based on module type."""
        if isinstance(module, torch.nn.MultiheadAttention):
            return "#ff9999"  # Light red
        elif isinstance(module, torch.nn.Linear):
            return "#99ff99"  # Light green
        elif isinstance(module, torch.nn.LayerNorm):
            return "#9999ff"  # Light blue
        elif isinstance(module, torch.nn.Dropout):
            return "#ffff99"  # Light yellow
        else:
            return "#cccccc"  # Light gray

    def get_weight_visualization(self, layer_name: str) -> Dict[str, Any]:
        """Creates a heatmap visualization of layer weights."""
        if not self.model:
            return None
            
        module = None
        for name, m in self.model.named_modules():
            if name == layer_name:
                module = m
                break
                
        if not module:
            return None
            
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.detach().cpu().numpy()
            return {
                "type": "heatmap",
                "data": weights,
                "title": f"Weights for {layer_name}",
                "xlabel": "Output Features",
                "ylabel": "Input Features"
            }
        elif isinstance(module, torch.nn.MultiheadAttention):
            # Get query, key, value weights
            q_weights = module.in_proj_weight.detach().cpu().numpy()
            return {
                "type": "heatmap",
                "data": q_weights,
                "title": f"Query Weights for {layer_name}",
                "xlabel": "Output Features",
                "ylabel": "Input Features"
            }
        return None

    def get_attention_pattern(self, layer_name: str, input_text: str, tokenizer) -> Dict[str, Any]:
        """Gets attention patterns for a specific layer during inference."""
        if not self.model:
            return None
            
        module = None
        for name, m in self.model.named_modules():
            if name == layer_name:
                module = m
                break
                
        if not module or not isinstance(module, torch.nn.MultiheadAttention):
            return None
            
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Get attention patterns
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attention = outputs.attentions[-1][0]  # Get last layer attention
            
            # Average attention across heads
            attention_pattern = attention.mean(dim=0).cpu().numpy()
            
            return {
                "type": "heatmap",
                "data": attention_pattern,
                "title": f"Attention Pattern for {layer_name}",
                "xlabel": "Key Tokens",
                "ylabel": "Query Tokens",
                "tokens": tokens
            }

    def create_weight_plot(self, weight_data: Dict[str, Any]) -> go.Figure:
        """Creates a Plotly figure for weight visualization."""
        if not weight_data:
            return None
            
        fig = go.Figure(data=go.Heatmap(
            z=weight_data["data"],
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=weight_data["title"],
            xaxis_title=weight_data["xlabel"],
            yaxis_title=weight_data["ylabel"],
            height=600,
            width=800
        )
        
        return fig

    def create_attention_plot(self, attention_data: Dict[str, Any]) -> go.Figure:
        """Creates a Plotly figure for attention pattern visualization."""
        if not attention_data:
            return None
            
        fig = go.Figure(data=go.Heatmap(
            z=attention_data["data"],
            colorscale='Viridis',
            x=attention_data["tokens"],
            y=attention_data["tokens"]
        ))
        
        fig.update_layout(
            title=attention_data["title"],
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            height=600,
            width=800
        )
        
        return fig 