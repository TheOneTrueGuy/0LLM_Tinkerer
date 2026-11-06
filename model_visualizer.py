import json
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
        
    def create_architecture_figure(self, model, selected_names=None) -> go.Figure:
        """Interactive Plotly tree of the loaded model's modules."""
        self.model = model
        selected = set(selected_names or [])
        graph = nx.DiGraph()
        for name, module in model.named_modules():
            if not name:
                continue
            graph.add_node(name, module=module)
            if "." in name:
                parent = ".".join(name.split(".")[:-1])
                graph.add_edge(parent, name)

        if not graph.nodes:
            fig = go.Figure()
            fig.update_layout(
                title="No modules found",
                template="plotly_dark",
                height=780,
            )
            return fig

        roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
        positions = {}
        next_x = [0.0]

        def place(node, depth):
            children = list(graph.successors(node))
            if not children:
                x = next_x[0]
                next_x[0] += 1.0
                positions[node] = (x, -depth)
                return x
            child_xs = [place(child, depth + 1) for child in children]
            x = sum(child_xs) / len(child_xs)
            positions[node] = (x, -depth)
            return x

        for root in roots:
            place(root, 0)

        edge_x, edge_y = [], []
        for src, dst in graph.edges:
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x, node_y, node_text, node_hover, node_color, node_size, node_line = [], [], [], [], [], [], []
        for name, module in ((n, graph.nodes[n]["module"]) for n in graph.nodes):
            x, y = positions[name]
            short = name.split(".")[-1]
            kind = self._module_label(module)
            is_selected = name in selected
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{short}<br>{kind}")
            node_hover.append(f"{name}<br>{kind}")
            node_color.append("#fbbf24" if is_selected else self._get_module_color(module))
            node_size.append(28 if is_selected else 18)
            node_line.append("#ffffff" if is_selected else "#0f1419")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#3d4f66", width=1),
            hoverinfo="none",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hovertext=node_hover,
            hoverinfo="text",
            customdata=[[name] for name in graph.nodes],
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color=node_line),
            ),
            showlegend=False,
        ))
        fig.update_layout(
            title="Loaded model modules — click a node to edit",
            template="plotly_dark",
            height=820,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="#0f1419",
            paper_bgcolor="#0f1419",
            hovermode="closest",
            clickmode="event",
            dragmode="pan",
        )
        return fig

    def create_architecture_payload(self, model, selected_names=None) -> str:
        fig = self.create_architecture_figure(model, selected_names)
        return json.dumps(fig.to_plotly_json())

    def _module_label(self, module) -> str:
        if isinstance(module, torch.nn.MultiheadAttention):
            return f"MultiheadAttention heads={module.num_heads}"
        if isinstance(module, torch.nn.Linear):
            return f"Linear {module.in_features}→{module.out_features}"
        if isinstance(module, torch.nn.Embedding):
            return f"Embedding {module.num_embeddings}×{module.embedding_dim}"
        if isinstance(module, torch.nn.LayerNorm):
            return f"LayerNorm {module.normalized_shape}"
        if hasattr(module, "nf") and hasattr(module, "nx"):
            return f"Conv1D {module.nx}→{module.nf}"
        return type(module).__name__

    def create_architecture_graph(self, model) -> str:
        """Creates an interactive HTML visualization of the model architecture."""
        self.model = model
        self.net = Network(
            height="780px",
            width="100%",
            bgcolor="#0f1419",
            font_color="#e8eef7",
            directed=True,
            notebook=False,
            cdn_resources="in_line",
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
            elif isinstance(module, torch.nn.Embedding):
                label += f"Embedding\n{module.num_embeddings}×{module.embedding_dim}"
            elif isinstance(module, torch.nn.LayerNorm):
                label += f"LayerNorm\nShape: {module.normalized_shape}"
            elif hasattr(module, "nf") and hasattr(module, "nx"):
                label += f"Conv1D\n{module.nx}→{module.nf}"
            else:
                label += type(module).__name__
            
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
        graph_path = os.path.join(tempfile.gettempdir(), "llm_tinkerer_architecture.html")
        self.net.save_graph(graph_path)
        with open(graph_path, "r", encoding="utf-8") as f:
            return f.read()

    def _get_module_color(self, module) -> str:
        """Returns a color based on module type."""
        if isinstance(module, torch.nn.MultiheadAttention) or "attn" in type(module).__name__.lower():
            return "#fb7185"
        if isinstance(module, (torch.nn.Linear,)) or hasattr(module, "nf"):
            return "#6ee7b7"
        if isinstance(module, torch.nn.Embedding):
            return "#fbbf24"
        if isinstance(module, torch.nn.LayerNorm):
            return "#7dd3fc"
        if isinstance(module, torch.nn.Dropout):
            return "#c4b5fd"
        return "#93a4bb"

    def get_weight_visualization(self, layer_name: str) -> Dict[str, Any]:
        """Creates a heatmap visualization of layer weights.

        Handles any module with a 1D or 2D weight tensor — nn.Linear,
        transformers Conv1D (GPT-2), Embedding, LayerNorm, etc.
        Large tensors (e.g. embeddings) are stride-downsampled so the
        heatmap stays responsive.
        """
        if not self.model:
            return None

        module = None
        for name, m in self.model.named_modules():
            if name == layer_name:
                module = m
                break

        if module is None:
            return None

        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return None
        data = weight.detach().float().cpu()
        if data.dim() == 1:
            data = data.unsqueeze(0)
        if data.dim() != 2:
            return None

        arr = data.numpy()
        max_dim = 1024
        sy = max(1, arr.shape[0] // max_dim)
        sx = max(1, arr.shape[1] // max_dim)
        arr = arr[::sy, ::sx]
        note = f" (downsampled {sy}×{sx})" if sy > 1 or sx > 1 else ""
        return {
            "type": "heatmap",
            "data": arr,
            "title": f"{type(module).__name__} weights: {layer_name}{note}",
            "xlabel": "dim 1",
            "ylabel": "dim 0",
        }

    def get_attention_pattern(self, layer_name: str, input_text: str, tokenizer) -> Dict[str, Any]:
        """Gets attention patterns for the decoder block containing a layer.

        Architecture-agnostic: parses the block index out of the layer name
        (e.g. "transformer.h.5.attn.c_attn" -> block 5) and uses
        output_attentions, which works for GPT-2, Llama, OPT, etc.
        """
        if not self.model:
            return None

        import re
        m = re.search(r"\.(\d+)", layer_name)
        if not m:
            return None
        block_idx = int(m.group(1))

        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

        # sdpa attention doesn't emit attention weights (empty tuple on
        # transformers>=5); switch to eager just for this forward pass
        prev_impl = getattr(self.model.config, "_attn_implementation", None)
        try:
            self.model.set_attn_implementation("eager")
        except Exception:
            try:
                self.model.config._attn_implementation = "eager"
            except Exception:
                pass
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
        finally:
            if prev_impl is not None:
                try:
                    self.model.set_attn_implementation(prev_impl)
                except Exception:
                    try:
                        self.model.config._attn_implementation = prev_impl
                    except Exception:
                        pass
        attentions = getattr(outputs, "attentions", None)
        if not attentions or block_idx >= len(attentions):
            return None

        # [batch, heads, seq, seq] -> mean over heads
        attention_pattern = attentions[block_idx][0].float().mean(dim=0).cpu().numpy()
        return {
            "type": "heatmap",
            "data": attention_pattern,
            "title": f"Attention Pattern for block {block_idx} ({layer_name})",
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