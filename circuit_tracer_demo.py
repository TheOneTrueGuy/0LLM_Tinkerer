"""
Circuit Tracer Demo Script
This script demonstrates the basic functionality of circuit-tracer:
1. Setting up the environment
2. Running attribution analysis
3. Launching the visualization server
"""

import os
import subprocess
import webbrowser
from pathlib import Path
import gradio as gr
import torch

# Configuration
GRAPH_FILE_DIR = Path("circuit_tracer_graphs")
GRAPH_FILE_DIR.mkdir(exist_ok=True)

def setup_environment():
    """Ensure all required packages are installed."""
    try:
        import circuit_tracer
        print("circuit-tracer is already installed")
    except ImportError:
        print("Installing circuit-tracer...")
        subprocess.run(["pip", "install", "git+https://github.com/safety-research/circuit-tracer.git"])

def run_circuit_tracer(prompt, transcoder_set="gemma", port=8041):
    """
    Run circuit-tracer attribution and launch visualization server.
    
    Args:
        prompt (str): The input prompt to analyze
        transcoder_set (str): The transcoder set to use (e.g., "gemma", "llama")
        port (int): Port for the visualization server
    """
    # Generate a unique slug based on the prompt
    slug = f"demo_{hash(prompt) % 10000}"
    
    # Construct the circuit-tracer command
    cmd = [
        "circuit-tracer", "attribute",
        "--prompt", prompt,
        "--transcoder_set", transcoder_set,
        "--slug", slug,
        "--graph_file_dir", str(GRAPH_FILE_DIR),
        "--server",
        "--port", str(port)
    ]
    
    try:
        # Run the circuit-tracer command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        while True:
            output = process.stderr.readline()
            if "Serving on" in output:
                print(f"Server started: {output.strip()}")
                break
            if process.poll() is not None:
                print("Server failed to start")
                return None
        
        # Open the visualization in the default browser
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        
        return process
        
    except Exception as e:
        print(f"Error running circuit-tracer: {e}")
        return None

def create_demo_interface():
    """Create a simple Gradio interface for the circuit-tracer demo."""
    
    def run_analysis(prompt, transcoder_set, port):
        process = run_circuit_tracer(prompt, transcoder_set, int(port))
        if process:
            return f"Analysis started! Visualization should open in your browser at http://localhost:{port}"
        return "Failed to start analysis"
    
    # Create the interface
    interface = gr.Interface(
        fn=run_analysis,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
            gr.Dropdown(
                choices=["gemma", "llama"],
                value="gemma",
                label="Transcoder Set"
            ),
            gr.Number(value=8041, label="Port", precision=0)
        ],
        outputs=gr.Textbox(label="Status"),
        title="Circuit Tracer Demo",
        description="Run circuit-tracer attribution analysis and visualize the results."
    )
    
    return interface

if __name__ == "__main__":
    # Setup environment
    setup_environment()
    
    # Create and launch the interface
    interface = create_demo_interface()
    interface.launch() 