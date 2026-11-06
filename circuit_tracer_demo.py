"""
Circuit Tracer Demo Script
This script demonstrates the basic functionality of circuit-tracer:
1. Setting up the environment
2. Running attribution analysis
3. Launching the visualization server
"""

import os
import subprocess
import time
import webbrowser
from pathlib import Path
import gradio as gr
import torch

# Configuration
GRAPH_FILE_DIR = Path("circuit_tracer_graphs")
GRAPH_FILE_DIR.mkdir(exist_ok=True)

# circuit-tracer lives in a separate venv (circuit_venv) because its deps
# (transformers<5, huggingface-hub<1.0) conflict with the main env.
# Resolution order: CIRCUIT_TRACER_EXE env var -> sibling circuit_venv ->
# "circuit-tracer" on PATH.
def circuit_tracer_exe() -> str:
    exe = os.environ.get("CIRCUIT_TRACER_EXE")
    if exe:
        return exe
    local = Path(__file__).parent / "circuit_venv" / "Scripts" / "circuit-tracer.exe"
    if local.exists():
        return str(local)
    local_posix = Path(__file__).parent / "circuit_venv" / "bin" / "circuit-tracer"
    if local_posix.exists():
        return str(local_posix)
    return "circuit-tracer"


def setup_environment():
    """Check the circuit-tracer CLI is reachable; tell the user how to install it if not."""
    exe = circuit_tracer_exe()
    try:
        probe = subprocess.run(
            [exe, "--help"], capture_output=True, text=True, timeout=60
        )
        if probe.returncode == 0:
            print(f"circuit-tracer found: {exe}")
            return True
    except Exception:
        pass
    print(
        "circuit-tracer CLI not found. Install it into its own env:\n"
        "  python -m venv circuit_venv\n"
        "  circuit_venv\\Scripts\\python.exe -m pip install git+https://github.com/safety-research/circuit-tracer.git\n"
        "or set CIRCUIT_TRACER_EXE to an existing install."
    )
    return False

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
        circuit_tracer_exe(), "attribute",
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
        deadline = time.time() + 120
        started = False
        while time.time() < deadline:
            output = process.stderr.readline()
            if "Serving on" in output:
                print(f"Server started: {output.strip()}")
                started = True
                break
            if process.poll() is not None:
                err = process.stderr.read() if process.stderr else ""
                print(f"Server failed to start: {err}")
                return None
        if not started:
            print("Timed out waiting for circuit-tracer server")
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