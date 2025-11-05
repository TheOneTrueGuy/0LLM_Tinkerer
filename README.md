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

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for larger models)
- See requirements.txt for full list of dependencies 