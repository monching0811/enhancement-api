"""
Enhancement AI - HuggingFace Spaces App
========================================
A simple Gradio interface for text enhancement using FLAN-T5.

Usage:
    1. This file should be in the same folder as your trained model
    2. Run: python app.py
    3. Open http://localhost:7860 in your browser

Requirements:
    pip install gradio torch transformers
"""

import os
import json
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration
MODEL_PATH = "./enhancement-ai-model/final_model"

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU")
else:
    print("Using CPU")

# Load config
config_path = "./enhancement-ai-model/model_config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {"max_source_length": 128, "max_target_length": 128}

MAX_SOURCE_LENGTH = config.get("max_source_length", 128)
MAX_TARGET_LENGTH = config.get("max_target_length", 128)

print(f"Model loaded from {MODEL_PATH}")

def enhance_text(text, task="Enhance"):
    """Enhance text using the trained model."""
    if not text.strip():
        return ""
    
    # Build prompt based on task
    task_prompts = {
        "Enhance": f"Enhance: '{text}'",
        "Expand": f"Expand: '{text}'",
        "Refine": f"Refine: '{text}'",
        "Creative": f"Add creative elements: '{text}'",
        "Summary": f"Summarize: '{text}'"
    }
    
    prompt = task_prompts.get(task, f"Enhance: '{text}'")
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_SOURCE_LENGTH,
        truncation=True
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    
    # Decode
    enhanced = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return enhanced

def generate_prompt(genre, theme=""):
    """Generate a writing prompt."""
    if theme:
        prompt = f"Write a {genre} prompt about {theme}"
    else:
        prompt = f"Write a creative {genre} writing prompt"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_SOURCE_LENGTH,
        truncation=True
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated

# Gradio Interface
with gr.Blocks(title="Enhancement AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✨ Enhancement AI")
    gr.Markdown("Transform your writing with AI-powered enhancement")
    
    with gr.Tab("Text Enhancement"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Your Text",
                    placeholder="Enter your text here...",
                    lines=4
                )
                task = gr.Dropdown(
                    label="Enhancement Type",
                    choices=["Enhance", "Expand", "Refine", "Creative", "Summary"],
                    value="Enhance"
                )
                enhance_btn = gr.Button("Enhance", variant="primary")
            with gr.Column():
                output_text = gr.Textbox(
                    label="Enhanced Text",
                    lines=4,
                    readonly=True
                )
        
        enhance_btn.click(
            fn=enhance_text,
            inputs=[input_text, task],
            outputs=output_text
        )
    
    with gr.Tab("Writing Prompts"):
        with gr.Row():
            with gr.Column():
                genre = gr.Dropdown(
                    label="Genre",
                    choices=["Fantasy", "Sci-Fi", "Romance", "Mystery", "Horror", "Adventure", "Drama"],
                    value="Fantasy"
                )
                theme = gr.Textbox(
                    label="Theme (optional)",
                    placeholder="Enter a theme...",
                    lines=2
                )
                prompt_btn = gr.Button("Generate Prompt", variant="primary")
            with gr.Column():
                prompt_output = gr.Textbox(
                    label="Generated Prompt",
                    lines=5,
                    readonly=True
                )
        
        prompt_btn.click(
            fn=generate_prompt,
            inputs=[genre, theme],
            outputs=prompt_output
        )
    
    gr.Markdown("---")
    gr.Markdown("Powered by FLAN-T5 | [GitHub](https://github.com/monching0811/enhancement-api)")

if __name__ == "__main__":
    print("\n🚀 Starting Enhancement AI App")
    print("📍 Open http://localhost:7860 in your browser\n")
    demo.launch()
