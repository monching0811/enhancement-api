"""
Enhancement AI API Server
=========================
FastAPI server to serve predictions from your trained model.

Usage:
    1. Place this file in the same folder as your trained model
    2. Run: python enhancement_api_server.py
    3. Server will start at http://localhost:8000

Requirements:
    pip install fastapi uvicorn torch transformers
"""

import os
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional

# Configuration
LOCAL_MODEL_PATH = "./enhancement-ai-model/final_model"
FALLBACK_MODEL = "google-t5/t5-base"
API_HOST = "0.0.0.0"
API_PORT = 8000

def is_model_valid(path):
    """Check if model files are complete"""
    if not os.path.exists(path):
        return False
    model_file = os.path.join(path, "model.safetensors")
    if os.path.exists(model_file):
        file_size = os.path.getsize(model_file)
        return file_size > 100_000_000  # 100MB minimum
    return False

# Determine model to use
print("\n" + "="*60)
print("Enhancement AI API Server")
print("="*60 + "\n")

if is_model_valid(LOCAL_MODEL_PATH):
    model_to_use = LOCAL_MODEL_PATH
    print("✓ Using your trained model")
else:
    model_to_use = FALLBACK_MODEL
    print("⚠ Local model incomplete - using fallback")
    print("  Model: google-t5/t5-base\n")
    print("📝 To use your trained model:")
    print("   1. Export from Colab: model.save_pretrained('./model')")
    print("   2. Push files to GitHub and restart\n")

# Load tokenizer
print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    print("✓ Tokenizer loaded")
except Exception as e:
    print(f"✗ Error: {e}")
    raise

# Load model
print(f"Loading model: {model_to_use}")
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_to_use)
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Error: {e}")
    raise

model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    print("✓ GPU available")
else:
    print("ℹ Using CPU")

# Load config
config_path = "./enhancement-ai-model/model_config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {"max_source_length": 128, "max_target_length": 128}

MAX_SOURCE_LENGTH = config.get("max_source_length", 128)
MAX_TARGET_LENGTH = config.get("max_target_length", 128)

print(f"Model loaded from: {model_to_use}")
print(f"Max source length: {MAX_SOURCE_LENGTH}")
print(f"Max target length: {MAX_TARGET_LENGTH}")
print("\n" + "="*60)
print("API Server Ready!")
print("="*60)
print(f"API: http://localhost:{API_PORT}")
print(f"Docs: http://localhost:{API_PORT}/docs\n")

# FastAPI app
app = FastAPI(
    title="Enhancement AI API",
    description="API for text enhancement using your trained model",
    version="1.0.0"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class EnhancementRequest(BaseModel):
    text: str
    task: Optional[str] = "enhance"  # enhance, expand, refine, creative, summary

class EnhancementResponse(BaseModel):
    original: str
    enhanced: str
    task: str

class WritingPromptRequest(BaseModel):
    genre: str
    theme: Optional[str] = None
    style: Optional[str] = None

class WritingPromptResponse(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return {"status": "ok", "message": "Enhancement AI API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/enhance", response_model=EnhancementResponse)
async def enhance_content(request: EnhancementRequest):
    """Enhance text using the trained model"""
    try:
        # Build prompt based on task
        task = request.task or "enhance"
        
        if task == "expand":
            prompt = f"Expand: '{request.text}'"
        elif task == "refine":
            prompt = f"Refine: '{request.text}'"
        elif task == "creative":
            prompt = f"Add creative elements: '{request.text}'"
        elif task == "summary":
            prompt = f"Summarize: '{request.text}'"
        else:
            prompt = f"Enhance: '{request.text}'"
        
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
        
        return EnhancementResponse(
            original=request.text,
            enhanced=enhanced,
            task=task
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prompt", response_model=WritingPromptResponse)
async def generate_prompt(request: WritingPromptRequest):
    """Generate a writing prompt"""
    try:
        if request.theme:
            prompt = f"Write a {request.genre} prompt about {request.theme}"
        else:
            prompt = f"Write a creative {request.genre} writing prompt"
        
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
        generated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return WritingPromptResponse(prompt=generated_prompt)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting Enhancement AI API Server")
    print(f"📍 Server running at http://{API_HOST}:{API_PORT}")
    print(f"📝 API Documentation at http://{API_HOST}:{API_PORT}/docs\n")
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)
