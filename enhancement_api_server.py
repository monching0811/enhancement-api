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
MODEL_PATH = "./enhancement-ai-model/final_model"
API_HOST = "0.0.0.0"
API_PORT = 8000

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
print(f"Max source length: {MAX_SOURCE_LENGTH}")
print(f"Max target length: {MAX_TARGET_LENGTH}")

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
    allow_credentials=True,
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
