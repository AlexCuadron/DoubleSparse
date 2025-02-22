"""
FastAPI server for DoubleSparse LLM inference with sparse attention.
"""
import os
import sys
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import json

# Add DoubleSparse to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.model import Model
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for generation")
    max_new_tokens: Optional[int] = Field(100, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.95, description="Top-p sampling parameter")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")

class GenerateResponse(BaseModel):
    text: str = Field(..., description="Generated text")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")

# Initialize FastAPI app
app = FastAPI(
    title="DoubleSparse API",
    description="REST API for LLM inference using DoubleSparse with sparse attention",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and tokenizer on startup."""
    global model, tokenizer
    
    try:
        logger.info("Loading model and tokenizer...")
        model_name = "meta-llama/Llama-2-7b-chat-hf"  # Can be made configurable
        
        # Initialize model with sparse attention
        model = Model.from_pretrained(
            model_name,
            heavy_const=128,  # Sparse attention params
            group_factor=4,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint returning API status."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a prompt."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # Generate
        outputs = model.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate usage
        usage = {
            "prompt_tokens": len(input_ids[0]),
            "completion_tokens": len(outputs[0]) - len(input_ids[0]),
            "total_tokens": len(outputs[0])
        }
        
        return GenerateResponse(text=generated_text, usage=usage)
    
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Stream generated text token by token."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def stream_response():
        try:
            # Tokenize input
            input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            # Stream generation
            for token in model.generate_stream(
                input_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                text = tokenizer.decode(token, skip_special_tokens=True)
                yield f"data: {json.dumps({'text': text})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(stream_response(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=52309,  # Using the provided port
        reload=True
    )