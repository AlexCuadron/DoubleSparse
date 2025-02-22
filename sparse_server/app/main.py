"""
FastAPI server for DoubleSparse LLM inference with sparse attention.
"""
import os
import sys
import time
import uuid
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import json

from .config import settings
from ..models.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="Messages to generate completions for")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.95, description="Top-p sampling parameter")
    max_tokens: Optional[int] = Field(100, description="Maximum number of tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")

class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")

class ChatCompletionResponseChoice(BaseModel):
    index: int = Field(..., description="Index of the choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatCompletionResponseChoice] = Field(..., description="Generated completions")
    usage: CompletionUsage = Field(..., description="Token usage statistics")

class DeltaMessage(BaseModel):
    role: Optional[str] = Field(None, description="Role of the delta message")
    content: Optional[str] = Field(None, description="Content of the delta message")

class ChatCompletionStreamChoice(BaseModel):
    index: int = Field(..., description="Index of the choice")
    delta: DeltaMessage = Field(..., description="Delta message content")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing")

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatCompletionStreamChoice] = Field(..., description="Generated completion chunks")

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize model manager on startup."""
    global model_manager
    try:
        model_manager = ModelManager(
            model_name=settings.DEFAULT_MODEL,
            heavy_const=settings.HEAVY_CONST,
            group_factor=settings.GROUP_FACTOR,
            channel=settings.CHANNEL,
            architecture=settings.MODEL_ARCHITECTURE
        )
        model_manager.load_model()
        logger.info(f"Model manager initialized with sparse attention (heavy_const={settings.HEAVY_CONST}, group_factor={settings.GROUP_FACTOR})")
    except Exception as e:
        logger.error(f"Error initializing model manager: {str(e)}")
        raise RuntimeError(f"Failed to initialize model manager: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "model": settings.DEFAULT_MODEL,
        "sparse_attention": {
            "heavy_const": settings.HEAVY_CONST,
            "group_factor": settings.GROUP_FACTOR,
            "channel": settings.CHANNEL
        },
        "model_loaded": model_manager is not None
    }

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [{
            "id": settings.DEFAULT_MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "organization"
        }]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(request),
                media_type='text/event-stream'
            )

        # Generate completion
        completion_text, usage = model_manager.generate(
            messages=[msg.dict() for msg in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        response = ChatCompletionResponse(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(**usage)
        )

        return response

    except Exception as e:
        logger.error(f"Error during chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_chat_completion(request: ChatCompletionRequest):
    """Stream chat completion chunks."""
    completion_id = f"chatcmpl-{str(uuid.uuid4())}"
    
    try:
        # Start with role
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None
                )
            ]
        )
        yield f"data: {chunk.json()}\n\n"

        # Stream the content
        for text_chunk in model_manager.generate_stream(
            messages=[msg.dict() for msg in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        ):
            chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=text_chunk),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {chunk.json()}\n\n"

        # Send the final chunk
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {chunk.json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error during streaming: {str(e)}")
        error_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="error"
                )
            ]
        )
        yield f"data: {error_chunk.json()}\n\n"
        yield "data: [DONE]\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=52309,  # Using the provided port
        reload=True
    )