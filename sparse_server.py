"""
OpenAI-compatible server for DoubleSparse with sparse attention.
Uses the same architecture and parameters as perplexity_eval.py.
"""
import os
import sys
import json
import time
import uuid
import logging
import argparse
import pkg_resources
from typing import List, Dict, Optional, Tuple
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

def check_and_install_transformers():
    """Check transformers version and install required version if needed."""
    try:
        import transformers
        version = pkg_resources.get_distribution('transformers').version
        if pkg_resources.parse_version(version) < pkg_resources.parse_version('4.37.0'):
            logger.warning(f"Transformers version {version} does not support Qwen2. Installing 4.37.0...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.37.0"])
            logger.info("Transformers updated successfully. Reloading modules...")
            import importlib
            importlib.reload(transformers)
    except Exception as e:
        logger.error(f"Error checking/installing transformers: {str(e)}")
        raise

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check and install dependencies
check_and_install_transformers()

# Now import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

def clean_model_path(model_path: str) -> str:
    """Convert URL to model identifier if needed."""
    if model_path.startswith(('http://', 'https://')):
        # Remove protocol and domain
        parts = model_path.split('/')
        if 'huggingface.co' in parts:
            # Get parts after huggingface.co
            idx = parts.index('huggingface.co')
            return '/'.join(parts[idx + 1:])
    return model_path

def load_model(model_path: str, heavy_const: int, group_factor: int, channel: str = "qk") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load and prepare the model with sparse attention."""
    try:
        logger.info(f"Loading model from {model_path}...")
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        config = AutoConfig.from_pretrained(model_path)
        
        # Check if model type is supported
        model_type = config.model_type
        logger.info(f"Detected model type: {model_type}")
        
        if model_type == "qwen2":
            # For Qwen2, we need transformers>=4.37.0
            import transformers
            version = pkg_resources.get_distribution('transformers').version
            if pkg_resources.parse_version(version) < pkg_resources.parse_version('4.37.0'):
                raise ImportError(f"Qwen2 requires transformers>=4.37.0, but found {version}")
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Model loaded successfully")

        # Load channel config if available
        channel_path = os.path.join("config", model_path.split('/')[-1] + ".json")
        channel_config = None
        if os.path.exists(channel_path):
            with open(channel_path, "r") as f:
                channel_config = json.load(f)

        # Detect architecture from config
        if "Llama" in config.__class__.__name__:
            from evaluation.modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config
            logger.info("Detected LLaMA architecture")
            model = convert_kvcache_llama_heavy_recent(model, config, heavy_const, group_factor)
            if channel_config:
                model = convert_llama_channel_config(model, channel_config, channel)
        elif "Mistral" in config.__class__.__name__:
            from evaluation.modify_mistral import convert_kvcache_mistral_heavy_recent, convert_mistral_channel_config
            logger.info("Detected Mistral architecture")
            model = convert_kvcache_mistral_heavy_recent(model, config, heavy_const, group_factor)
            if channel_config:
                model = convert_mistral_channel_config(model, channel_config, channel)
        elif "Qwen2" in config.__class__.__name__:
            from evaluation.modify_qwen2 import convert_kvcache_qwen2_heavy_recent, convert_qwen2_channel_config
            logger.info("Detected Qwen2 architecture")
            model = convert_kvcache_qwen2_heavy_recent(model, config, heavy_const, group_factor)
            if channel_config:
                model = convert_qwen2_channel_config(model, channel_config, channel)
        else:
            raise ValueError(f"Unsupported model architecture: {config.__class__.__name__}")

        model.eval()
        logger.info(f"Model loaded with sparse attention (heavy_const={heavy_const}, group_factor={group_factor})")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def create_app(model_path: str, heavy_const: int, group_factor: int, channel: str = "qk", offloading: bool = False):
    """Create FastAPI app with the specified model and parameters."""
    # Clean model path
    model_path = clean_model_path(model_path)

    # Initialize FastAPI app
    app = FastAPI(
        title="DoubleSparse API",
        description="OpenAI-compatible REST API for LLM inference using DoubleSparse with sparse attention",
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

    # Load model
    model, tokenizer = load_model(model_path, heavy_const, group_factor, channel)

    def _prepare_prompt(messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "status": "ok",
            "model": model_path,
            "sparse_attention": {
                "heavy_const": heavy_const,
                "group_factor": group_factor,
                "channel": channel
            }
        }

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [{
                "id": model_path,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization"
            }]
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        """Create a chat completion."""
        try:
            # Prepare input
            prompt = _prepare_prompt([msg.dict() for msg in request.messages])
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device) if hasattr(inputs, 'attention_mask') else None

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Get only the new tokens
            new_tokens = outputs[0][len(input_ids[0]):]
            completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Create response
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
                usage=CompletionUsage(
                    prompt_tokens=len(input_ids[0]),
                    completion_tokens=len(new_tokens),
                    total_tokens=len(outputs[0])
                )
            )

            return response

        except Exception as e:
            logger.error(f"Error during chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start DoubleSparse API server.')
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-2-7b-hf", help='Selected model')
    parser.add_argument('--offloading', action='store_true', help='Whether to use offloading')
    parser.add_argument('--channel', type=str, default="qk", choices=["q", "k", "qk"], help='Channel selection')
    parser.add_argument('--heavy_const', type=int, default=128, help='Heavy constant')
    parser.add_argument('--group_factor', type=int, default=2, help='Group factor')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Server host')
    parser.add_argument('--port', type=int, default=52309, help='Server port')

    args = parser.parse_args()

    # Create and start the app
    app = create_app(
        model_path=args.model_path,
        heavy_const=args.heavy_const,
        group_factor=args.group_factor,
        channel=args.channel,
        offloading=args.offloading
    )

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )