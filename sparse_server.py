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
from typing import List, Dict, Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

def create_app(model_path: str, heavy_const: int, group_factor: int, channel: str = "qk", offloading: bool = False):
    """Create FastAPI app with the specified model and parameters."""
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

    # Load model and convert to sparse attention
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

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
            if request.stream:
                return StreamingResponse(
                    stream_chat_completion(request),
                    media_type='text/event-stream'
                )

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

            # Prepare input
            prompt = _prepare_prompt([msg.dict() for msg in request.messages])
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device) if hasattr(inputs, 'attention_mask') else None

            # Stream generation
            generated_tokens = []
            past_key_values = None

            with torch.no_grad():
                for _ in range(request.max_tokens or 100):
                    outputs = model(
                        input_ids if past_key_values is None else input_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )

                    next_token_logits = outputs.logits[:, -1, :]
                    if request.temperature > 0:
                        probs = torch.nn.functional.softmax(next_token_logits / request.temperature, dim=-1)
                        if request.top_p < 1.0:
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_indices_to_remove = cumsum_probs > request.top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            probs.masked_fill_(indices_to_remove, 0.0)
                            next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    generated_tokens.append(next_token[0].item())

                    # Update inputs and attention mask
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    if attention_mask is not None:
                        attention_mask = torch.cat([
                            attention_mask,
                            attention_mask.new_ones((attention_mask.shape[0], 1))
                        ], dim=-1)

                    # Update past key values
                    past_key_values = outputs.past_key_values

                    # Decode and yield new token
                    current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta=DeltaMessage(content=current_text),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {chunk.json()}\n\n"

                    # Check for EOS token
                    if next_token[0].item() == tokenizer.eos_token_id:
                        break

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