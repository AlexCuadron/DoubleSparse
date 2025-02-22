"""
Model manager for handling DoubleSparse model initialization and inference.
"""
import os
import sys
import logging
from typing import List, Optional, Iterator, Dict
import torch
from transformers import AutoTokenizer

# Add DoubleSparse to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.model import Model

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        heavy_const: int = 128,
        group_factor: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self.heavy_const = heavy_const
        self.group_factor = group_factor
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model {self.model_name}...")
            
            # Initialize model with sparse attention
            self.model = Model.from_pretrained(
                self.model_name,
                heavy_const=self.heavy_const,
                group_factor=self.group_factor,
                device=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _prepare_prompt(self, messages: List[Dict[str, str]]) -> str:
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

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Dict[str, int]:
        """Generate completion for the given messages."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Prepare prompt
        prompt = self._prepare_prompt(messages)
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if self.device == "cuda":
            input_ids = input_ids.cuda()
        
        # Generate
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Get only the new tokens
        new_tokens = outputs[0][len(input_ids[0]):]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Calculate usage
        usage = {
            "prompt_tokens": len(input_ids[0]),
            "completion_tokens": len(new_tokens),
            "total_tokens": len(outputs[0])
        }
        
        return generated_text, usage

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Iterator[str]:
        """Stream completion for the given messages."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Prepare prompt
        prompt = self._prepare_prompt(messages)
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if self.device == "cuda":
            input_ids = input_ids.cuda()
        
        # Stream generation
        prev_text = ""
        for token in self.model.generate_stream(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        ):
            current_text = self.tokenizer.decode(token, skip_special_tokens=True)
            # Yield only the new text
            new_text = current_text[len(prev_text):]
            if new_text:
                yield new_text
            prev_text = current_text