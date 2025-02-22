"""
Model manager for handling DoubleSparse model initialization and inference.
"""
import os
import sys
import json
import logging
from typing import List, Optional, Iterator, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Add DoubleSparse to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from evaluation.modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config
from evaluation.modify_mistral import convert_kvcache_mistral_heavy_recent, convert_mistral_channel_config

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        heavy_const: int = 128,
        group_factor: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        channel: str = "qk",
        architecture: str = "llama"
    ):
        self.model_name = model_name
        self.device = device
        self.heavy_const = heavy_const
        self.group_factor = group_factor
        self.channel = channel
        self.architecture = architecture
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model {self.model_name}...")
            
            # Load model and config
            kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            config = AutoConfig.from_pretrained(self.model_name)
            
            # Load channel config
            channel_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                f"{self.model_name.split('/')[-1]}.json"
            )
            
            if not os.path.exists(channel_path):
                logger.warning(f"Channel config not found at {channel_path}, using default config")
                channel_config = {}
            else:
                with open(channel_path, "r") as f:
                    channel_config = json.load(f)
            
            # Convert model to use sparse attention
            if self.architecture == "llama":
                logger.info("Converting model to use LLaMA sparse attention...")
                self.model = convert_kvcache_llama_heavy_recent(
                    self.model, 
                    config, 
                    self.heavy_const,
                    self.group_factor
                )
                if channel_config:
                    self.model = convert_llama_channel_config(
                        self.model,
                        channel_config,
                        self.channel
                    )
            elif self.architecture == "mistral":
                logger.info("Converting model to use Mistral sparse attention...")
                self.model = convert_kvcache_mistral_heavy_recent(
                    self.model,
                    config,
                    self.heavy_const,
                    self.group_factor
                )
                if channel_config:
                    self.model = convert_mistral_channel_config(
                        self.model,
                        channel_config,
                        self.channel
                    )
            else:
                raise ValueError(f"Unsupported architecture: {self.architecture}")
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully with sparse attention (heavy_const={self.heavy_const}, group_factor={self.group_factor})")
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
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, 'attention_mask') else None
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
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
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, 'attention_mask') else None
        
        # Stream generation
        generated_tokens = []
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(
                    input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                if temperature > 0:
                    # Apply temperature and top_p sampling
                    probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumsum_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        probs.masked_fill_(indices_to_remove, 0.0)
                        next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_tokens.append(next_token[0].item())
                
                # Update inputs and attention mask for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1))
                    ], dim=-1)
                
                # Update past key values
                past_key_values = outputs.past_key_values
                
                # Decode and yield new token
                current_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                yield current_text
                
                # Check for EOS token
                if next_token[0].item() == self.tokenizer.eos_token_id:
                    break