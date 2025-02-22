"""
Configuration settings for the DoubleSparse API server.
"""
from typing import Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Information
    API_TITLE: str = "DoubleSparse API"
    API_DESCRIPTION: str = "OpenAI-compatible REST API for LLM inference using DoubleSparse with sparse attention"
    API_VERSION: str = "1.0.0"
    
    # Author Information
    AUTHOR_NAME: str = "AlexCuadron"
    AUTHOR_EMAIL: str = "alex.cl.2000@gmail.com"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 52309
    
    # Model Settings
    DEFAULT_MODEL: str = "meta-llama/Llama-2-7b-chat-hf"
    MODEL_ARCHITECTURE: str = "llama"  # Model architecture (llama, mistral, qwen2)
    
    # Sparse Attention Settings
    HEAVY_CONST: int = 128    # Number of tokens to keep for attention
    GROUP_FACTOR: int = 4     # Channel grouping factor
    CHANNEL: str = "qk"       # Channel selection (q, k, qk)
    
    # Generation Defaults
    DEFAULT_MAX_TOKENS: int = 100
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.95
    
    class Config:
        env_file = ".env"

settings = Settings()