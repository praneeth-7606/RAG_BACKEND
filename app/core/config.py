import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    GEMINI_API_KEY: str
    
    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    
    # File Configuration
    MAX_FILE_SIZE: int = 50  # MB
    UPLOAD_DIR: str = "./data/uploads"
    VECTOR_STORE_DIR: str = "./data/vector_store"
    
    # Supported file types
    SUPPORTED_EXTENSIONS: list = [".pdf", ".txt"]
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()