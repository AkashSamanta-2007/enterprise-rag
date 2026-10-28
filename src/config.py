"""
EnterpriseRAG Configuration Module
Loads and manages environment variables for the application.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Configuration class for EnterpriseRAG application.
    All settings are loaded from environment variables with sensible defaults.
    """
    
    # Groq API Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile")
    
    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
    
    # PDF Input Configuration
    PDF_INPUT_DIR: str = os.getenv("PDF_INPUT_DIR", "./data/documents")
    
    # Text Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate required configuration.
        Returns True if all required settings are present.
        """
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please set it in your .env file.")
        return True
    
    @classmethod
    def get_project_root(cls) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist."""
        Path(cls.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.PDF_INPUT_DIR).mkdir(parents=True, exist_ok=True)


# Create a global config instance
config = Config()
