"""
Groq LLM Client
Handles communication with Groq API for text generation.
"""

import logging
import os
from typing import List, Dict, Any, Optional

try:
    from groq import Groq
except ImportError:
    Groq = None

from src.config import config
from src.logging_utils import setup_secure_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = setup_secure_logging(__name__)


class LLMError(Exception):
    """Custom exception for LLM errors."""
    pass


class GroqClient:
    """
    Client for Groq API to generate chat completions.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key. Defaults to config.GROQ_API_KEY.
            model: Model name. Defaults to config.GROQ_MODEL_NAME.
        """
        if Groq is None:
            raise LLMError(
                "groq package is not installed. Please install it with: pip install groq"
            )
        
        self.api_key = api_key if api_key is not None else config.GROQ_API_KEY
        self.model = model if model is not None else config.GROQ_MODEL_NAME
        
        if not self.api_key:
            raise LLMError(
                "Groq API key is not set. Please set GROQ_API_KEY in your .env file."
            )
        
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.model}")
        except Exception as e:
            logger.error("Failed to initialize Groq client")
            raise LLMError(f"Failed to initialize Groq client: {e}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters for the API.
        
        Returns:
            Generated response content.
        """
        try:
            logger.info(f"Sending request to Groq API with model {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            content = response.choices[0].message.content
            logger.info("Successfully received response from Groq API")
            return content
            
        except Exception as e:
            error_msg = str(e)
            logger.error("Groq API error occurred")
            
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower() or "invalid api key" in error_msg.lower():
                raise LLMError("Authentication failed. Check your GROQ_API_KEY.")
            elif "rate" in error_msg.lower() or "429" in error_msg:
                raise LLMError("Rate limit exceeded. Please try again later.")
            elif "model" in error_msg.lower():
                raise LLMError(f"Model error. Check if '{self.model}' is available.")
            else:
                raise LLMError("Groq API error occurred. Please try again.")
    
    def generate_answer(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate answer from a prompt.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
        
        Returns:
            Generated answer.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_completion(messages)
