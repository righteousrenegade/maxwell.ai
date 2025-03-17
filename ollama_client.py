#!/usr/bin/env python3
"""
Ollama client for the voice assistant.

This module contains the OllamaClient class for interacting with the Ollama API.
"""

# Standard library imports
import os
import re
import json
import logging
import requests
from utils import setup_logger

# Get the logger instance
logger = logging.getLogger("maxwell")
if not logger.handlers:
    logger = setup_logger()

# Check if the Ollama Python client is available
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False

class OllamaClient:
    """Client for interacting with Ollama API using the official Python client."""
    
    def __init__(self, model="dolphin-llama3:8b-v2.9-q4_0", base_url="http://localhost:11434"):
        """Initialize Ollama client.
        
        Args:
            model: The model to use for generation
            base_url: The base URL for the Ollama API
        """
        self.model = model
        self.base_url = base_url
        
        # Set Ollama host if specified
        if base_url != "http://localhost:11434":
            # Extract host and port from base_url
            match = re.match(r'https?://([^:/]+)(?::(\d+))?', base_url)
            if match:
                host = match.group(1)
                port = match.group(2) or "11434"
                os.environ["OLLAMA_HOST"] = f"{host}:{port}"
        
        # Check if the official client is available
        if not OLLAMA_CLIENT_AVAILABLE:
            logger.warning("Official Ollama Python client not found. Using REST API fallback.")
            self._use_official_client = False
            self.api_url = f"{base_url}/api/generate"
            # Test connection and ensure model is available using REST API
            self._ensure_ollama_running_with_model_rest()
        else:
            self._use_official_client = True
            # Test connection and ensure model is available using Python client
            self._ensure_ollama_running_with_model_client()
    
    def _ensure_ollama_running_with_model_client(self):
        """Ensure Ollama is running and the model is available using the Python client."""
        try:
            # List available models
            models = ollama.list()
            model_names = [model.get('name') for model in models.get('models', [])]
            
            # Check if our model is available
            if self.model not in model_names:
                logger.warning(f"Model '{self.model}' not found in available models: {model_names}")
                logger.info(f"Pulling model '{self.model}'...")
                
                # Pull the model
                ollama.pull(self.model)
                logger.info(f"Model '{self.model}' pulled successfully")
            else:
                logger.info(f"Model '{self.model}' is available")
                
            logger.info(f"Connected to Ollama at {self.base_url} using Python client")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.error("Please ensure Ollama is running and accessible")
            raise
    
    def _ensure_ollama_running_with_model_rest(self):
        """Ensure Ollama is running and the model is available using the REST API."""
        try:
            # List available models
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                logger.error(f"Error connecting to Ollama: {response.status_code} {response.text}")
                logger.error("Please ensure Ollama is running and accessible")
                raise Exception(f"Error connecting to Ollama: {response.status_code}")
                
            models = response.json()
            model_names = [model.get('name') for model in models.get('models', [])]
            
            # Check if our model is available
            if self.model not in model_names:
                logger.warning(f"Model '{self.model}' not found in available models: {model_names}")
                logger.info(f"Pulling model '{self.model}'...")
                
                # Pull the model
                pull_response = requests.post(f"{self.base_url}/api/pull", json={"name": self.model})
                if pull_response.status_code != 200:
                    logger.error(f"Error pulling model: {pull_response.status_code} {pull_response.text}")
                    raise Exception(f"Error pulling model: {pull_response.status_code}")
                    
                logger.info(f"Model '{self.model}' pulled successfully")
            else:
                logger.info(f"Model '{self.model}' is available")
                
            logger.info(f"Connected to Ollama at {self.base_url} using REST API")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.error("Please ensure Ollama is running and accessible")
            raise
    
    def generate(self, prompt, system_prompt=None, max_tokens=1000, temperature=0.7, stream=False):
        """Generate text using Ollama.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator for streaming
        """
        if self._use_official_client:
            return self._generate_with_client(prompt, system_prompt, max_tokens, temperature, stream)
        else:
            return self._generate_with_rest(prompt, system_prompt, max_tokens, temperature, stream)
    
    def _generate_with_client(self, prompt, system_prompt=None, max_tokens=1000, temperature=0.7, stream=False):
        """Generate text using the Ollama Python client.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator for streaming
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            # Add system prompt if provided
            if system_prompt:
                params["system"] = system_prompt
                
            # Generate
            if stream:
                # Return a generator for streaming
                return self._stream_with_client(params)
            else:
                # Generate and return the full response
                response = ollama.generate(**params)
                return response.get("response", "")
        except Exception as e:
            logger.error(f"Error generating with Ollama client: {e}")
            return f"Error: {str(e)}"
    
    def _stream_with_client(self, params):
        """Stream text using the Ollama Python client.
        
        Args:
            params: Parameters for generation
            
        Yields:
            Chunks of generated text
        """
        try:
            # Stream the response
            for chunk in ollama.generate(**params, stream=True):
                yield chunk.get("response", "")
        except Exception as e:
            logger.error(f"Error streaming with Ollama client: {e}")
            yield f"Error: {str(e)}"
    
    def _generate_with_rest(self, prompt, system_prompt=None, max_tokens=1000, temperature=0.7, stream=False):
        """Generate text using the Ollama REST API.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator for streaming
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                },
                "stream": stream
            }
            
            # Add system prompt if provided
            if system_prompt:
                params["system"] = system_prompt
                
            # Generate
            if stream:
                # Return a generator for streaming
                return self._stream_with_rest(params)
            else:
                # Generate and return the full response
                response = requests.post(self.api_url, json=params)
                if response.status_code != 200:
                    logger.error(f"Error generating with Ollama REST API: {response.status_code} {response.text}")
                    return f"Error: {response.status_code} {response.text}"
                    
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Error generating with Ollama REST API: {e}")
            return f"Error: {str(e)}"
    
    def _stream_with_rest(self, params):
        """Stream text using the Ollama REST API.
        
        Args:
            params: Parameters for generation
            
        Yields:
            Chunks of generated text
        """
        try:
            # Stream the response
            response = requests.post(self.api_url, json=params, stream=True)
            if response.status_code != 200:
                logger.error(f"Error streaming with Ollama REST API: {response.status_code} {response.text}")
                yield f"Error: {response.status_code} {response.text}"
                return
                
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON: {line}")
                        continue
        except Exception as e:
            logger.error(f"Error streaming with Ollama REST API: {e}")
            yield f"Error: {str(e)}" 