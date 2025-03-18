#!/usr/bin/env python3
"""
LLM Provider abstraction for the voice assistant.

This module defines the LLMProvider abstract class and concrete implementations
for different LLM providers (Ollama, OpenAI).
"""

import os
import re
import json
import logging
import requests
import time
import traceback

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Iterator, Tuple

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
    logger.warning("Ollama Python client not available")

# Check if the OpenAI client is available
try:
    import openai
    from openai import OpenAIError
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    logger.warning("OpenAI client not available")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config):
        """Initialize the LLM provider.
        
        Args:
            config: The configuration object with LLM settings
        """
        self.config = config
        self.available = False
        self.model = None
        self.last_check_time = 0
        self.check_interval = 60  # Check connectivity every 60 seconds
        self.system_prompt = config.system_prompt if hasattr(config, 'system_prompt') else None
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the provider and check availability.
        
        Returns:
            bool: True if the provider is available, False otherwise
        """
        pass
        
    @abstractmethod
    def check_connection(self, force=False, detailed=False) -> bool:
        """Check if the provider is available.
        
        Args:
            force: Force a new check even if checked recently
            detailed: Whether to print detailed connection information
            
        Returns:
            bool: True if the provider is available, False otherwise
        """
        pass
        
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat request to the LLM.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters for the provider
            
        Returns:
            str: The LLM's response
        """
        pass
        
    def get_model_name(self) -> str:
        """Get the name of the currently configured model.
        
        Returns:
            str: The model name
        """
        return self.model
        
    def format_user_query(self, query: str) -> List[Dict[str, str]]:
        """Format a user query as a list of messages.
        
        Args:
            query: The user's query
            
        Returns:
            List[Dict[str, str]]: A list of message dictionaries
        """
        messages = []
        
        # Add system prompt if available
        if self.system_prompt:
            logger.info(f"Adding system prompt: {self.system_prompt}")
            messages.append({"role": "system", "content": self.system_prompt})
            
        # Add user message
        messages.append({"role": "user", "content": query})
        
        return messages
        
    def simple_query(self, query: str, **kwargs) -> str:
        """Simple interface to send a query to the LLM.
        
        Args:
            query: The query text
            **kwargs: Additional parameters for the provider
            
        Returns:
            str: The LLM's response
        """
        messages = self.format_user_query(query)
        return self.chat(messages, **kwargs)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""
    
    def __init__(self, config):
        """Initialize the Ollama provider.
        
        Args:
            config: The configuration object with Ollama settings
        """
        super().__init__(config)
        
        # Extract Ollama-specific configuration
        self.host = config.ollama_host if hasattr(config, 'ollama_host') else "http://localhost"
        self.port = config.ollama_port if hasattr(config, 'ollama_port') else 11434
        self.model = config.model if hasattr(config, 'model') else "llama3"
        self.base_url = f"{self.host}:{self.port}"
        self.ollama_client = None
        
        # Call initialize to set up the connection
        self.available = self.initialize()
        
    def initialize(self) -> bool:
        """Initialize the Ollama provider and check availability."""
        logger.info(f"Initializing Ollama provider with host {self.base_url} and model {self.model}")
        
        # Check if Ollama client is available
        if not OLLAMA_CLIENT_AVAILABLE:
            logger.warning("Ollama Python client not available, connectivity checks will be more limited")
            
        # Check connection
        try:
            available = self.check_connection(force=True)
            if not available:
                logger.warning("Ollama service is not available. Will use command mode only.")
                print("⚠️ Warning: Ollama service is not available. Using command mode only.")
            return available
        except Exception as e:
            logger.error(f"Error initializing Ollama client: {e}")
            logger.error(traceback.format_exc())
            print(f"⚠️ Warning: Failed to initialize Ollama client: {e}")
            return False
            
    def check_connection(self, force=False, detailed=False) -> bool:
        """Check if Ollama is available."""
        current_time = time.time()
        
        # Skip check if we checked recently, unless forced
        if not force and current_time - self.last_check_time < self.check_interval:
            return self.available
            
        self.last_check_time = current_time
        
        # Test the Ollama API
        try:
            logger.info(f"Testing Ollama API at {self.base_url}")
            
            if detailed:
                print(f"🔍 Testing connection to Ollama API...")
                
            # Check if Ollama client is available and use it if possible
            if OLLAMA_CLIENT_AVAILABLE:
                try:
                    # Set environment variable for the host
                    os.environ["OLLAMA_HOST"] = self.base_url
                    
                    # Try to list models
                    models_list = ollama.list()
                    
                    # Extract model names for detailed info
                    if detailed:
                        try:
                            model_names = []
                            if isinstance(models_list, dict) and "models" in models_list:
                                model_names = [m.get("name") for m in models_list["models"] 
                                             if isinstance(m, dict) and "name" in m]
                            elif isinstance(models_list, list):
                                model_names = [m.get("name") for m in models_list 
                                             if isinstance(m, dict) and "name" in m]
                                             
                            if model_names:
                                print(f"✅ Ollama is running! Available models: {', '.join(model_names)}")
                                
                                # Check if our model is available
                                if self.model not in model_names:
                                    print(f"⚠️ Model {self.model} is not available")
                                    
                                    # Try to use a default model if available
                                    if "llama3" in model_names or any("llama3" in m for m in model_names):
                                        m = next((m for m in model_names if "llama3" in m), "llama3")
                                        self.model = m
                                        print(f"🔄 Using fallback model: {self.model}")
                                    elif "llama2" in model_names:
                                        self.model = "llama2"
                                        print(f"🔄 Using fallback model: llama2")
                                    elif model_names:
                                        self.model = model_names[0]
                                        print(f"🔄 Using fallback model: {self.model}")
                            else:
                                print("✅ Ollama is running, but couldn't determine available models")
                        except Exception as model_ex:
                            print("✅ Ollama is running, but couldn't determine available models")
                            logger.warning(f"Error parsing model list: {model_ex}")
                    
                    # If we got here, Ollama is available
                    self.available = True
                    logger.info("Ollama connection check: SUCCESSFUL")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error connecting to Ollama via Python client: {e}")
                    if detailed:
                        print(f"❌ Error connecting to Ollama: {str(e)}")
                    self.available = False
                    return False
            else:
                # Fall back to direct HTTP API check if client not available
                try:
                    response = requests.get(f"{self.host}:{self.port}/api/tags", timeout=5)
                    if response.status_code != 200:
                        if detailed:
                            print(f"❌ Error connecting to Ollama API: Status {response.status_code}")
                        self.available = False
                        return False
                        
                    # Process the response
                    data = response.json()
                    
                    if detailed:
                        # Try to extract model names
                        try:
                            model_names = [model.get('name') for model in data.get('models', [])]
                            print(f"✅ Ollama is running! Available models: {', '.join(model_names)}")
                            
                            # Check if our model is available
                            if self.model not in model_names and model_names:
                                print(f"⚠️ Model {self.model} is not available")
                                # Use first available model as fallback
                                self.model = model_names[0]
                                print(f"🔄 Using fallback model: {self.model}")
                        except Exception as model_ex:
                            print("✅ Ollama is running, but couldn't determine available models")
                            logger.warning(f"Error parsing model list: {model_ex}")
                    
                    # If we got here, Ollama is available
                    self.available = True
                    logger.info("Ollama connection check: SUCCESSFUL")
                    return True
                    
                except requests.RequestException as e:
                    if detailed:
                        print(f"❌ Error connecting to Ollama API: {str(e)}")
                    logger.error(f"Error connecting to Ollama via HTTP: {e}")
                    self.available = False
                    return False
        except Exception as e:
            if detailed:
                print(f"❌ Error checking Ollama connection: {str(e)}")
            logger.error(f"Error checking Ollama connection: {e}")
            self.available = False
            return False
            
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat request to Ollama."""
        try:
            logger.info(f"Sending chat request to Ollama model {self.model}")
            
            # Check if connection is available
            if not self.available and not self.check_connection(force=True):
                logger.error("Cannot chat with Ollama - service is not available")
                raise Exception("Ollama service is not available")
                
            # Extract parameters from kwargs with defaults
            temperature = kwargs.get('temperature', 0.7)
            max_tokens = kwargs.get('max_tokens', 1000)
            
            # Use the Ollama client if available
            if OLLAMA_CLIENT_AVAILABLE:
                # Set environment variable for the host if specified
                os.environ["OLLAMA_HOST"] = self.base_url
                
                # Use the chat API
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )
                
                # Extract response text from the returned object
                if isinstance(response, dict) and "message" in response:
                    if isinstance(response["message"], dict) and "content" in response["message"]:
                        return response["message"]["content"]
                
                # Fallback if structure is unexpected
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    return response.message.content
                    
                # Final fallback
                return str(response)
                
            else:
                # Fall back to HTTP API
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                response = requests.post(
                    f"{self.host}:{self.port}/api/chat",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
                # Parse the response
                data = response.json()
                
                # Extract the response message
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                    
                # Fallback
                return str(data)
                
        except Exception as e:
            logger.error(f"Error in Ollama chat: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"Error communicating with Ollama: {str(e)}")


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, config):
        """Initialize the OpenAI provider.
        
        Args:
            config: The configuration object with OpenAI settings
        """
        super().__init__(config)
        
        # Extract OpenAI-specific configuration
        self.api_key = config.openai_api_key if hasattr(config, 'openai_api_key') else "n/a"
        print("\n\n-----\nJust set self.api_key to: |", self.api_key, "| ....")
        self.base_url = config.openai_base_url if hasattr(config, 'openai_base_url') else "http://localhost:1234/v1"
        self.model = config.openai_model if hasattr(config, 'openai_model') else "gpt-3.5-turbo"
        self.temperature = config.openai_temperature if hasattr(config, 'openai_temperature') else 0.7
        self.max_tokens = config.openai_max_tokens if hasattr(config, 'openai_max_tokens') else None
        self.openai_client = None
        
        # Call initialize to set up the connection
        self.available = self.initialize()
        
    def initialize(self) -> bool:
        """Initialize the OpenAI provider and check availability."""
        logger.info(f"Initializing OpenAI provider with base URL {self.base_url} and model {self.model}")
        
        # Check if OpenAI client is available
        if not OPENAI_CLIENT_AVAILABLE:
            logger.error("OpenAI client not available")
            print("⚠️ Error: OpenAI client not available. Install with 'pip install openai'")
            return False
            
        # Check for invalid API key (literal "None" string or empty)
        if not self.api_key or self.api_key == None:
            logger.error("OpenAI API key is not set or is set to 'None'")
            print("❌ ERROR: OpenAI API key is not set or is set to 'None'")
            print("Please provide a valid API key with --openai-api-key")
            return False
            
        # Check for invalid base URL
        if "://" not in self.base_url:
            logger.error(f"OpenAI base URL is invalid (missing protocol): {self.base_url}")
            print(f"❌ ERROR: OpenAI base URL is invalid (missing protocol): {self.base_url}")
            print("Example: --openai-base-url=https://api.openai.com/v1")
            return False
            
        # Initialize the OpenAI client
        try:
            self.openai_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Check connection
            available = self.check_connection(force=True)
            if not available:
                logger.warning("OpenAI API is not available. Will use command mode only.")
                print("⚠️ Warning: OpenAI API is not available. Using command mode only.")
            return available
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            logger.error(traceback.format_exc())
            print(f"⚠️ Warning: Failed to initialize OpenAI client: {e}")
            return False
            
    def check_connection(self, force=False, detailed=False) -> bool:
        """Check if OpenAI API is available."""
        current_time = time.time()
        
        # Skip check if we checked recently, unless forced
        if not force and current_time - self.last_check_time < self.check_interval:
            return self.available
            
        self.last_check_time = current_time
        
        # Quick validation of key config
        if not self.api_key or self.api_key == None:
            if detailed:
                print("❌ ERROR: API key is not set or is 'None'")
                print("   You must provide a valid API key with --openai-api-key")
            self.available = False
            return False
            
        # Test the OpenAI API with a simple models listing request
        try:
            logger.info(f"Testing OpenAI API at {self.base_url}")
            
            if detailed:
                print(f"🔍 Testing connection to OpenAI API...")
                print(f"   URL: {self.base_url}")
                print(f"   Model: {self.model}")
                # Don't print the actual API key value
                print(f"   API Key: {'Set' if self.api_key and self.api_key != 'None' else 'NOT SET'}")
            
            # Use a separate client instance with a short timeout for testing
            test_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=5.0  # 5 second timeout for faster diagnostics
            )
            
            # Use list models to test connectivity - this is a lightweight call
            try:
                models_response = test_client.models.list()
                
                # Check if we got a valid response
                if detailed:
                    try:
                        # Extract model IDs for information
                        if hasattr(models_response, 'data') and models_response.data:
                            model_ids = [model.id for model in models_response.data if hasattr(model, 'id')]
                            print(f"✅ OpenAI API is available! Available models: {', '.join(model_ids[:5])}...")
                            
                            # Check if our model is in the list
                            if self.model not in model_ids:
                                print(f"⚠️ Warning: Model {self.model} may not be available")
                                print("💡 The API might still work if the model is available but not listed")
                                
                                # Try to suggest an alternative model
                                if model_ids:
                                    print(f"💡 Available models include: {', '.join(model_ids[:5])}")
                        else:
                            print("✅ OpenAI API is available, but couldn't determine available models")
                    except Exception as model_ex:
                        logger.warning(f"Error parsing model list: {model_ex}")
                        print("✅ OpenAI API is available, but couldn't parse model list")
                
                # If we get here, the API is working
                self.available = True
                logger.info("OpenAI API connection check: SUCCESSFUL")
                return True
                
            except OpenAIError as oe:
                # Handle specific OpenAI errors
                logger.error(f"OpenAI API error: {oe}")
                if detailed:
                    print(f"❌ OpenAI API error: {str(oe)}")
                    
                    # Check for common error types and provide more specific guidance
                    err_msg = str(oe).lower()
                    if "authentication" in err_msg or "api key" in err_msg:
                        print("💡 This appears to be an authentication error.")
                        print("   Check that your API key is correct and properly formatted.")
                    elif "not found" in err_msg:
                        print("💡 This appears to be a 'not found' error.")
                        print("   Check that your base URL is correct.")
                    elif "timeout" in err_msg or "timed out" in err_msg:
                        print("💡 The API request timed out.")
                        print("   The server might be slow or unreachable.")
                    
                    print("💡 Check your API key and base URL settings")
                
                self.available = False
                logger.warning("OpenAI API connection check: FAILED")
                return False
                
            except Exception as e:
                # Handle general API errors
                logger.error(f"Error checking OpenAI API 2: {e}")
                if detailed:
                    print(f"❌ Error checking OpenAI API: {str(e)}")
                    print("💡 Check your API key and base URL settings")
                self.available = False
                logger.warning("OpenAI API connection check: FAILED")
                return False
                
        except Exception as e:
            # Handle any other errors
            logger.error(f"Error checking OpenAI API 3: {e}")
            if detailed:
                print(f"❌ Error connecting to OpenAI API: {str(e)}")
            self.available = False
            return False
            
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat request to OpenAI."""
        try:
            logger.info(f"Sending chat request to OpenAI model {self.model}")
            
            # Check if connection is available
            if not self.available and not self.check_connection(force=True):
                logger.error("Cannot chat with OpenAI - service is not available")
                raise Exception("OpenAI service is not available")
                
            # Extract parameters from kwargs with defaults
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # Prepare parameters for the API call
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            
            # Add max_tokens if specified
            if max_tokens:
                params["max_tokens"] = max_tokens
                
            # Make the API call
            response = self.openai_client.chat.completions.create(**params)
            
            # Extract the response text
            if hasattr(response, 'choices') and len(response.choices) > 0:
                # Get the first choice's content
                first_choice = response.choices[0]
                if hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
                    return first_choice.message.content
                else:
                    logger.warning(f"Unexpected response structure: {first_choice}")
                    return str(first_choice)
            else:
                logger.warning(f"Unexpected response format: {response}")
                return str(response)
                
        except OpenAIError as oe:
            # Handle specific OpenAI errors
            logger.error(f"OpenAI API error: {oe}")
            
            # Extract useful info from error message
            err_msg = str(oe).lower()
            
            # Provide more helpful error messages based on error type
            if "authentication" in err_msg or "api key" in err_msg:
                raise Exception("Authentication error with OpenAI API. Check that your API key is correct.")
            elif "not found" in err_msg:
                raise Exception("Resource not found error with OpenAI API. Check that your base URL and model are correct.")
            elif "timeout" in err_msg or "timed out" in err_msg:
                raise Exception("OpenAI API request timed out. The server might be slow or unreachable.")
            elif "rate limit" in err_msg:
                raise Exception("OpenAI API rate limit exceeded. Please try again later or reduce request frequency.")
            else:
                # Pass through the original error with some context
                raise Exception(f"OpenAI API error: {str(oe)}")
                
        except Exception as e:
            logger.error(f"Error in OpenAI chat: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"Error communicating with OpenAI: {str(e)}")
            

def create_llm_provider(config) -> LLMProvider:
    """Factory function to create the appropriate LLM provider.
    
    Args:
        config: The configuration object with LLM settings
        
    Returns:
        LLMProvider: The initialized LLM provider
    """
    provider_type = config.llm_provider.lower() if hasattr(config, 'llm_provider') else "ollama"
    
    if provider_type == "ollama":
        logger.info(f"Initializing Ollama provider with base URL {config.ollama_base_url} and model {config.ollama_model}")
        return OllamaProvider(config)
    elif provider_type == "openai":
        logger.info(f"Initializing OpenAI provider with base URL {config.openai_base_url} and model {config.openai_model}")
        return OpenAIProvider(config)
    else:
        logger.error(f"Unknown LLM provider type: {provider_type}")
        raise ValueError(f"Unknown LLM provider type: {provider_type}") 