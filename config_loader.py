#!/usr/bin/env python3
"""
Configuration loader for Maxwell Assistant that supports both YAML and .env files
"""

import os
import sys
import yaml
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Get the logger instance
logger = logging.getLogger("maxwell")

# Default config paths
DEFAULT_YAML_PATH = "config.yaml"
DEFAULT_ENV_PATH = ".env"

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        logger.warning(f"[CONFIG] YAML file not found: {config_path}")
        return {}
        
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"[CONFIG] ✓ Successfully loaded YAML config from: {os.path.abspath(config_path)}")
            return config or {}
    except Exception as e:
        logger.error(f"[CONFIG] ✗ Error loading YAML from {config_path}: {e}")
        return {}

def load_config(yaml_path: Optional[str] = None, env_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from both YAML and .env files, with .env taking precedence
    
    Args:
        yaml_path: Path to YAML config file (default: config.yaml)
        env_path: Path to .env file (default: .env)
        
    Returns:
        Dict containing all configuration values
    """
    # Use default paths if none provided
    yaml_path = yaml_path or DEFAULT_YAML_PATH
    env_path = env_path or DEFAULT_ENV_PATH
    
    logger.info("=" * 60)
    logger.info(f"[CONFIG] LOADING CONFIGURATION")
    logger.info(f"[CONFIG] YAML Path: {os.path.abspath(yaml_path)}")
    logger.info(f"[CONFIG] .env Path: {os.path.abspath(env_path)}")
    logger.info("=" * 60)
    
    # Create a dict to track where each config value comes from (for debugging)
    config_sources = {}
    
    # Load YAML config first (lower precedence)
    yaml_config = load_yaml_config(yaml_path)
    flat_config = flatten_config(yaml_config)
    
    # Mark all YAML values in our source tracker
    for key in flat_config:
        config_sources[key] = f"YAML ({yaml_path})"
    
    
    # Also check actual environment variables (highest precedence)
    env_var_keys = {
        "OPENAI_API_KEY": "openai_api_key",
        "OPENAI_BASE_URL": "openai_base_url",
        "LLM_PROVIDER": "llm_provider",
        "TTS_MODEL_PATH": "tts_model_path",
        "TTS_VOICES_PATH": "tts_voices_path",
    }
    
    for env_key, config_key in env_var_keys.items():
        if env_key in os.environ:
            flat_config[config_key] = os.environ[env_key]
            config_sources[config_key] = "Environment variable"
    
    # Log all configuration sources for debugging
    logger.info("=" * 60)
    logger.info("[CONFIG] CONFIGURATION VALUES LOADED FROM:")
    for key, source in sorted(config_sources.items()):
        # Hide actual API key value
        if key == "openai_api_key":
            value = "[REDACTED]" if flat_config.get(key) else "Not set"
        else:
            value = flat_config.get(key)
        logger.info(f"[CONFIG] {key} = {value} (from {source})")
    logger.info("=" * 60)
    
    return flat_config

def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert nested YAML config to flat dictionary for backward compatibility"""
    flat_config = {}
    
    # General settings
    if "general" in config:
        for key, value in config["general"].items():
            flat_config[key] = value
            logger.debug(f"[CONFIG] Added general.{key} = {value}")
            
    # LLM settings
    if "llm" in config:
        if "provider" in config["llm"]:
            flat_config["llm_provider"] = config["llm"]["provider"]
        if "system_prompt" in config["llm"]:
            flat_config["system_prompt"] = config["llm"]["system_prompt"]
            
    # Ollama settings
    if "ollama" in config:
        if "model" in config["ollama"]:
            flat_config["model"] = config["ollama"]["model"]
        if "host" in config["ollama"]:
            flat_config["ollama_host"] = config["ollama"]["host"]
        if "port" in config["ollama"]:
            flat_config["ollama_port"] = config["ollama"]["port"]
            
    # OpenAI settings
    if "openai" in config:
        if "api_key" in config["openai"]:
            flat_config["openai_api_key"] = config["openai"]["api_key"]
        if "base_url" in config["openai"]:
            flat_config["openai_base_url"] = config["openai"]["base_url"]
        if "model" in config["openai"]:
            flat_config["openai_model"] = config["openai"]["model"]
        if "temperature" in config["openai"]:
            flat_config["openai_temperature"] = config["openai"]["temperature"]
        if "max_tokens" in config["openai"]:
            flat_config["openai_max_tokens"] = config["openai"]["max_tokens"]
            
    # TTS settings
    if "tts" in config:
        if "model_path" in config["tts"]:
            flat_config["tts_model_path"] = config["tts"]["model_path"]
        if "voices_path" in config["tts"]:
            flat_config["tts_voices_path"] = config["tts"]["voices_path"]
    
    # Final log of important settings
    logger.debug(f"[CONFIG] Final use_mcp value in flat_config: {flat_config.get('use_mcp')}")
            
    return flat_config

def get_nested_value(config: Dict[str, Any], key_path: str, default=None) -> Any:
    """Get a nested value from the configuration using dot notation"""
    keys = key_path.split('.')
    result = config
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
            
    return result

# Helper function to convert flat config to Config object for compatibility 
def create_config_object(flat_config: Dict[str, Any]) -> Any:
    """Create a Config object from flat dictionary for backward compatibility"""
    class ConfigObject:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
                
            # Ensure use_mcp is correctly set 
            if 'use_mcp' in config_dict:
                logger.debug(f"[CONFIG] Setting use_mcp explicitly to {config_dict['use_mcp']}")
                self.use_mcp = bool(config_dict['use_mcp'])
                logger.debug(f"[CONFIG] Verified use_mcp is now {self.use_mcp}")
                
        def get(self, key, default=None):
            return getattr(self, key, default)
            
        def __str__(self):
            return json.dumps({k: v for k, v in self.__dict__.items() if k != 'get'}, 
                              default=str, indent=2)
    
        
    return ConfigObject(flat_config)

# For demonstration/testing
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load config
    config = load_config()
    
    # Print loaded config
    print("\nLoaded configuration:")
    print("=" * 60)
    for key, value in sorted(config.items()):
        # Hide API key
        if key == "openai_api_key" and value:
            print(f"{key}: [REDACTED]")
        else:
            print(f"{key}: {value}")
    print("=" * 60) 