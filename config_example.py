#!/usr/bin/env python3
"""
Example of how to use the new configuration system
"""

import os
import sys
import logging
from config_loader import load_config, create_config_object

# Configure colorful output if colorama is available
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama
    COLOR_SUPPORT = True
except ImportError:
    COLOR_SUPPORT = False
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = Style = DummyColor()

# Set up logging with color if available
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("maxwell")

def print_header(text):
    """Print a formatted header"""
    if COLOR_SUPPORT:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "=" * 60)
        print(f" {text} ".center(60, "="))
        print("=" * 60 + f"{Style.RESET_ALL}")
    else:
        print("\n" + "=" * 60)
        print(f" {text} ".center(60, "="))
        print("=" * 60)

def print_section(text):
    """Print a formatted section header"""
    if COLOR_SUPPORT:
        print(f"\n{Fore.GREEN}{Style.BRIGHT}## {text} {Style.RESET_ALL}")
    else:
        print(f"\n## {text}")

def print_value(key, value, source=None):
    """Print a config value with optional source"""
    if COLOR_SUPPORT:
        if source:
            print(f"{Fore.YELLOW}{key}: {Fore.WHITE}{value} {Fore.BLUE}(from {source}){Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}{key}: {Fore.WHITE}{value}{Style.RESET_ALL}")
    else:
        if source:
            print(f"{key}: {value} (from {source})")
        else:
            print(f"{key}: {value}")

def main():
    """Example of loading and using configuration"""
    
    print_header("MAXWELL CONFIGURATION SYSTEM")
    
    # Get custom config paths from command line if provided
    yaml_path = None
    env_path = None
    
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.yaml') or sys.argv[1].endswith('.yml'):
            yaml_path = sys.argv[1]
            print(f"Using custom YAML config: {yaml_path}")
        elif sys.argv[1].endswith('.env'):
            env_path = sys.argv[1]
            print(f"Using custom .env file: {env_path}")
    
    if len(sys.argv) > 2:
        if sys.argv[2].endswith('.env'):
            env_path = sys.argv[2]
            print(f"Using custom .env file: {env_path}")
    
    # 1. Load configuration from specified or default locations
    config_dict = load_config(yaml_path=yaml_path, env_path=env_path)
    
    # 2. Convert to object for compatibility with existing code
    config = create_config_object(config_dict)
    
    # 3. Display basic configuration
    print_section("BASIC CONFIGURATION")
    print_value("LLM Provider", config.get('llm_provider', 'Not set'))
    print_value("Voice", config.get('voice', 'Not set'))
    print_value("Speed", config.get('speed', 'Not set'))
    
    # 4. OpenAI settings
    print_section("OPENAI CONFIGURATION")
    print_value("API Key", '[SET]' if config.get('openai_api_key') else '[NOT SET]')
    print_value("Base URL", config.get('openai_base_url', 'Not set'))
    print_value("Model", config.get('openai_model', 'Not set'))
    
    # 5. Kokoro TTS settings
    print_section("KOKORO TTS CONFIGURATION")
    print_value("Model Path", config.get('tts_model_path', 'Not set'))
    print_value("Voices Path", config.get('tts_voices_path', 'Not set'))
    
    # 6. Test TTS with config
    print_section("TTS INITIALIZATION TEST")
    try:
        from speech import TextToSpeech
        tts = TextToSpeech(config=config)
        status = "succeeded" if tts.engine else "failed"
        if COLOR_SUPPORT:
            status_color = f"{Fore.GREEN}succeeded{Style.RESET_ALL}" if tts.engine else f"{Fore.RED}failed{Style.RESET_ALL}"
            print(f"TTS initialization {status_color}")
        else:
            print(f"TTS initialization {status}")
    except Exception as e:
        if COLOR_SUPPORT:
            print(f"{Fore.RED}Error initializing TTS: {e}{Style.RESET_ALL}")
        else:
            print(f"Error initializing TTS: {e}")
    
    print_header("END OF CONFIGURATION EXAMPLE")
    
    # Print instructions for next steps
    if COLOR_SUPPORT:
        print(f"\n{Fore.CYAN}TIP: To see detailed logs about which files were loaded,")
        print(f"     look at the log messages above starting with {Fore.YELLOW}[CONFIG]{Style.RESET_ALL}")
    else:
        print("\nTIP: To see detailed logs about which files were loaded,")
        print("     look at the log messages above starting with [CONFIG]")

if __name__ == "__main__":
    main() 