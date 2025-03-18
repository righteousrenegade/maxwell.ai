#!/usr/bin/env python3
"""
Configuration classes and defaults for Maxwell.
"""

# =============================================================================
# DEFAULT CONFIGURATION VALUES
# =============================================================================
# HOW TO USE THIS FILE:
# 
# 1. To set default values that you want to use regularly without command line args,
#    simply modify the values in the CONFIG_DEFAULTS dictionary below.
#
# 2. For example, to always use your OpenAI API key without typing it on the command line:
#    - Change "openai_api_key": None to "openai_api_key": "sk-your-actual-api-key-here"
#
# 3. To switch to using OpenAI by default instead of Ollama:
#    - Change "llm_provider": "ollama" to "llm_provider": "openai"
#
# 4. To customize the system prompt for your assistant:
#    - Modify the text in the "system_prompt" field
#
# 5. These are just defaults - you can still override them with command line arguments
#    Example: python streaming_assistant.py --llm-provider=ollama
#
# NOTE: Be careful not to commit sensitive information like API keys to public repositories!

CONFIG_DEFAULTS = {
    # General settings
    "wake_word": "hello maxwell",
    "interrupt_word": "stop talking",
    "voice": "bm_lewis",
    "speed": 1.15,
    "listen_timeout": 7,
    
    # LLM provider settings
    "llm_provider": "ollama",  # "ollama" or "openai"
    
    # System prompt for all LLM providers
    "system_prompt": """
You are Maxwell, a helpful and friendly, intelligent voice assistant.
You will answer questions in 2-3 sentences. You are pithy and concise, and have an exceptional wit.
You will make subtle jokes and sarcastic remarks occasionally in your response.
""",
    
    # Ollama configuration
    "model": "llama3",
    "ollama_host": "http://localhost",
    "ollama_port": 11434,
    
    # OpenAI configuration
    "openai_api_key": "n/a",  # Set your API key here for convenience
    "openai_base_url": "http://localhost:1234/v1",
    "openai_model": "gemma-3-27b-it",
    "openai_temperature": 0.7,
    "openai_max_tokens": None,  # Default to None (lets the API decide)
    
    # Performance and debugging
    "energy_threshold": 300,  # Lower = more sensitive microphone
    "save_audio": False,      # Save audio files for debugging
    "debug": False,           # Enable debug logging
}

class Config:
    """Configuration for Maxwell"""
    def __init__(
        self,
        wake_word=CONFIG_DEFAULTS["wake_word"], 
        interrupt_word=CONFIG_DEFAULTS["interrupt_word"],
        voice=CONFIG_DEFAULTS["voice"],
        speed=CONFIG_DEFAULTS["speed"],
        offline_mode=False,
        continuous_conversation=False,
        model=CONFIG_DEFAULTS["model"],
        ollama_host=CONFIG_DEFAULTS["ollama_host"],
        ollama_port=CONFIG_DEFAULTS["ollama_port"],
        test_mode=False,
        use_mcp=False,
        mcp_port=8080,
        keyboard_mode=False,
        mic_index=None,
        mic_name=None,
        always_listen=False,
        energy_threshold=CONFIG_DEFAULTS["energy_threshold"],
        debug=CONFIG_DEFAULTS["debug"],
        save_audio=CONFIG_DEFAULTS["save_audio"],
        sample_rate=None,
        # LLM options
        llm_provider=CONFIG_DEFAULTS["llm_provider"],  # "ollama" or "openai"
        system_prompt=CONFIG_DEFAULTS["system_prompt"],     # General system prompt for all LLM providers
        # OpenAI options
        openai_api_key=CONFIG_DEFAULTS["openai_api_key"],
        openai_base_url=CONFIG_DEFAULTS["openai_base_url"],
        openai_model=CONFIG_DEFAULTS["openai_model"],
        openai_system_prompt=None,  # Deprecated in favor of system_prompt
        openai_temperature=CONFIG_DEFAULTS["openai_temperature"],
        openai_max_tokens=CONFIG_DEFAULTS["openai_max_tokens"],
        listen_timeout=CONFIG_DEFAULTS["listen_timeout"]
    ):
        """Initialize the configuration"""
        self.wake_word = wake_word
        self.interrupt_word = interrupt_word
        self.voice = voice
        self.speed = speed
        self.offline_mode = offline_mode
        self.continuous_conversation = continuous_conversation
        self.model = model
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.test_mode = test_mode
        self.use_mcp = use_mcp
        self.mcp_port = mcp_port
        self.keyboard_mode = keyboard_mode
        self.mic_index = mic_index
        self.mic_name = mic_name
        self.always_listen = always_listen
        self.energy_threshold = energy_threshold
        self.debug = debug
        self.save_audio = save_audio
        self.sample_rate = sample_rate
        
        # LLM options
        self.llm_provider = llm_provider
        
        # System prompt (use the general one if provided, otherwise use provider-specific one)
        self.system_prompt = system_prompt or openai_system_prompt or CONFIG_DEFAULTS["system_prompt"]
        
        # OpenAI and LLM provider options
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model
        self.openai_system_prompt = self.system_prompt  # For backwards compatibility
        self.openai_temperature = openai_temperature
        self.openai_max_tokens = openai_max_tokens
        self.listen_timeout = listen_timeout

        print("\n\n-----\nJust set self.openai_api_key to: |", self.openai_api_key, "| ....")
