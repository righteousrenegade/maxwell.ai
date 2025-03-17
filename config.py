class Config:
    """Configuration for Maxwell"""
    def __init__(
        self,
        wake_word="hello maxwell", 
        interrupt_word="stop talking",
        voice="bm_lewis",
        speed=1.15,
        offline_mode=False,
        continuous_conversation=False,
        model="gemma-3-27b-it",
        ollama_host="localhost",
        ollama_port=11434,
        test_mode=False,
        use_mcp=False,
        mcp_port=8080,
        keyboard_mode=False,
        mic_index=None,
        always_listen=False,
        energy_threshold=300,
        debug=False,
        save_audio=False,
        sample_rate=None,
        # OpenAI options
        llm_provider="openai",  # "ollama" or "openai"
        openai_api_key="n/a",
        openai_base_url="http://localhost:1234/v1",
        openai_model="phi-3",
        openai_system_prompt=None,
        openai_temperature=0.7,
        openai_max_tokens=None,
        listen_timeout=7
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
        self.always_listen = always_listen
        self.energy_threshold = energy_threshold
        self.debug = debug
        self.save_audio = save_audio
        self.sample_rate = sample_rate
        
        # OpenAI and LLM provider options
        self.llm_provider = llm_provider
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model
        self.openai_system_prompt = openai_system_prompt
        self.openai_temperature = openai_temperature
        self.openai_max_tokens = openai_max_tokens
        self.listen_timeout = listen_timeout 