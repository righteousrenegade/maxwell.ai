class Config:
    def __init__(
        self,
        wake_word="hey maxwell", 
        interrupt_word="stop talking",
        voice="bm_lewis",
        speed=1.25,
        offline_mode=False,
        continuous_conversation=False,
        model="dolphin-llama3:8b-v2.9-q4_0",
        ollama_host="localhost",
        ollama_port=11434,
        test_mode=False,
        use_mcp=False,
        mcp_port=8080,
        keyboard_interrupt=False,
        keyboard_mode=False,
        mic_index=None,
        always_listen=False,
        energy_threshold=300,
        debug=False,
        save_audio=False,
        sample_rate=None,
        auto_find_mic=False,
        listen_timeout=7
    ):
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
        self.keyboard_interrupt = keyboard_interrupt
        self.keyboard_mode = keyboard_mode
        self.mic_index = mic_index
        self.always_listen = always_listen
        self.energy_threshold = energy_threshold
        self.debug = debug
        self.save_audio = save_audio
        self.sample_rate = sample_rate
        self.auto_find_mic = auto_find_mic
        self.listen_timeout = listen_timeout 