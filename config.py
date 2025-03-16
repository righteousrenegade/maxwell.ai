class Config:
    def __init__(self, 
                 wake_word="hey maxwell", 
                 interrupt_word="stop talking",
                 voice="bm_lewis",
                 speed=1.25,
                 offline_mode=False,
                 continuous_conversation=False,
                 model="dolphin-llama3:8b-v2.9-q4_0",
                 ollama_host="localhost",
                 ollama_port=11434,
                 test_mode=False):
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