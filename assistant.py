#!/usr/bin/env python3
import argparse
import time
import os
import signal
import sys
from speech import SpeechRecognizer, TextToSpeech
from commands import CommandExecutor
from utils import setup_logger, download_models
from config import Config
from mcp_tools import MCPToolProvider
import logging

logger = setup_logger()

class Maxwell:
    def __init__(self, config):
        self.config = config
        self.running = True
        self.in_conversation = False
        self.speaking = False
        
        # Initialize components
        logger.info("Initializing Text-to-Speech...")
        self.tts = TextToSpeech(voice=config.voice, speed=config.speed)
        
        logger.info("Initializing Speech Recognition...")
        self.recognizer = SpeechRecognizer(
            wake_word=config.wake_word,
            interrupt_word=config.interrupt_word,
            offline_mode=config.offline_mode
        )
        
        logger.info("Initializing Command Executor...")
        self.command_executor = CommandExecutor(self)
        
        # Initialize MCP Tool Provider
        if config.use_mcp:
            logger.info("Initializing Tool Provider...")
            self.mcp_provider = MCPToolProvider(self)
            self.mcp_provider.start_server()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"Maxwell initialized with wake word: '{config.wake_word}'")
        
    def signal_handler(self, sig, frame):
        logger.info("Shutdown signal received, exiting...")
        self.running = False
        self.cleanup()
        sys.exit(0)
        
    def cleanup(self):
        logger.info("Cleaning up resources...")
        if hasattr(self, 'tts'):
            self.tts.cleanup()
        if hasattr(self, 'recognizer'):
            self.recognizer.cleanup()
        if hasattr(self, 'mcp_provider'):
            self.mcp_provider.stop_server()
            
    def speak(self, text):
        logger.info(f"Speaking: {text}")
        self.speaking = True
        self.tts.speak(text)
        self.speaking = False
        
    def listen(self):
        return self.recognizer.listen()
        
    def process_command(self, command):
        if command and command.lower().startswith("execute "):
            cmd = command[8:].strip()
            return self.command_executor.execute_command(cmd)
        return False
        
    def handle_query(self, query):
        if not query:
            self.speak("I didn't catch that. Could you please repeat?")
            return
            
        logger.info(f"Processing query: {query}")
        
        # Check for conversation end
        if "end conversation" in query.lower():
            self.speak("Ending conversation mode.")
            self.in_conversation = False
            return
            
        # Check for commands
        if self.process_command(query):
            return
            
        # Otherwise, send to LLM
        response = self.command_executor.query_llm(query)
        self.speak(response)
        
    def run(self):
        logger.info("Maxwell is running. Say the wake word to begin.")
        self.speak(f"Hello, Maxwell here.")
        
        # Add a test mode option for immediate conversation
        if self.config.test_mode:
            logger.info("Test mode enabled. Entering conversation mode immediately.")
            self.speak("Test mode enabled. I'm listening.")
            self.in_conversation = True
        
        while self.running:
            # Listen for wake word or in conversation mode
            if not self.in_conversation:
                logger.info(f"Waiting for wake word: '{self.config.wake_word}'")
                wake_word_detected = self.recognizer.detect_wake_word()
                if wake_word_detected:
                    logger.info("Wake word detected!")
                    self.speak("Yes?")
                    self.in_conversation = True
                else:
                    time.sleep(0.1)  # Prevent CPU hogging
                    continue
            
            # In conversation mode
            logger.info("In conversation mode, listening for query...")
            query = self.listen()
            
            # Check for interrupt
            if self.speaking and self.recognizer.detect_interrupt():
                self.tts.stop()
                self.speaking = False
                self.speak("Sure.")
                continue
                
            self.handle_query(query)
            
            # If not in continuous conversation mode, exit after one interaction
            if not self.config.continuous_conversation:
                logger.info("Exiting conversation mode (continuous mode disabled)")
                self.in_conversation = False
                self.speak("Call me if you need me.")

def main():
    parser = argparse.ArgumentParser(description="Maxwell Voice Assistant")
    parser.add_argument("--wake-word", default="hey maxwell", help="Wake word to activate the assistant")
    parser.add_argument("--interrupt-word", default="stop talking", help="Word to interrupt the assistant")
    parser.add_argument("--voice", default="bm_lewis", help="Voice for text-to-speech")
    parser.add_argument("--speed", default=1.25, type=float, help="Speech speed (1.0 is normal)")
    parser.add_argument("--offline", action="store_true", help="Use offline speech recognition")
    parser.add_argument("--continuous", action="store_true", help="Stay in conversation mode until explicitly ended")
    parser.add_argument("--list-voices", action="store_true", help="List available TTS voices and exit")
    parser.add_argument("--model", default="dolphin-llama3:8b-v2.9-q4_0", help="Ollama model to use")
    parser.add_argument("--ollama-host", default="localhost", help="Ollama host address")
    parser.add_argument("--ollama-port", default=11434, type=int, help="Ollama port")
    parser.add_argument("--test", action="store_true", help="Test mode - immediately enter conversation mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--use-mcp", action="store_true", help="Enable MCP tools integration")
    parser.add_argument("--mcp-port", default=8080, type=int, help="Port for MCP server")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # List voices if requested
    if args.list_voices:
        from speech import TextToSpeech
        TextToSpeech.list_available_voices()
        return
        
    # Download required models
    download_models(offline_mode=args.offline)
    
    # Create config
    config = Config(
        wake_word=args.wake_word,
        interrupt_word=args.interrupt_word,
        voice=args.voice,
        speed=args.speed,
        offline_mode=args.offline,
        continuous_conversation=args.continuous,
        model=args.model,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        test_mode=args.test,
        use_mcp=args.use_mcp,
        mcp_port=args.mcp_port
    )
    
    # Create and run assistant
    assistant = Maxwell(config)
    try:
        assistant.run()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    main() 