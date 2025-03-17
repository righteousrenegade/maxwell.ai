#!/usr/bin/env python3
import argparse
import time
import os
import signal
import sys
import logging
import speech_recognition as sr
import traceback
from speech import TextToSpeech
from commands import CommandExecutor
from utils import setup_logger, download_models
from config import Config
import random
from audio_manager import AudioManager

# Setup logger only once with a specific name to avoid duplicates
logger = logging.getLogger("maxwell")
if not logger.handlers:  # Only setup if not already configured
    logger = setup_logger()

def list_microphones():
    """List all available microphones"""
    try:
        mics = sr.Microphone.list_microphone_names()
        logger.info(f"Found {len(mics)} microphones:")
        for i, mic_name in enumerate(mics):
            logger.info(f"  {i}: {mic_name}")
        return mics
    except Exception as e:
        logger.error(f"Error listing microphones: {e}")
        logger.error(traceback.format_exc())
        return []

class Maxwell:
    def __init__(self, config):
        self.config = config
        self.running = True
        self.in_conversation = False
        self.speaking = False
        
        # Initialize components
        logger.info("Initializing Text-to-Speech...")
        self.tts = TextToSpeech(voice=config.voice, speed=config.speed)
        
        # Initialize AudioManager
        logger.info("Initializing AudioManager...")
        self.audio_manager = AudioManager(mic_index=config.mic_index, energy_threshold=config.energy_threshold)
        
        # Set callbacks
        self.audio_manager.on_speech_detected = self._on_speech_detected
        self.audio_manager.on_speech_recognized = self._on_speech_recognized
        
        # Speech recognition state
        self.wake_word = config.wake_word.lower()
        self.interrupt_word = config.interrupt_word.lower()
        
        logger.info("Initializing Command Executor...")
        self.command_executor = CommandExecutor(self)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"Maxwell initialized with wake word: '{config.wake_word}'")
        
    def _on_speech_detected(self, event_type, text):
        """Callback for speech detection events"""
        if event_type == "wake_word":
            logger.info(f"Wake word detected: {text}")
            self._play_listening_sound()
            self.speak("Yes?")
            self.in_conversation = True
            self.audio_manager.set_conversation_mode(True)
        elif event_type == "interrupt":
            logger.info(f"Interrupt word detected: {text}")
            self.tts.stop()
            print("🛑 Speech interrupted!")
    
    def _on_speech_recognized(self, text):
        """Callback for recognized speech in conversation mode"""
        logger.info(f"Speech recognized in conversation mode: {text}")
        print(f"🎯 I heard: \"{text}\"")
        
        # Check for "end conversation" command
        if "end conversation" in text.lower():
            logger.info("End conversation command detected")
            self.speak("Ending conversation.")
            self.in_conversation = False
            self.audio_manager.set_conversation_mode(False)
            print("🔴 Conversation ended. Say the wake word to start again.")
            return
            
        # Process the command
        self.handle_query(text)
        
    def signal_handler(self, sig, frame):
        """Handle interrupt signals"""
        logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
        self.cleanup()
        sys.exit(0)
        
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        # Only clean up once
        if hasattr(self, 'cleaned_up') and self.cleaned_up:
            logger.info("Already cleaned up, skipping...")
            return
            
        try:
            # Clean up TTS
            if hasattr(self, 'tts'):
                self.tts.cleanup()
                
            # Clean up AudioManager
            if hasattr(self, 'audio_manager'):
                self.audio_manager.stop()
                
            # Mark as cleaned up
            self.cleaned_up = True
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            logger.error(traceback.format_exc())
        
    def speak(self, text):
        """Speak text with interrupt support"""
        if not text:
            return
            
        # Set speaking flag
        self.speaking = True
        
        # Print what we're saying
        print(f"🔊 Maxwell: \"{text}\"")
        
        try:
            # Start speaking
            self.tts.speak(text)
        except Exception as e:
            logger.error(f"Error in speak: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Clear speaking flag
            self.speaking = False
        
    def handle_query(self, query):
        """Handle a user query"""
        if not query:
            logger.info("Empty query, ignoring")
            return
            
        logger.info(f"Handling query: {query}")
        
        # Check for "end conversation" command
        if "end conversation" in query.lower():
            logger.info("End conversation command detected")
            self.speak("Ending conversation.")
            self.in_conversation = False
            self.audio_manager.set_conversation_mode(False)
            print("🔴 Conversation ended. Say the wake word to start again.")
            return
            
        # Process the command
        try:
            # Execute the command
            response = self.command_executor.execute(query)
            
            # Speak the response if there is one
            if response:
                self.speak(response)
                
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            logger.error(traceback.format_exc())
            self.speak("I'm sorry, I encountered an error processing your request.")
        
    def run(self):
        logger.info("Maxwell is running. Say the wake word to begin.")
        
        # Start the audio manager
        self.audio_manager.start(wake_word=self.wake_word, interrupt_word=self.interrupt_word)
        
        # Announce startup
        self.speak(f"Hello, Maxwell here. Say '{self.wake_word}' to get my attention.")
        
        # Add a test mode option for immediate conversation
        if self.config.test_mode or self.config.always_listen:
            logger.info("Test mode or always-listen mode enabled. Entering conversation mode immediately.")
            self.speak("I'm listening for commands.")
            self.in_conversation = True
            self.audio_manager.set_conversation_mode(True)
        
        # Keyboard mode - use keyboard input instead of microphone
        if self.config.keyboard_mode:
            logger.info("Keyboard mode enabled. Type your queries instead of speaking.")
            self.speak("Keyboard mode enabled. Type your queries instead of speaking.")
            
            while self.running:
                try:
                    # Use a simple input prompt
                    user_input = input("\n💬 You: ")
                    
                    if user_input.lower() == "exit":
                        logger.info("Exiting keyboard mode.")
                        break
                    
                    # Process the input as if it was spoken
                    self.handle_query(user_input)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt detected. Exiting.")
                    break
                except Exception as e:
                    logger.error(f"Error in keyboard mode: {e}")
            
            return  # Exit the run method
        
        try:
            # Print initial instructions
            print("\n" + "="*50)
            print("🎤 Maxwell Voice Assistant")
            print("="*50)
            print(f"• Say '{self.wake_word}' to get Maxwell's attention")
            print(f"• Say '{self.interrupt_word}' to stop Maxwell from talking")
            print(f"• Say 'end conversation' to exit conversation mode")
            print("="*50 + "\n")
            
            # Main loop - just keep the program running
            while self.running:
                time.sleep(0.1)  # Sleep to prevent CPU hogging
                
        finally:
            # Make sure we clean up
            self.cleanup()

    def _play_listening_sound(self):
        """Play a sound to indicate we're listening"""
        # This is a simple visual indicator for now
        print("\n🔵 Listening...")

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
    parser.add_argument("--mic-index", type=int, help="Specific microphone index to use")
    parser.add_argument("--keyboard-mode", action="store_true", help="Use keyboard input instead of microphone")
    parser.add_argument("--always-listen", action="store_true", help="Always listen for commands without wake word")
    parser.add_argument("--energy-threshold", type=int, default=300, help="Energy threshold for speech recognition (lower = more sensitive)")
    parser.add_argument("--list-mics", action="store_true", help="List available microphones and exit")
    parser.add_argument("--save-audio", action="store_true", help="Save audio files for debugging")
    parser.add_argument("--mic-name", help="Specific microphone name to use (partial match)")
    parser.add_argument("--sample-rate", type=int, help="Sample rate to use for microphone input")
    parser.add_argument("--listen-timeout", type=int, default=7, 
                        help="Timeout in seconds for listening to commands (default: 7)")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # List voices if requested
    if args.list_voices:
        from speech import TextToSpeech
        TextToSpeech.list_available_voices()
        return
        
    # List microphones if requested
    if args.list_mics:
        list_microphones()
        return
        
    # Download required models
    download_models(offline_mode=args.offline)
    
    # If mic-name is specified, find the matching microphone index
    if args.mic_name and not args.mic_index:
        mics = list_microphones()
        for i, mic_name in enumerate(mics):
            if args.mic_name.lower() in mic_name.lower():
                logger.info(f"Found microphone matching '{args.mic_name}': {i}: {mic_name}")
                args.mic_index = i
                break
        else:
            logger.warning(f"No microphone found matching '{args.mic_name}'")
    
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
        mcp_port=args.mcp_port,
        keyboard_mode=args.keyboard_mode,
        mic_index=args.mic_index,
        always_listen=args.always_listen,
        energy_threshold=args.energy_threshold,
        save_audio=args.save_audio,
        sample_rate=args.sample_rate,
        listen_timeout=args.listen_timeout
    )
    
    # Log the configuration
    logger.info(f"Starting Maxwell with configuration:")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")
    
    # Create and run assistant
    assistant = Maxwell(config)
    try:
        assistant.run()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        logger.error(traceback.format_exc())
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    main() 