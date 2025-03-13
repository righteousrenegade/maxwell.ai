#!/usr/bin/env python3
"""
Voice Assistant with Ollama LLM and Kokoro TTS.

Features:
- Voice recognition with wake word detection
- Text-to-speech using Kokoro TTS
- Integration with Ollama for LLM responses
- Conversation mode that stays active until explicitly ended
- Command execution with web search capabilities
- Non-blocking speech recognition that works while speaking

Usage:
- Say "Hey Maxwell" (or just "Maxwell") to activate
- The assistant will stay in conversation mode until you say "end convo"
- Say "execute [command]" to run a command (e.g., "execute search artificial intelligence")
- Say "stop talking" to interrupt the assistant while it's speaking
"""

# Standard library imports
import os
import sys
import time
import threading
import argparse
import logging

# Import from our modules
from utils import (
    logger, stop_listening, is_speaking, conversation_history, 
    last_speaking_time, speaking_cooldown, waiting_for_user_input, 
    max_silence_before_prompt, wake_word, wake_word_active, 
    pending_user_input, listen_while_speaking, interrupt_word, 
    command_prefix, is_interrupted, available_commands,
    setup_signal_handlers, print_assistant_response,
    print_user_input, is_wake_word, is_end_conversation, is_command,
    extract_command, is_interrupt_command
)
from tools import WebBrowser
from ollama_client import OllamaClient, OLLAMA_CLIENT_AVAILABLE
from tts import KokoroTTS
from speech import SpeechRecognizer, VoskSpeechRecognizer, VOSK_AVAILABLE
from concurrent_speech import ConcurrentSpeechRecognizer
from commands import CommandExecutor

class ConversationalAssistant:
    """Main class for the conversational assistant."""
    
    def __init__(self, ollama_model="dolphin-llama3:8b-v2.9-q4_0", tts_voice="bm_lewis", 
                 language="en-us", speed=1.25, vad_aggressiveness=3,
                 use_offline_recognition=False, vosk_model_path="vosk-model-small-en-us",
                 skip_tts=False, skip_speech=False, energy_threshold=300, 
                 use_wake_word=True, custom_wake_word=None, listen_timeout=15.0,
                 pause_threshold=2.0, phrase_time_limit=15.0, auto_download=False,
                 enable_listen_while_speaking=True, custom_interrupt_word=None,
                 custom_command_prefix=None):
        """Initialize the conversational assistant."""
        # Initialize Ollama client
        self.ollama = OllamaClient(model=ollama_model)
        
        # Initialize TTS only if not skipping
        self.skip_tts = skip_tts
        if not skip_tts:
            self.tts = KokoroTTS(voice=tts_voice, language=language, speed=speed, auto_download=auto_download)
            # Set the callback for when speech is finished
            self.tts.on_speech_finished = self._process_pending_input
            # Set the callback for when speech is interrupted
            self.tts.on_speech_interrupted = self._handle_speech_interrupted
        else:
            # Create a dummy TTS object that does nothing
            logger.info("Skipping TTS initialization")
            self.tts = None
        
        # Initialize speech recognition only if not skipping
        self.skip_speech = skip_speech
        
        # Always enable listen while speaking with the new implementation
        global listen_while_speaking
        listen_while_speaking = True
        
        if not skip_speech:
            # Initialize the concurrent speech recognizer
            self.recognizer = ConcurrentSpeechRecognizer(
                vad_aggressiveness=vad_aggressiveness,
                language=language,
                use_offline=use_offline_recognition,
                vosk_model_path=vosk_model_path,
                energy_threshold=energy_threshold,
                pause_threshold=pause_threshold,
                phrase_time_limit=phrase_time_limit,
                listen_timeout=listen_timeout
            )
        else:
            logger.info("Skipping speech recognition initialization")
            self.recognizer = None
        
        # Initialize command executor
        self.command_executor = CommandExecutor()
        
        # System prompt for the assistant
        self.system_prompt = (
            "You are a helpful, friendly, and concise voice assistant named Maxwell. "
            "Keep your responses conversational but brief (1-3 sentences when possible). "
            "If you don't know something, say so clearly. "
            "Avoid lengthy introductions or unnecessary details."
            "When asked for details about something, you will provide a detailed response, but not overly so."
        )
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Set energy threshold
        self.energy_threshold = energy_threshold
        
        # Wake word settings
        self.use_wake_word = use_wake_word
        global wake_word
        if custom_wake_word:
            wake_word = custom_wake_word.lower()
            
        # Set interrupt word
        global interrupt_word
        if custom_interrupt_word:
            interrupt_word = custom_interrupt_word.lower()
            
        # Set command prefix
        global command_prefix
        if custom_command_prefix:
            command_prefix = custom_command_prefix.lower()
            
        # Set listen timeout
        self.listen_timeout = listen_timeout
        
        # Conversation mode flag
        self.conversation_mode = False
        
        # Processing flag to prevent multiple simultaneous processing
        self.is_processing = False
        
        logger.info(f"Conversational assistant initialized{' with wake word: ' + wake_word if use_wake_word else ''}")
        logger.info(f"Interrupt word set to '{interrupt_word}'")
        logger.info(f"Command prefix set to '{command_prefix}'")
        logger.info(f"Listen timeout set to {self.listen_timeout} seconds")
        logger.info(f"Pause threshold set to {pause_threshold} seconds")
        logger.info(f"Phrase time limit set to {phrase_time_limit} seconds")
    
    def _handle_speech_interrupted(self):
        """Handle speech interruption."""
        global is_interrupted, is_speaking, last_speaking_time, waiting_for_user_input, pending_user_input
        
        logger.info("Handling speech interruption")
        
        # Reset the interrupted flag immediately
        is_interrupted = False
        
        # Ensure speaking flag is reset immediately
        is_speaking = False
        last_speaking_time = time.time()
        waiting_for_user_input = True
        
        # Clear any pending input that might be an interrupt command
        if pending_user_input and is_interrupt_command(pending_user_input):
            logger.info(f"Clearing pending interrupt command: {pending_user_input}")
            pending_user_input = None
        
        # Acknowledge the interruption
        response_text = "I'll stop talking now."
        logger.info(f"Speaking interruption response: {response_text}")
        
        # Print the response
        print_assistant_response(response_text)
        
        # We don't speak the response to avoid a loop of interruptions
        # But we do need to ensure all flags are properly reset
        self.is_processing = False  # Ensure processing flag is reset
        logger.info("Speech interruption handled successfully")
    
    def _process_pending_input(self):
        """Process pending user input after speech is finished."""
        global pending_user_input, is_interrupted, is_speaking, last_speaking_time, waiting_for_user_input
        
        # Make a local copy of pending_user_input and clear the global immediately
        # to avoid race conditions
        local_pending_input = pending_user_input
        pending_user_input = None
        
        if local_pending_input:
            logger.info(f"Processing pending input: {local_pending_input}")
            
            # Don't process if it's an interrupt command (no longer relevant)
            if is_interrupt_command(local_pending_input):
                logger.info(f"Skipping processing of interrupt command after speech: {local_pending_input}")
                return
            
            # Process the input in a separate thread
            processing_thread = threading.Thread(
                target=self.process_speech_to_response,
                args=(local_pending_input,)
            )
            processing_thread.daemon = True
            processing_thread.start()
    
    def process_speech_to_response(self, speech_text):
        """Process speech text to generate and speak a response.
        
        Args:
            speech_text: Recognized speech text
        """
        global waiting_for_user_input, wake_word, wake_word_active, interrupt_word, is_interrupted, command_prefix, is_speaking, last_speaking_time, pending_user_input
        
        # Check for interrupt command first, before any other processing
        if speech_text and is_interrupt_command(speech_text):
            logger.info(f"Interrupt command detected: {speech_text}")
            # Set the interrupted flag
            is_interrupted = True
            
            # IMMEDIATELY stop speaking - this is critical for responsiveness
            if not self.skip_tts and self.tts is not None and is_speaking:
                try:
                    # Force stop any ongoing speech
                    logger.info("Forcefully stopping speech due to interrupt command")
                    self.tts.stop_speaking()
                except Exception as e:
                    logger.error(f"Error stopping speech: {e}")
                    
                    # Even if there's an error, ensure flags are reset
                    is_speaking = False
                    last_speaking_time = time.time()
                    waiting_for_user_input = True
            return
        
        # For non-interrupt commands, check if we're already processing
        if self.is_processing:
            logger.info("Already processing speech, ignoring new input")
            return
            
        # Set the processing flag for non-interrupt commands
        self.is_processing = True
        
        try:
            if not speech_text:
                waiting_for_user_input = True  # Reset to waiting for input if no speech
                return
            
            # Print and log the captured input in a clearly visible way
            print_user_input(speech_text)
            
            # Check for command execution
            if is_command(speech_text):
                # Extract the command and arguments
                command_name, command_args = extract_command(speech_text)
                
                # Execute the command
                response_text = self.command_executor.execute_command(command_name, command_args)
                
                # Print and log the response
                print_assistant_response(response_text)
                
                # Speak the response
                waiting_for_user_input = False  # Not waiting for input while speaking
                if not self.skip_tts and self.tts is not None:
                    self.tts.speak(response_text)
                
                # After speaking, we're waiting for user input again
                waiting_for_user_input = True
                return
            
            # Check for wake word if wake word is enabled and not already active
            if self.use_wake_word:
                # If wake word is not active, check if the input contains the wake word or similar
                if not wake_word_active and not self.conversation_mode:
                    if is_wake_word(speech_text):
                        # Wake word detected
                        wake_word_active = True
                        self.conversation_mode = True
                        logger.info(f"Wake word detected: {speech_text}")
                        
                        # Remove wake word from text
                        wake_word_parts = wake_word.lower().split()
                        main_name = wake_word_parts[-1] if len(wake_word_parts) > 0 else wake_word
                        
                        for phrase in [wake_word, main_name, f"hi {main_name}", f"hello {main_name}", 
                                      f"ok {main_name}", f"okay {main_name}", f"hey {main_name}", 
                                      f"{main_name} please"]:
                            speech_text = speech_text.lower().replace(phrase, "").strip()
                        
                        if not speech_text:
                            # Just acknowledge if no command after wake word
                            response_text = f"Yes, {main_name.capitalize()} here. How can I help you?"
                            logger.info(f"Speaking wake word acknowledgement: {response_text}")
                            
                            # Print and log the response
                            print_assistant_response(response_text)
                            
                            # Speak the response
                            waiting_for_user_input = False  # Not waiting for input while speaking
                            if not self.skip_tts and self.tts is not None:
                                self.tts.speak(response_text)
                            
                            # After speaking, we're waiting for user input again
                            waiting_for_user_input = True
                            return
                    else:
                        # No wake word, ignore input
                        logger.info(f"Ignoring input without wake word: {speech_text}")
                        return
            
            # Check for end conversation
            if is_end_conversation(speech_text):
                # End conversation mode
                self.conversation_mode = False
                wake_word_active = False
                
                # Acknowledge end of conversation
                response_text = "Goodbye! Let me know if you need anything else."
                logger.info(f"Speaking end conversation acknowledgement: {response_text}")
                
                # Print and log the response
                print_assistant_response(response_text)
                
                # Speak the response
                waiting_for_user_input = False  # Not waiting for input while speaking
                if not self.skip_tts and self.tts is not None:
                    self.tts.speak(response_text)
                
                # After speaking, we're waiting for user input again
                waiting_for_user_input = True
                return
            
            # Process with Ollama
            try:
                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": speech_text})
                
                # Generate response
                logger.info("Generating response with Ollama...")
                
                # Prepare messages for the model
                messages = [{"role": "system", "content": self.system_prompt}]
                messages.extend(self.conversation_history[-10:])  # Include last 10 messages
                
                # Convert messages to prompt format
                prompt = ""
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "system":
                        prompt += f"<|system|>\n{content}\n"
                    elif role == "user":
                        prompt += f"<|user|>\n{content}\n"
                    elif role == "assistant":
                        prompt += f"<|assistant|>\n{content}\n"
                
                # Add final assistant prompt
                prompt += "<|assistant|>\n"
                
                # Generate response
                response = self.ollama.generate(prompt, system_prompt=self.system_prompt)
                
                # Clean up response
                response_text = response.strip()
                
                # Add to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                # Print and log the response
                print_assistant_response(response_text)
                
                # Speak the response
                waiting_for_user_input = False  # Not waiting for input while speaking
                if not self.skip_tts and self.tts is not None:
                    self.tts.speak(response_text)
                
                # After speaking, we're waiting for user input again
                waiting_for_user_input = True
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                
                # Speak error message
                error_message = "I'm sorry, I encountered an error while processing your request. Please try again."
                logger.info(f"Speaking error message: {error_message}")
                
                # Print and log the error message
                print_assistant_response(error_message)
                
                # Speak the error message
                waiting_for_user_input = False  # Not waiting for input while speaking
                if not self.skip_tts and self.tts is not None:
                    self.tts.speak(error_message)
                
                # After speaking, we're waiting for user input again
                waiting_for_user_input = True
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            # Ensure we reset waiting_for_user_input even if there's an error
            waiting_for_user_input = True
        finally:
            # Reset processing flag
            self.is_processing = False
            # Ensure we're waiting for user input
            waiting_for_user_input = True
    
    def run(self, text_input=None):
        """Run the conversational assistant.
        
        Args:
            text_input: Optional text input to process (for testing)
        """
        global stop_listening, waiting_for_user_input, max_silence_before_prompt, wake_word_active
        global pending_user_input, is_interrupted, is_speaking, last_speaking_time
        
        # Initial greeting
        greeting = "Hello! I'm Maxwell, your voice assistant. Say 'Hey Maxwell' to activate me."
        if not self.use_wake_word:
            greeting = "Hello! I'm Maxwell, your voice assistant. How can I help you today?"
            
        logger.info(f"Speaking greeting: {greeting}")
        if not self.skip_tts and self.tts is not None:
            self.tts.speak(greeting)
        
        # If text input is provided, process it and exit
        if text_input:
            input_message = f"PROCESSING TEXT INPUT: \"{text_input}\""
            print("\n" + "="*50)
            print(input_message)
            print("="*50 + "\n")
            logger.info(input_message)
            
            # Check for command execution
            if is_command(text_input):
                # Extract the command and arguments
                command_name, command_args = extract_command(text_input)
                
                # Execute the command
                response_text = self.command_executor.execute_command(command_name, command_args)
                print(f"\nCommand response: {response_text}")
                return
            
            # If using wake word, check if the text input contains the wake word
            if self.use_wake_word and not wake_word_active and not self.conversation_mode:
                # Check for wake word in text input
                if is_wake_word(text_input):
                    # Wake word detected
                    wake_word_active = True
                    self.conversation_mode = True
                    logger.info(f"Wake word detected in text input: {text_input}")
                    
                    # Remove wake word from text
                    wake_word_parts = wake_word.lower().split()
                    main_name = wake_word_parts[-1] if len(wake_word_parts) > 0 else wake_word
                    
                    for phrase in [wake_word, main_name, f"hi {main_name}", f"hello {main_name}", 
                                  f"ok {main_name}", f"okay {main_name}", f"hey {main_name}", 
                                  f"{main_name} please"]:
                        text_input = text_input.lower().replace(phrase, "").strip()
                    
                    if not text_input:
                        # Just acknowledge if no command after wake word
                        print(f"\nYes, {main_name.capitalize()} here. How can I help you?")
                        return
                else:
                    # No wake word, ignore input
                    logger.info(f"Ignoring text input without wake word: {text_input}")
                    print("\nPlease use the wake word to activate the assistant.")
                    return
            
            # Process the text input
            self.process_speech_to_response(text_input)
            return
        
        # Initialize and start the concurrent speech recognizer if not skipping speech
        if not self.skip_speech and self.recognizer is not None:
            # Start the continuous listening thread
            self.recognizer.start_listening()
            
            # Main loop
            logger.info("Listening for commands...")
            print("\nListening for commands... Say 'Hey Maxwell' to activate me.")
            print(f"You can interrupt me by saying '{interrupt_word}' while I'm speaking.")
            print(f"You can execute commands by saying '{command_prefix} [command]'.")
            print(f"Available commands: {', '.join(available_commands.keys())}")
            print("Say 'end conversation' to exit conversation mode.\n")
            
            # Track time of last user input and speaking
            last_user_input_time = time.time()
            global last_speaking_time, is_speaking
            
            # Main loop
            waiting_for_user_input = True
            while not stop_listening:
                try:
                    # Check for speech from the queue (non-blocking)
                    speech_text = self.recognizer.get_speech(block=False)
                    
                    if speech_text:
                        # Update the last input time
                        last_user_input_time = time.time()
                        
                        # Check for interrupt command while speaking
                        if is_speaking and is_interrupt_command(speech_text):
                            logger.info(f"Interrupt command detected in main loop: {speech_text}")
                            # Set the interrupted flag
                            is_interrupted = True
                            
                            # Print and log what was heard for the interrupt
                            print_user_input(speech_text)
                            
                            # IMMEDIATELY stop speaking - this is critical for responsiveness
                            if not self.skip_tts and self.tts is not None:
                                try:
                                    # Force stop any ongoing speech
                                    logger.info("Forcefully stopping speech due to interrupt command")
                                    self.tts.stop_speaking()
                                except Exception as e:
                                    logger.error(f"Error stopping speech: {e}")
                                    
                                    # Even if there's an error, ensure flags are reset
                                    is_speaking = False
                                    last_speaking_time = time.time()
                                    waiting_for_user_input = True
                                    
                            # Acknowledge the interruption (without speaking)
                            print_assistant_response("I'll stop talking now.")
                            
                            # Ensure flags are reset
                            is_speaking = False
                            last_speaking_time = time.time()
                            waiting_for_user_input = True
                            
                            # Don't reset the interrupted flag here - let the callback handle it
                            continue  # Skip further processing of this input
                        
                        # If we're currently speaking, store the input to process after speaking
                        if is_speaking and not is_interrupt_command(speech_text):
                            logger.info(f"Received input while speaking, will process after current speech: {speech_text}")
                            pending_user_input = speech_text
                        else:
                            # We're not speaking, process immediately
                            # We're no longer waiting for user input while processing
                            waiting_for_user_input = False
                            
                            # Process in a separate thread to allow interruption
                            processing_thread = threading.Thread(
                                target=self.process_speech_to_response,
                                args=(speech_text,)
                            )
                            processing_thread.daemon = True
                            processing_thread.start()
                    else:
                        # Check if we should prompt the user after a period of silence
                        current_time = time.time()
                        time_since_last_input = current_time - last_user_input_time
                        time_since_speaking = current_time - last_speaking_time
                        
                        # If we've been silent for a while and we're in conversation mode, prompt the user
                        if (max_silence_before_prompt > 0 and 
                            time_since_last_input > max_silence_before_prompt and 
                            time_since_speaking > speaking_cooldown and 
                            self.conversation_mode and 
                            not is_speaking and 
                            waiting_for_user_input and
                            not self.is_processing):
                            
                            # Prompt the user
                            prompt_text = "Is there anything else you'd like help with?"
                            logger.info(f"Prompting user after silence: {prompt_text}")
                            
                            # Print and speak the prompt
                            print_assistant_response(prompt_text)
                            
                            if not self.skip_tts and self.tts is not None:
                                self.tts.speak(prompt_text)
                            
                            # Reset the last input time to avoid repeated prompts
                            last_user_input_time = current_time
                        
                        # Small sleep to reduce CPU usage when no speech detected
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt detected")
                    break
            
            # Stop the speech recognizer
            if self.recognizer is not None:
                self.recognizer.stop()
        
        logger.info("Conversational assistant stopped")

def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Voice Assistant with Ollama LLM and Kokoro TTS")
    
    # Ollama options
    parser.add_argument("--ollama-model", type=str, default="dolphin-llama3:8b-v2.9-q4_0",
                        help="Ollama model to use (default: dolphin-llama3:8b-v2.9-q4_0)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama API URL (default: http://localhost:11434)")
    
    # TTS options
    parser.add_argument("--voice", type=str, default="bm_lewis",
                        help="TTS voice to use (default: bm_lewis)")
    parser.add_argument("--language", type=str, default="en-us",
                        help="TTS language to use (default: en-us)")
    parser.add_argument("--speed", type=float, default=1.25,
                        help="TTS speed (default: 1.25)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech (default: enabled)")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available TTS voices and exit")
    parser.add_argument("--list-languages", action="store_true",
                        help="List available TTS languages and exit")
    
    # Speech recognition options
    parser.add_argument("--offline", action="store_true",
                        help="Use offline speech recognition with Vosk (default: online with Google)")
    parser.add_argument("--vosk-model", type=str, default="vosk-model-small-en-us",
                        help="Path to Vosk model directory (default: vosk-model-small-en-us)")
    parser.add_argument("--no-speech", action="store_true",
                        help="Disable speech recognition (default: enabled)")
    parser.add_argument("--energy-threshold", type=int, default=300,
                        help="Energy threshold for speech detection (default: 300)")
    parser.add_argument("--vad-level", type=int, default=3, choices=[0, 1, 2, 3],
                        help="VAD aggressiveness level (0-3, default: 3)")
    parser.add_argument("--text-input", type=str,
                        help="Process a single text input and exit")
    parser.add_argument("--cooldown", type=float, default=1.0,
                        help="Cooldown period in seconds after speaking (default: 1.0)")
    parser.add_argument("--max-silence", type=float, default=10.0,
                        help="Maximum silence in seconds before prompting in conversation mode (default: 10.0, 0 to disable)")
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Disable wake word activation (default: enabled)")
    parser.add_argument("--wake-word", type=str, default="hey maxwell",
                        help="Custom wake word to use (default: 'hey maxwell')")
    parser.add_argument("--listen-timeout", type=float, default=15.0,
                        help="Timeout in seconds for speech recognition (default: 15.0)")
    parser.add_argument("--pause-threshold", type=float, default=2.0,
                        help="Pause threshold in seconds for speech recognition (default: 2.0)")
    parser.add_argument("--phrase-time-limit", type=float, default=15.0,
                        help="Maximum phrase time limit in seconds (default: 15.0)")
    parser.add_argument("--auto-download", action="store_true",
                        help="Automatically download missing model files after confirmation")
    parser.add_argument("--no-listen-while-speaking", action="store_true",
                        help="Disable listening while speaking (default: enabled)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--interrupt-word", type=str, default="stop talking",
                        help="Word or phrase to interrupt the assistant (default: 'stop talking')")
    parser.add_argument("--command-prefix", type=str, default="execute",
                        help="Prefix for command execution (default: 'execute')")
    parser.add_argument("--list-commands", action="store_true",
                        help="List available commands and exit")
    
    args = parser.parse_args()
    
    # Set global cooldown period
    global speaking_cooldown, max_silence_before_prompt, wake_word, interrupt_word, command_prefix
    speaking_cooldown = args.cooldown
    max_silence_before_prompt = args.max_silence
    
    # Set wake word
    if args.wake_word:
        wake_word = args.wake_word.lower()
    
    # Set interrupt word
    if args.interrupt_word:
        interrupt_word = args.interrupt_word.lower()
    
    # Set command prefix
    if args.command_prefix:
        command_prefix = args.command_prefix.lower()
    
    logger.info(f"Speaking cooldown set to {speaking_cooldown} seconds")
    logger.info(f"Maximum silence before prompt set to {max_silence_before_prompt} seconds")
    if not args.no_wake_word:
        logger.info(f"Wake word set to '{wake_word}'")
    logger.info(f"Interrupt word set to '{interrupt_word}'")
    logger.info(f"Command prefix set to '{command_prefix}'")
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # List available commands and exit
    if args.list_commands:
        print("\nAvailable commands:")
        for cmd, desc in available_commands.items():
            print(f"  {cmd}: {desc}")
        sys.exit(0)
    
    # Create TTS instance for listing voices/languages
    if args.list_voices or args.list_languages:
        # Can't list voices or languages if TTS is disabled
        if args.no_tts:
            logger.error("Cannot list voices or languages when TTS is disabled (--no-tts)")
            sys.exit(1)
            
        try:
            tts = KokoroTTS(auto_download=args.auto_download)
            if args.list_voices:
                tts.list_available_voices()
            if args.list_languages:
                tts.list_available_languages()
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    
    # Set up signal handlers
    setup_signal_handlers()
    
    # Create and run the assistant
    assistant = ConversationalAssistant(
        ollama_model=args.ollama_model,
        tts_voice=args.voice,
        language=args.language,
        speed=args.speed,
        vad_aggressiveness=args.vad_level,
        use_offline_recognition=args.offline,
        vosk_model_path=args.vosk_model,
        skip_tts=args.no_tts,
        skip_speech=args.no_speech,
        energy_threshold=args.energy_threshold,
        use_wake_word=not args.no_wake_word,
        custom_wake_word=args.wake_word,
        listen_timeout=args.listen_timeout,
        pause_threshold=args.pause_threshold,
        phrase_time_limit=args.phrase_time_limit,
        auto_download=args.auto_download,
        enable_listen_while_speaking=not args.no_listen_while_speaking,
        custom_interrupt_word=args.interrupt_word,
        custom_command_prefix=args.command_prefix
    )
    
    try:
        assistant.run(args.text_input)
    except Exception as e:
        logger.error(f"Error running assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 