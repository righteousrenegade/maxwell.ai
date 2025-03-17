#!/usr/bin/env python3
import argparse
import time
import os
import signal
import sys
import logging
import traceback
import threading
import speech_recognition as sr
import pyaudio
import wave
import numpy as np
import tempfile
import winsound
import platform
import ollama
from speech import TextToSpeech
from commands import CommandExecutor
from utils import setup_logger, download_models
from config import Config
import random

# Import MCP tools if available
try:
    from mcp_tools import MCPToolProvider
    HAS_MCP_TOOLS = True
except ImportError:
    HAS_MCP_TOOLS = False
    print("⚠️ MCP tools not available, skipping import")

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

class StreamingAudioManager:
    """Audio manager that uses PyAudio directly for more reliable audio capture"""
    def __init__(self, mic_index=None, energy_threshold=300):
        self.mic_index = mic_index
        self.energy_threshold = energy_threshold
        self.running = False
        self.in_conversation = False
        
        # Audio settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Standard rate for speech recognition
        self.CHUNK = 1024  # Buffer size
        self.RECORD_SECONDS = 7  # Increased maximum recording time to capture longer utterances
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = True  # Enable dynamic threshold adjustment
        self.recognizer.pause_threshold = 0.8  # Longer pause threshold for better sentence completion
        
        # Callbacks
        self.on_speech_detected = None
        self.on_speech_recognized = None
        
        # Wake word and interrupt word
        self.wake_word = None
        self.interrupt_word = None
        
        # Create a temporary directory for audio files
        self.temp_dir = tempfile.mkdtemp()
        
        # Background processing
        self.processing_thread = None
        self.should_stop = False
        
        # Debug info
        self.last_audio_timestamp = time.time()
        self.recognition_count = 0
        self.recognition_attempts = 0
        self.recognition_successes = 0
        self.listen_count = 0  # Add this counter for debug mode
        
        # Add a flag to prevent processing while the assistant is speaking
        self.pause_for_speaking = False
        
        # List available microphones
        self._list_microphones()
        
        # Diagnostics
        self.debug_mode = False
        self.last_volume = 0
        
        logger.info(f"StreamingAudioManager initialized with energy threshold: {energy_threshold}")
        
    def _list_microphones(self):
        """List available microphones for debugging purposes"""
        try:
            mics = sr.Microphone.list_microphone_names()
            logger.info(f"Found {len(mics)} microphones:")
            for i, mic in enumerate(mics):
                logger.info(f"  {i}: {mic}")
                
            if self.mic_index is not None:
                if self.mic_index < len(mics):
                    logger.info(f"Selected microphone {self.mic_index}: {mics[self.mic_index]}")
                else:
                    logger.error(f"Selected microphone index {self.mic_index} is out of range (max: {len(mics)-1})")
            else:
                logger.info("Using default microphone")
                
        except Exception as e:
            logger.error(f"Error listing microphones: {e}")
            logger.error(traceback.format_exc())
    
    def set_conversation_mode(self, enabled):
        """Set whether we're in conversation mode"""
        self.in_conversation = enabled
        logger.info(f"Conversation mode: {enabled}")
        if enabled:
            print("🎯 Conversation mode enabled - listening for commands")
        else:
            print("💤 Conversation mode disabled - waiting for wake word")
    
    def is_in_conversation(self):
        """Check if we're in conversation mode"""
        return self.in_conversation
    
    def _processing_thread_func(self):
        """Background thread that continuously processes audio"""
        logger.info("Audio processing thread started")
        print("🎤 Microphone activated and listening")
        
        self.listen_count = 0
        while not self.should_stop:
            self.listen_count += 1
            
            # Skip audio processing if we're paused for speaking
            if self.pause_for_speaking:
                time.sleep(0.1)  # Small delay
                continue
                
            if self.listen_count % 5 == 0:
                if not self.in_conversation:
                    print("👂 Waiting for wake word...")
                else:
                    print("👂 Listening for commands...")
            
            # Record audio
            audio_data = self._record_audio()
            
            # Check if we should stop
            if self.should_stop:
                logger.info("Stop flag detected, exiting processing loop")
                break
                
            # Check if we got some audio
            if audio_data is None or len(audio_data) == 0:
                time.sleep(0.1)  # Small delay
                continue
                
            # Save audio to temporary file
            temp_file = os.path.join(self.temp_dir, f"audio_{int(time.time())}.wav")
            self._save_audio(audio_data, temp_file)
            
            # Recognize speech
            text = self._recognize_speech(temp_file)
            
            if text:
                # Process the recognized speech
                self._process_speech(text)
                
        logger.info("Audio processing thread stopped")
        print("🛑 Microphone listening stopped")
    
    def _record_audio(self):
        """Record audio from microphone when sound is detected"""
        try:
            # Open stream
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=self.mic_index
            )
            
            # Wait for sound above threshold
            frames = []
            silent_frames = 0
            is_speaking = False
            max_silent_frames = int(self.RATE / self.CHUNK * 1.5)  # Increased silence tolerance to 1.5 seconds
            
            # Start a timer
            start_time = time.time()
            timeout = 0.5  # Half second timeout for checking sound
            
            # Debug volume levels occasionally
            if self.debug_mode and random.random() < 0.1:  # 10% chance to log
                # Read some data to check levels
                debug_data = stream.read(self.CHUNK, exception_on_overflow=False)
                debug_audio = np.frombuffer(debug_data, dtype=np.int16)
                volume = np.abs(debug_audio).mean()
                logger.debug(f"Current volume: {volume:.1f} (threshold: {self.energy_threshold})")
            
            while time.time() - start_time < timeout and not is_speaking and not self.should_stop:
                # Read audio data
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Check volume
                volume = np.abs(audio_data).mean()
                self.last_volume = volume
                
                # Show volume in debug mode
                if self.debug_mode and self.listen_count % 10 == 0:
                    logger.debug(f"Current volume: {volume:.1f} (threshold: {self.energy_threshold})")
                    
                if volume > self.energy_threshold:
                    is_speaking = True
                    frames.append(data)
                    logger.debug(f"Sound detected! Volume: {volume}")
                    print("🔊 Sound detected! Recording...")
                    
                    # Play a beep sound when sound is first detected
                    self._play_sound(frequency=1000, duration=50)  # Short higher-pitched beep
                
            # If no sound detected or should stop, return None
            if not is_speaking or self.should_stop:
                stream.stop_stream()
                stream.close()
                return None
                
            # Record until silence
            recording_start = time.time()
            max_record_time = self.RECORD_SECONDS  # Maximum recording time
            
            while not self.should_stop:
                # Check if we've recorded too long
                if time.time() - recording_start > max_record_time:
                    logger.info("Maximum recording time reached")
                    break
                    
                # Read audio data
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Check volume
                volume = np.abs(audio_data).mean()
                
                if volume < self.energy_threshold:
                    silent_frames += 1
                    if silent_frames > max_silent_frames:
                        logger.debug("Silence detected, stopping recording")
                        break
                else:
                    silent_frames = 0
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
            if self.should_stop:
                return None
                
            logger.info(f"Recording completed, {len(frames)} chunks captured")
            return b''.join(frames)
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _save_audio(self, audio_data, filename):
        """Save audio data to WAV file"""
        if not audio_data:
            return
            
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(audio_data)
            wf.close()
            logger.debug(f"Audio saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
    
    def _recognize_speech(self, audio_file):
        """Recognize speech from audio file with multiple recognition engines"""
        if not os.path.exists(audio_file):
            return None
            
        try:
            # Load the audio file
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                
            # First try with Google (most accurate)
            try:
                self.recognition_attempts += 1
                text = self.recognizer.recognize_google(audio)
                self.recognition_successes += 1
                self.last_audio_timestamp = time.time()
                self.recognition_count += 1
                
                # Play a sound to indicate successful recognition
                self._play_recognition_sound()
                
                logger.info(f"Recognized with Google: '{text}'")
                print(f"✓ Recognized: \"{text}\"")
                return text
            except sr.UnknownValueError:
                logger.debug("Google could not understand audio, trying backup engines")
                
                # Try with Sphinx (offline fallback) if available
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    logger.info(f"Recognized with Sphinx: '{text}'")
                    print(f"✓ Recognized (fallback): \"{text}\"")
                    return text
                except:
                    # If Sphinx is not installed or fails
                    logger.debug("Sphinx recognition failed or not available")
                    pass
                
                print("❓ Couldn't understand what you said")
                return None
                
            except sr.RequestError as e:
                logger.error(f"Recognition error: {e}")
                print(f"⚠️ Recognition service error: {e}")
                
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            print(f"❌ Error: {str(e)}")
            
        return None
    
    def _play_recognition_sound(self):
        """Play a sound to indicate successful speech recognition"""
        try:
            if platform.system() == 'Windows':
                # Use Windows-specific beep
                winsound.Beep(1000, 100)  # 1000Hz for 100ms
            else:
                # Cross-platform alternative
                print('\a', end='', flush=True)  # ASCII bell character
        except Exception as e:
            logger.debug(f"Error playing recognition sound: {e}")
            # Silently fail if sound can't be played
            pass
    
    def _process_speech(self, text):
        """Process recognized speech"""
        if not text or self.should_stop:
            return
            
        text_lower = text.lower()
        
        # Check for interrupt word
        if self.interrupt_word and self.interrupt_word in text_lower:
            logger.info(f"Interrupt word detected in: '{text}'")
            print(f"🛑 Interrupt word detected!")
            if self.on_speech_detected:
                logger.info("Calling interrupt callback")
                self.on_speech_detected("interrupt", text)
            return
        
        # Check for wake word if not in conversation
        if not self.in_conversation:
            wake_word_detected = self._check_wake_word(text)
                
            if wake_word_detected:
                logger.info(f"Wake word detected in: '{text}'")
                print(f"🔔 Wake word detected!")
                self.in_conversation = True
                if self.on_speech_detected:
                    logger.info("Calling wake word callback")
                    self.on_speech_detected("wake_word", text)
                return
            else:
                logger.debug("Wake word not detected")
                print("🔕 Wake word not detected, continuing to listen...")
                return
        
        # Process regular speech if in conversation
        logger.info(f"Processing speech in conversation mode: '{text}'")
        if self.on_speech_recognized:
            logger.info("Calling speech recognized callback")
            self.on_speech_recognized(text)
        else:
            logger.warning("No speech recognized callback registered")
    
    def _check_wake_word(self, text):
        """Check if the wake word is in the text"""
        if not self.wake_word:
            return False
            
        text_lower = text.lower()
        
        # Exact match
        if self.wake_word in text_lower:
            logger.info(f"Wake word exact match: '{self.wake_word}' in '{text_lower}'")
            return True
            
        # Partial match (e.g., "hey max" instead of "hey maxwell")
        wake_parts = self.wake_word.split()
        if len(wake_parts) > 1:
            # Check if the first word and part of the second word match
            if wake_parts[0] in text_lower and wake_parts[1][:3] in text_lower:
                logger.info(f"Wake word partial match: '{wake_parts[0]}' and '{wake_parts[1][:3]}' in '{text_lower}'")
                return True
                
        return False
    
    def start(self, wake_word=None, interrupt_word=None):
        """Start the audio processing"""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("StreamingAudioManager is already running")
            return
            
        self.wake_word = wake_word.lower() if wake_word else None
        self.interrupt_word = interrupt_word.lower() if interrupt_word else None
        logger.info(f"Starting StreamingAudioManager with wake_word='{self.wake_word}', interrupt_word='{self.interrupt_word}'")
        print(f"🎤 Starting speech recognition with wake word: '{self.wake_word}'")
        
        try:
            # Reset stop flag
            self.should_stop = False
            self.running = True
            
            # Start processing thread
            logger.info("Starting audio processing thread...")
            self.processing_thread = threading.Thread(
                target=self._processing_thread_func,
                daemon=True
            )
            self.processing_thread.start()
            
            logger.info("StreamingAudioManager started successfully")
            print("✅ Speech recognition started successfully")
            
        except Exception as e:
            logger.error(f"Error starting StreamingAudioManager: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ Failed to start speech recognition: {str(e)}")
            self.running = False
            raise
    
    def stop(self):
        """Stop the audio processing"""
        logger.info("Stopping StreamingAudioManager...")
        print("🛑 Stopping speech recognition...")
        
        if not self.processing_thread or not self.processing_thread.is_alive():
            logger.info("No active processing thread to stop")
            return
            
        # Signal thread to stop
        self.should_stop = True
        self.running = False
        logger.info("Set should_stop flag to True")
        
        # Wait for thread to terminate
        logger.info("Waiting for processing thread to stop...")
        self.processing_thread.join(timeout=2)
        
        if self.processing_thread.is_alive():
            logger.warning("Processing thread did not stop within timeout, continuing anyway")
            print("⚠️ Processing thread did not stop cleanly (timeout)")
        else:
            logger.info("Processing thread stopped successfully")
            print("✅ Processing thread stopped successfully")
        
        # Clean up PyAudio
        try:
            self.p.terminate()
        except:
            pass
            
        # Remove temporary files
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except:
                    pass
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
                
        logger.info("StreamingAudioManager stopped")
    
    def get_debug_info(self):
        """Get debug information about the audio manager"""
        return {
            "running": self.running,
            "in_conversation": self.in_conversation,
            "last_audio_timestamp": self.last_audio_timestamp,
            "recognition_count": self.recognition_count,
            "energy_threshold": self.energy_threshold,
            "recognition_attempts": self.recognition_attempts,
            "recognition_successes": self.recognition_successes
        }
    
    def pause_listening(self, pause=True):
        """Pause or resume audio processing"""
        self.pause_for_speaking = pause
        logger.debug(f"Audio processing {'paused' if pause else 'resumed'}")
    
    def set_debug_mode(self, enabled=True):
        """Enable or disable debug mode for more verbose audio diagnostics"""
        self.debug_mode = enabled
        logger.info(f"Audio debug mode: {enabled}")
    
    def check_microphone_status(self):
        """Check if the microphone is working and return status"""
        try:
            # Try to open a test stream to check microphone
            test_stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=self.mic_index
            )
            
            # Read some data to test
            data = test_stream.read(self.CHUNK, exception_on_overflow=False)
            
            # Convert to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Check volume
            volume = np.abs(audio_data).mean()
            
            # Close the test stream
            test_stream.stop_stream()
            test_stream.close()
            
            logger.info(f"Microphone test successful. Current volume level: {volume:.1f}")
            
            if volume < 10:
                logger.warning("Microphone volume is very low. This might affect speech recognition.")
                return False, "Volume too low"
                
            return True, f"Volume level: {volume:.1f}"
            
        except Exception as e:
            logger.error(f"Error testing microphone: {e}")
            return False, str(e)
    
    def _play_sound(self, frequency=800, duration=100):
        """Play a beep sound with the given frequency and duration"""
        try:
            if platform.system() == 'Windows':
                # Use Windows-specific beep
                winsound.Beep(frequency, duration)
            else:
                # Cross-platform alternative
                print('\a', end='', flush=True)  # ASCII bell character
        except Exception as e:
            logger.debug(f"Error playing sound: {e}")
            # Silently fail if sound can't be played
            pass

class StreamingMaxwell:
    def __init__(self, config):
        self.config = config
        self.running = True
        self.in_conversation = False
        self.speaking = False
        self.cleaned_up = False
        self.should_stop = False
        
        # Initialize components
        logger.info("Initializing Text-to-Speech...")
        self.tts = TextToSpeech(voice=config.voice, speed=config.speed)
        
        # Initialize AudioManager
        logger.info("Initializing StreamingAudioManager...")
        self.audio_manager = StreamingAudioManager(
            mic_index=config.mic_index, 
            energy_threshold=config.energy_threshold
        )
        
        # Set callbacks
        self.audio_manager.on_speech_detected = self._on_speech_detected
        self.audio_manager.on_speech_recognized = self._on_speech_recognized
        
        # Speech recognition state
        self.wake_word = config.wake_word.lower()
        self.interrupt_word = config.interrupt_word.lower()
        
        logger.info("Initializing Command Executor...")
        self.command_executor = CommandExecutor(self)
        
        # Initialize Ollama client for direct LLM access
        logger.info(f"Initializing Ollama client (host: {config.ollama_host}:{config.ollama_port})...")
        try:
            # Command executor will handle the Ollama client initialization
            pass
        except Exception as e:
            logger.error(f"Error initializing Ollama client: {e}")
            logger.error(traceback.format_exc())
            print(f"⚠️ Warning: Failed to initialize Ollama client: {e}")
        
        # Initialize MCP Tools if enabled
        self.mcp_tool_provider = None
        if config.use_mcp and HAS_MCP_TOOLS:
            logger.info(f"Initializing MCP Tool Provider (port: {config.mcp_port})...")
            self.mcp_tool_provider = MCPToolProvider(self)
            self.mcp_tool_provider.start_server()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"StreamingMaxwell initialized with wake word: '{config.wake_word}'")
        
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
            
            # Clean up MCP Tools if enabled
            if hasattr(self, 'mcp_tool_provider') and self.mcp_tool_provider:
                logger.info("Stopping MCP Tool Provider...")
                self.mcp_tool_provider.stop_server()
                
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
        print("(Press SPACE to interrupt speech)")
        
        try:
            # Play a beep sound when Maxwell starts speaking
            self._play_sound(frequency=900, duration=70)  # Medium-pitched beep
            
            # Pause audio processing to avoid self-listening
            if hasattr(self.audio_manager, 'pause_listening'):
                self.audio_manager.pause_listening(True)
                
            # Add a small delay to ensure audio processing is fully paused
            time.sleep(0.2)
                
            # Use direct audio playback - will be stopped by global keyboard interrupt handler
            self.tts.speak(text)
            
            # Play a beep when Maxwell finishes speaking normally
            self._play_sound(frequency=850, duration=70)  # Medium-pitched beep
            
            # Add a longer pause after speaking to avoid cutting off and prevent immediate listening
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C directly during speech
            print("\n🛑 Speech interrupted by user")
            self.tts.stop()
            
            # Play a beep when speech is interrupted
            self._play_sound(frequency=600, duration=120)  # Lower-pitched longer beep for interruption
        except Exception as e:
            logger.error(f"Error in speak: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Resume audio processing
            if hasattr(self.audio_manager, 'pause_listening'):
                self.audio_manager.pause_listening(False)
                
            # Clear speaking flag
            self.speaking = False
            
            # Log that speaking has finished
            logger.debug("Speaking completed, listening resumed")
    
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
        
        # Special handling for "execute" commands - ensure they're processed as commands
        if query.lower().startswith("execute "):
            try:
                logger.info("Processing explicit execute command - BYPASS LLM")
                print("🛠️ Processing as direct command...")
                response = self.command_executor.execute_command(query[8:].strip())
                if response:
                    self.speak(response)
                return
            except Exception as e:
                logger.error(f"Error processing execute command: {e}")
                logger.error(traceback.format_exc())
                self.speak(f"I'm sorry, I encountered an error processing your command: {str(e)}")
                return
        
        # Handle common search phrases
        if query.lower().startswith("search ") or query.lower().startswith("search for ") or \
           query.lower().startswith("look up ") or query.lower().startswith("find "):
            logger.info("Search command detected - BYPASS LLM")
            print("🔍 Processing as search command...")
            search_term = query.lower().replace("search for ", "").replace("search ", "").replace("look up ", "").replace("find ", "")
            response = self.command_executor.search(search_term)
            if response:
                self.speak(response)
            return
            
        # Check if this is a direct command first
        parts = query.split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        
        if command in self.command_executor.available_commands:
            try:
                logger.info(f"Direct command '{command}' detected - BYPASS LLM")
                print(f"🛠️ Processing as direct '{command}' command...")
                response = self.command_executor.execute(query)
                if response:
                    self.speak(response)
                return
            except Exception as e:
                logger.error(f"Error executing direct command: {e}")
                self.speak(f"I'm sorry, I encountered an error executing that command: {str(e)}")
                return
        
        # Check for other common phrases that map to commands
        common_phrases = {
            "what time is it": "time",
            "what's the time": "time",
            "tell me the time": "time",
            "what day is it": "date",
            "what's the date": "date",
            "tell me the date": "date",
            "tell me a joke": "joke"
        }
        
        for phrase, cmd in common_phrases.items():
            if query.lower().startswith(phrase):
                logger.info(f"Common phrase '{phrase}' detected - BYPASS LLM - using command '{cmd}'")
                print(f"🛠️ Processing as {cmd} command...")
                response = self.command_executor.execute_command(cmd)
                if response:
                    self.speak(response)
                return
                
        # Process through the normal flow (which might use Ollama or commands)
        try:
            # Check if Ollama is available before trying to process the query
            if hasattr(self.command_executor, 'ollama_available') and not self.command_executor.ollama_available:
                # Try to reconnect to Ollama
                reconnected = self.command_executor.check_ollama_connection(force=True)
                if not reconnected:
                    logger.warning("LLM service unavailable, suggesting direct commands instead")
                    self.speak("I'm sorry, the language model service is not available right now. You can use specific commands like 'execute time', 'execute weather', or check the connection with 'execute check ollama'.")
                    return
            
            logger.info("No direct command match - trying LLM or command executor")
            print("💭 Processing normally...")
            # Execute the command/query
            response = self.command_executor.execute(query)
            
            # Speak the response if there is one
            if response:
                self.speak(response)
                
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            logger.error(traceback.format_exc())
            error_msg = str(e)
            
            # Provide more helpful error messages based on error type
            if "connect" in error_msg.lower() or "connection" in error_msg.lower() or "refused" in error_msg.lower():
                self.speak("I'm sorry, I couldn't connect to the language model service. You can use specific commands like 'execute time' or check the connection with 'execute check ollama'.")
            else:
                self.speak("I'm sorry, I encountered an error processing your request. You can try using direct commands or check if the language model service is available.")
    
    def execute_tool(self, tool_name, **kwargs):
        """Execute a tool by name - used by the command executor"""
        if not self.config.use_mcp or not self.mcp_tool_provider:
            logger.warning(f"Tool execution requested but MCP is not enabled: {tool_name}")
            return f"I'm sorry, tool execution is not available."
            
        logger.info(f"Executing tool: {tool_name} with args: {kwargs}")
        result = self.mcp_tool_provider.execute_tool(tool_name, **kwargs)
        logger.info(f"Tool execution result: {result}")
        return result
        
    def get_available_tools(self):
        """Get a list of available tools - used by the command executor"""
        if not self.config.use_mcp or not self.mcp_tool_provider:
            return {}
            
        return self.mcp_tool_provider.get_tool_descriptions()
    
    def run(self):
        logger.info("Maxwell is running. Say the wake word to begin.")
        
        # Check microphone status before starting
        mic_status, message = self.audio_manager.check_microphone_status()
        if not mic_status:
            logger.warning(f"Microphone check warning: {message}")
            print(f"⚠️ Microphone check warning: {message}")
        else:
            logger.info(f"Microphone check passed: {message}")
            
        # Enable audio debug mode if debug logging is enabled
        if logger.level == logging.DEBUG:
            self.audio_manager.set_debug_mode(True)
            logger.debug("Audio debug mode enabled")
        
        # Display LLM provider information
        llm_provider = self.command_executor.llm_provider
        logger.info(f"Using LLM provider: {llm_provider}")
        
        if llm_provider == "ollama":
            # Check if Ollama is available
            ollama_available = self.command_executor.check_ollama_connection(force=True, detailed=True)
            if ollama_available:
                logger.info(f"✅ Ollama LLM service is available with model: {self.config.model}")
                print(f"🧠 Ollama LLM available with model: {self.config.model}")
                print(f"   Host: {self.config.ollama_host}:{self.config.ollama_port}")
            else:
                logger.warning("❌ Ollama LLM service is not available, only direct commands will work")
                self._display_command_only_message("Ollama", "check ollama")
        
        elif llm_provider == "openai":
            # Check if OpenAI is available
            openai_available = self.command_executor.check_openai_connection(force=True, detailed=True)
            if openai_available:
                logger.info(f"✅ OpenAI API is available with model: {self.config.openai_model}")
                print(f"🧠 OpenAI API available with model: {self.config.openai_model}")
                print(f"   Base URL: {self.config.openai_base_url}")
            else:
                logger.warning("❌ OpenAI API is not available, only direct commands will work")
                self._display_command_only_message("OpenAI", "check openai")
                
                # Offer to switch to Ollama if available
                print("\n" + "="*60)
                print("🔄 PROVIDER SWITCHING OPTIONS:")
                print("OpenAI connection failed. You have these options:")
                print("1. Continue with OpenAI provider (commands only, no LLM)")
                print("2. Try using Ollama as fallback")
                print("3. Exit and restart with different options")
                print("="*60)
                
                try:
                    choice = input("Enter your choice (1-3): ").strip()
                    
                    if choice == "2":
                        # Try to switch to Ollama
                        switched = self.command_executor.try_switch_to_ollama()
                        if switched:
                            # Update config to match new provider
                            self.config.llm_provider = "ollama"
                            print("✅ Successfully switched to Ollama provider")
                            # Update provider variables
                            llm_provider = "ollama"
                        else:
                            print("❌ Could not switch to Ollama provider")
                            print("Continuing with OpenAI in command-only mode")
                    elif choice == "3":
                        print("Exiting. Restart with different options.")
                        print("Example: --llm-provider=ollama")
                        self.cleanup()
                        sys.exit(0)
                    else:
                        print("Continuing with OpenAI in command-only mode")
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user. Continuing with current settings.")
                except Exception as e:
                    logger.error(f"Error during provider switch: {e}")
                    print(f"Error: {e}")
                    print("Continuing with current settings.")
        
        else:
            # Unknown provider
            logger.warning(f"❌ Unknown LLM provider: {llm_provider}, only direct commands will work")
            self._display_command_only_message(llm_provider, "")
        
        # Start the audio manager
        self.audio_manager.start(wake_word=self.wake_word, interrupt_word=self.interrupt_word)
        
        # Print instructions for interrupting speech
        print("\n" + "="*60)
        print("💡 KEYBOARD SHORTCUTS:")
        print("  • Press SPACE while Maxwell is speaking to interrupt speech")
        print("  • Press Ctrl+C twice quickly to exit the program")
        print("="*60 + "\n")
        
        # Add more detailed OpenAI connection checks if that's the selected provider
        if llm_provider == "openai":
            print("\n" + "="*60)
            print(f"🔍 CHECKING OPENAI CONNECTION DETAILS:")
            print(f"  • Provider:    {self.config.llm_provider}")
            print(f"  • Base URL:    {self.config.openai_base_url}")
            print(f"  • Model:       {self.config.openai_model}")
            print(f"  • API Key:     {'Set' if self.config.openai_api_key and self.config.openai_api_key != 'None' else 'NOT SET'}")
            
            # Check if API key is missing or set to literal "None"
            if not self.config.openai_api_key or self.config.openai_api_key == "None":
                print("\n⚠️ ERROR: OpenAI API key is not set or is set to 'None'")
                print("Please provide a valid API key with --openai-api-key")
                print("Example: --openai-api-key=sk-your-key-here")
                
                # Try to see if we can switch to offline mode or Ollama
                print("\n💡 SUGGESTION: Switch to Ollama provider or fix the API key")
                print("Example: --llm-provider=ollama")
                print("="*60 + "\n")
            
            # Check if base URL might be invalid
            if "://" not in self.config.openai_base_url:
                print("\n⚠️ WARNING: OpenAI base URL looks invalid (missing protocol)")
                print(f"Current value: {self.config.openai_base_url}")
                print("Should include http:// or https://")
                print("Example: --openai-base-url=https://api.openai.com/v1")
                print("="*60 + "\n")
            
            # If we detect localhost in the URL, check if port is open
            if "localhost" in self.config.openai_base_url or "127.0.0.1" in self.config.openai_base_url:
                import socket
                host = "localhost"
                # Try to extract port from URL
                port = None
                try:
                    port_str = self.config.openai_base_url.split("://")[1].split(":")[1].split("/")[0]
                    port = int(port_str)
                except:
                    print("\n⚠️ WARNING: Could not determine port from OpenAI base URL")
                    print("If you're using a local server, make sure port is specified")
                    print("Example: --openai-base-url=http://localhost:1234/v1")
                    
                if port:
                    print(f"\n🔍 Checking if port {port} is open on {host}...")
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    try:
                        result = s.connect_ex((host, port))
                        if result == 0:
                            print(f"✅ Port {port} is open on {host}")
                        else:
                            print(f"❌ Port {port} is CLOSED on {host}")
                            print(f"Make sure your local OpenAI API server is running on port {port}")
                    except Exception as e:
                        print(f"❌ Error checking port: {e}")
                    finally:
                        s.close()
                        
                print("="*60 + "\n")
        
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
            
            print("\n==== Keyboard Input Mode ====")
            print(f"Type your queries or 'exit' to quit. Type '{self.wake_word}' to begin conversation.")
            
            # Initial state - waiting for wake word if not in test mode
            if not self.config.test_mode and not self.config.always_listen:
                keyboard_in_conversation = False
                print(f"Waiting for wake word '{self.wake_word}'...")
            else:
                keyboard_in_conversation = True
                print("Maxwell is listening for commands...")
                
            while True:
                try:
                    user_input = input("> ").strip()
                    
                    if user_input.lower() == 'exit':
                        logger.info("User requested exit in keyboard mode")
                        print("Exiting Maxwell...")
                        self.stop()
                        break
                        
                    if not keyboard_in_conversation:
                        # Check for wake word when not in conversation
                        if self.wake_word.lower() in user_input.lower():
                            logger.info("Wake word detected in keyboard input")
                            print(f"🔔 Wake word detected!")
                            keyboard_in_conversation = True
                            self.in_conversation = True
                            self.audio_manager.set_conversation_mode(True)
                            self.speak("I'm listening. How can I help?")
                        else:
                            print(f"Waiting for wake word '{self.wake_word}'...")
                            continue
                    else:
                        # We're in conversation mode
                        if "end conversation" in user_input.lower():
                            logger.info("End conversation detected in keyboard input")
                            keyboard_in_conversation = False
                            self.in_conversation = False
                            self.audio_manager.set_conversation_mode(False)
                            self.speak("Ending conversation.")
                            print(f"🔴 Conversation ended. Type '{self.wake_word}' to start again.")
                        else:
                            # Process the query
                            logger.info(f"Processing keyboard input: {user_input}")
                            self.handle_query(user_input)
                            
                except KeyboardInterrupt:
                    # Check if we're speaking - if so, just interrupt the speech
                    if self.speaking:
                        logger.info("KeyboardInterrupt detected during speech in keyboard mode")
                        self.tts.stop()
                        print("\n🛑 Speech interrupted!")
                        continue  # Continue the loop instead of exiting
                    else:
                        # If not speaking, exit the program
                        logger.info("KeyboardInterrupt detected in keyboard mode, exiting")
                        print("\nExiting Maxwell...")
                        self.stop()
                        break
                except Exception as e:
                    logger.error(f"Error in keyboard mode: {e}")
                    logger.error(traceback.format_exc())
                    print(f"Error: {str(e)}")
        
        else:
            # Normal operation mode - wait for the audio processing thread
            last_interrupt_time = 0  # For tracking double Ctrl+C
            
            try:
                # Keep the main thread alive
                while not self.should_stop:
                    try:
                        time.sleep(0.1)
                    except KeyboardInterrupt:
                        # Get current time
                        current_time = time.time()
                        
                        # Check if we're speaking
                        if self.speaking:
                            # Interrupt speech but keep running
                            logger.info("KeyboardInterrupt detected - stopping speech")
                            self.tts.stop()
                            print("\n🛑 Speech interrupted!")
                            # Reset last interrupt time
                            last_interrupt_time = current_time
                        else:
                            # Check if this is a double-press (within 1 second)
                            if current_time - last_interrupt_time < 1.0:
                                # Double Ctrl+C - exit the program
                                logger.info("Double KeyboardInterrupt detected, stopping")
                                print("\nDouble Ctrl+C detected! Stopping Maxwell...")
                                self.stop()
                                break
                            else:
                                # Single Ctrl+C - just note it and continue
                                print("\nPress Ctrl+C again within 1 second to exit")
                                last_interrupt_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in main thread: {e}")
                logger.error(traceback.format_exc())
                self.stop()
                
        logger.info("Maxwell has stopped")
        print("Maxwell has stopped. Goodbye!")
            
    def _play_listening_sound(self):
        """Play a sound to indicate we're listening"""
        try:
            if platform.system() == 'Windows':
                # Use Windows-specific beep
                winsound.Beep(800, 200)  # 800Hz for 200ms
            else:
                # Cross-platform alternative
                print('\a', end='', flush=True)  # ASCII bell character
        except Exception as e:
            logger.debug(f"Error playing listening sound: {e}")
            # Silently fail if sound can't be played
            pass
            
        # Visual indicator
        print("\n🔵 Listening...")

    def stop(self):
        """Stop the assistant"""
        logger.info("Stopping assistant...")
        self.should_stop = True
        self.running = False
        self.cleanup()

    def _play_sound(self, frequency=800, duration=100):
        """Play a beep sound with the given frequency and duration"""
        try:
            if platform.system() == 'Windows':
                # Use Windows-specific beep
                winsound.Beep(frequency, duration)
            else:
                # Cross-platform alternative
                print('\a', end='', flush=True)  # ASCII bell character
        except Exception as e:
            logger.debug(f"Error playing sound: {e}")
            # Silently fail if sound can't be played
            pass

    def _display_command_only_message(self, provider_name, check_command):
        """Display a message about command-only mode with available commands"""
        print("\n" + "="*60)
        print(f"⚠️ COMMAND-ONLY MODE: {provider_name} LLM service is not available")
        print("="*60)
        print("You can use these direct commands:")
        commands = sorted([cmd for cmd in self.command_executor.available_commands.keys() 
                         if len(cmd.split()) == 1 and cmd not in ["what", "what's", "tell"]])
        print(", ".join(commands))
        print("="*60 + "\n")
        print("💡 To use a command, say: 'execute [command]'")
        print("💡 Example: 'execute time' or 'execute joke'")
        if check_command:
            print(f"💡 To check connectivity, use 'execute {check_command}' command.")

def main():
    parser = argparse.ArgumentParser(description="Maxwell Voice Assistant (Streaming Edition)")
    parser.add_argument("--wake-word", default="hello maxwell", help="Wake word to activate the assistant")
    parser.add_argument("--interrupt-word", default="stop talking", help="Word to interrupt the assistant")
    parser.add_argument("--voice", default="bm_lewis", help="Voice for text-to-speech")
    parser.add_argument("--speed", default=1.25, type=float, help="Speech speed (1.0 is normal)")
    parser.add_argument("--offline", action="store_true", help="Use offline speech recognition")
    parser.add_argument("--continuous", action="store_true", help="Stay in conversation mode until explicitly ended")
    parser.add_argument("--list-voices", action="store_true", help="List available TTS voices and exit")
    
    # LLM provider options
    llm_group = parser.add_argument_group('LLM Provider Options')
    llm_group.add_argument("--llm-provider", default="openai", choices=["ollama", "openai"], 
                           help="LLM provider to use (ollama or openai)")
    
    # Ollama options
    ollama_group = parser.add_argument_group('Ollama Options')
    ollama_group.add_argument("--model", default="dolphin-llama3:8b-v2.9-q4_0", help="Ollama model to use")
    ollama_group.add_argument("--ollama-host", default="localhost", help="Ollama host address")
    ollama_group.add_argument("--ollama-port", default=11434, type=int, help="Ollama port")
    
    # OpenAI options
    openai_group = parser.add_argument_group('OpenAI Options')
    openai_group.add_argument("--openai-api-key", default="n/a", help="OpenAI API key (optional for local APIs)")
    openai_group.add_argument("--openai-base-url", default="http://localhost:1234/v1",
                             help="OpenAI API base URL (can be a local server URL)")
    openai_group.add_argument("--openai-model", default="gpt-3.5-turbo", 
                             help="OpenAI model name")
    openai_group.add_argument("--openai-system-prompt", 
                             help="System prompt for OpenAI chat completions")
    openai_group.add_argument("--openai-temperature", type=float, default=0.7,
                             help="Temperature for OpenAI completions (0.0-2.0)")
    openai_group.add_argument("--openai-max-tokens", type=int,
                             help="Max tokens for OpenAI completions")
    
    # Other options
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
        # OpenAI options
        llm_provider=args.llm_provider,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model,
        openai_system_prompt=args.openai_system_prompt,
        openai_temperature=args.openai_temperature,
        openai_max_tokens=args.openai_max_tokens
    )
    
    # Log the configuration
    logger.info(f"Starting Maxwell (Streaming Edition) with configuration:")
    for key, value in vars(config).items():
        # Don't log API keys
        if key == "openai_api_key" and value:
            logger.info(f"  {key}: [REDACTED]")
        else:
            logger.info(f"  {key}: {value}")
    
    # Create and run assistant
    assistant = StreamingMaxwell(config)
    try:
        if config.use_mcp and HAS_MCP_TOOLS:
            print(f"🔧 MCP tools integration enabled (port: {config.mcp_port})")
        elif config.use_mcp and not HAS_MCP_TOOLS:
            print("⚠️ MCP tools requested but not available")
            
        assistant.run()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        logger.error(traceback.format_exc())
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    main() 