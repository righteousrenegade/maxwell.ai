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
from speech import TextToSpeech, DummyTTS
from commands import CommandExecutor
from utils import setup_logger, download_models
from config import Config
import random
import json
import socket
import struct
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
from key_handler import KeyboardHandler

# Version
__version__ = "1.0.0"

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
        
        # Initialize audio manager and mcp provider
        self.audio_manager = None
        self.mcp_tool_provider = None
        self.tool_provider_started = False
        
        # Set wake and interrupt words
        self.wake_word = config.get('wake_word', 'maxwell').lower()
        self.interrupt_word = config.get('interrupt_word', 'stop talking').lower()
        
        # Initialize components
        logger.info("Initializing Text-to-Speech...")
        # Use system TTS by default (or pyttsx3 if specified)
        tts_provider = config.get('tts_provider', 'kokoro')
        tts_voice = config.get('voice')
        tts_speed = config.get('speed', 1.0)
        
        if tts_provider == 'none':
            # Dummy TTS that doesn't do anything
            logger.info("TTS disabled by configuration (tts_provider=none)")
            self.tts = DummyTTS()
        else:
            # System TTS (platform specific)
            self.tts = TextToSpeech(voice=tts_voice, speed=tts_speed, config=config)
        
        # Initialize keyboard handler for space bar interruption
        logger.info("Initializing keyboard handler...")
        self.keyboard_handler = KeyboardHandler(self.tts)
        
        # MCP Tool Provider setup
        self.enable_mcp = config.get('use_mcp', True)
        if self.enable_mcp and HAS_MCP_TOOLS:
            logger.info("Initializing MCP Tool Provider...")
            self.mcp_tool_provider = MCPToolProvider(self)
            self.tool_provider_started = False
        else:
            if self.enable_mcp:
                logger.warning("MCP tools requested but not available")
            self.mcp_tool_provider = None
        
        # Initialize Command Executor (will be set after initialization)
        logger.info("Initializing Command Executor...")
        self.command_executor = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"StreamingMaxwell initialized with wake word: '{self.wake_word}'")
        
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
            # Stop the keyboard handler
            if hasattr(self, 'keyboard_handler'):
                logger.info("Stopping keyboard handler...")
                self.keyboard_handler.stop()
            
            # Clean up TTS
            if hasattr(self, 'tts'):
                logger.info("Cleaning up TTS...")
                self.tts.cleanup()
                
            # Clean up AudioManager
            if hasattr(self, 'audio_manager') and self.audio_manager:
                logger.info("Stopping audio manager...")
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
                
            # Use direct audio playback - will be stopped by space bar handler
            self.tts.speak(text)
            
            # Wait for speech to complete
            while self.tts.is_speaking():
                time.sleep(0.1)
            
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
        """Handle a user query using MCP tools when possible"""
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
        
        # Special handling for "execute" commands
        query_lower = query.lower()
        if query_lower.startswith("execute "):
            # Extract the command part
            command_text = query[8:].strip()
            logger.info(f"Execute command detected: '{command_text}'")
            
            # Check for "execute search" pattern
            if command_text.lower().startswith("search "):
                search_term = command_text[7:].strip()
                logger.info(f"Execute search command detected: '{search_term}'")
                
                # Use MCP tools for search if available
                if self.enable_mcp and self.mcp_tool_provider:
                    logger.info(f"Using MCP search_web tool for: '{search_term}'")
                    print(f"🔍 Searching for: '{search_term}'")
                    response = self.mcp_tool_provider.execute_tool("search_web", query=search_term)
                    if response:
                        self.speak(response)
                    return
        
        # MCP tools integration - check if tools are available
        if self.enable_mcp and self.mcp_tool_provider:
            # Start the tool provider if not already started
            if not self.tool_provider_started:
                logger.info("Starting MCP tool provider...")
                self.mcp_tool_provider.start_server()
                self.tool_provider_started = True
                
                # Register MCP tools with command executor
                if self.command_executor:
                    logger.info("Registering MCP tools with command executor")
                    self.command_executor._register_mcp_tools()
                
            # Get available tools
            available_tools = self.mcp_tool_provider.get_tool_descriptions()
            logger.debug(f"Available tools: {available_tools}")
            
            # Direct mapping for common commands to MCP tools
            command_to_tool = {
                "time": {"tool": "get_time"},
                "what time is it": {"tool": "get_time"},
                "what's the time": {"tool": "get_time"},
                "tell me the time": {"tool": "get_time"},
                
                "date": {"tool": "get_date"},
                "what day is it": {"tool": "get_date"},
                "what's the date": {"tool": "get_date"},
                "tell me the date": {"tool": "get_date"},
                
                "joke": {"tool": "tell_joke"},
                "tell me a joke": {"tool": "tell_joke"},
                
                "weather": {"tool": "get_weather"},
                "what's the weather": {"tool": "get_weather"}
            }
            
            # Check for direct matches from query to tool
            for command, tool_info in command_to_tool.items():
                if command in query_lower:
                    tool_name = tool_info["tool"]
                    logger.info(f"Executing MCP tool: {tool_name}")
                    print(f"🛠️ Processing with MCP tool: '{tool_name}'")
                    response = self.mcp_tool_provider.execute_tool(tool_name)
                    if response:
                        self.speak(response)
                    return
            
            # Handle search commands - improved detection
            search_keywords = ["search", "search for", "look up", "find", "find information about"]
            is_search_command = False
            search_term = ""
            
            for keyword in search_keywords:
                if query_lower.startswith(keyword + " "):
                    is_search_command = True
                    search_term = query_lower.replace(keyword, "", 1).strip()
                    break
            
            if is_search_command and search_term:
                logger.info(f"Search command detected - search term: '{search_term}'")
                print(f"🔍 Processing search for: '{search_term}'")
                response = self.mcp_tool_provider.execute_tool("search_web", query=search_term)
                if response:
                    self.speak(response)
                return
                
            # Weather with location
            weather_patterns = ["weather in ", "weather for ", "what's the weather in ", "what is the weather in "]
            for pattern in weather_patterns:
                if pattern in query_lower:
                    location = query_lower.split(pattern, 1)[1].strip()
                    logger.info(f"Weather command detected with location: {location}")
                    print(f"🌤️ Getting weather for {location}...")
                    response = self.mcp_tool_provider.execute_tool("get_weather", location=location)
                    if response:
                        self.speak(response)
                    return
            
            # Set reminder
            if "remind me" in query_lower or "set a reminder" in query_lower:
                logger.info("Reminder command detected")
                reminder_text = query_lower.replace("remind me", "").replace("set a reminder", "").replace("to", "", 1).strip()
                response = self.mcp_tool_provider.execute_tool("set_reminder", text=reminder_text)
                if response:
                    self.speak(response)
                return
                
            # Set timer
            timer_patterns = ["set a timer for ", "timer for ", "set timer for "]
            for pattern in timer_patterns:
                if pattern in query_lower:
                    duration = query_lower.split(pattern, 1)[1].strip()
                    logger.info(f"Timer command detected: {duration}")
                    response = self.mcp_tool_provider.execute_tool("set_timer", duration=duration)
                    if response:
                        self.speak(response)
                    return
            
            # Play music
            if "play music" in query_lower or "play some music" in query_lower:
                logger.info("Play music command detected")
                response = self.mcp_tool_provider.execute_tool("play_music")
                if response:
                    self.speak(response)
                return
                
            # Play specific song or artist
            if "play " in query_lower:
                # Extract what to play
                play_request = query_lower.replace("play ", "", 1).strip()
                logger.info(f"Play specific music: {play_request}")
                
                # Check if it's a song by an artist
                if " by " in play_request:
                    song, artist = play_request.split(" by ", 1)
                    response = self.mcp_tool_provider.execute_tool("play_music", song=song, artist=artist)
                else:
                    # Could be a song name or artist name
                    response = self.mcp_tool_provider.execute_tool("play_music", song=play_request)
                
                if response:
                    self.speak(response)
                return
                
        # If we get here and have a command executor, fall back to it
        if hasattr(self, 'command_executor') and self.command_executor:
            # Use the LLM/command executor as fallback
            try:
                logger.info("No direct MCP tool match - using command executor")
                print("💭 Processing with command executor...")
                response = self.command_executor.execute(query)
                if response:
                    self.speak(response)
            except Exception as e:
                logger.error(f"Error handling query with command executor: {e}")
                self.speak("I'm sorry, I encountered an error processing your request.")
        else:
            # No command executor available
            logger.error("No command executor or MCP tools available to handle the query")
            self.speak("I'm sorry, I'm not able to process that request right now.")
    
    def run(self):
        """Main loop for the assistant"""
        logger.info("StreamingMaxwell is running")
        print("\n🤖 Maxwell Assistant is starting up...")
        
        try:
            # Start the keyboard handler
            self.keyboard_handler.start(self.tts)
            logger.info("Keyboard handler started - SPACE key will interrupt speech")
            
            # Initialize the command executor if not already done
            if not self.command_executor:
                logger.info("Initializing command executor...")
                from commands import CommandExecutor
                self.command_executor = CommandExecutor(self, self.config)
                logger.info("Command executor initialized")
            
            # Start MCP tool provider if enabled
            if self.enable_mcp and self.mcp_tool_provider:
                logger.info("Starting MCP tool provider...")
                self.mcp_tool_provider.start_server()
                self.tool_provider_started = True
                print(f"🔧 MCP tools integration enabled")
                
                # Log available tools
                available_tools = self.mcp_tool_provider.get_tool_descriptions()
                tool_names = list(available_tools.keys())
                logger.info(f"Available MCP tools: {', '.join(tool_names)}")
                
                # Register MCP tools with command executor
                if self.command_executor:
                    logger.info("Registering MCP tools with command executor")
                    self.command_executor._register_mcp_tools()
            
            # Initialize the LLM provider based on configuration
            llm_provider = getattr(self.config, 'llm_provider', 'none').lower()
            
            # Print LLM provider information
            if llm_provider == 'openai':
                print(f"🧠 Using OpenAI API ({self.config.get('openai_model', 'unknown model')})")
                logger.info(f"Using OpenAI API with model {self.config.get('openai_model', 'unknown')}")
                print(f"   API URL: {self.config.get('openai_base_url', 'unknown')}")
            elif llm_provider == 'ollama':
                model = getattr(self.config, 'model', 'unknown')
                base_url = getattr(self.config, 'ollama_base_url', 'unknown')
                print(f"🧠 Using Ollama ({model})")
                logger.info(f"Using Ollama with model {model} at {base_url}")
            elif llm_provider == 'none':
                print("🧠 No LLM provider configured")
                logger.info("No LLM provider configured")
            else:
                print(f"🧠 Using {llm_provider} provider")
                logger.info(f"Using provider: {llm_provider}")
            
            # Ensure the wake word is set
            if not self.wake_word and hasattr(self.config, 'wake_word'):
                self.wake_word = self.config.wake_word
                
            if not self.wake_word:
                self.wake_word = "maxwell"  # Default wake word
                
            # Initialize the audio manager if not already done
            if not self.audio_manager:
                mic_index = getattr(self.config, 'mic_index', None)
                energy_threshold = getattr(self.config, 'energy_threshold', 300)
                
                self.audio_manager = StreamingAudioManager(
                    mic_index=mic_index,
                    energy_threshold=energy_threshold
                )
                
                # Set callbacks
                self.audio_manager.on_speech_detected = self._on_speech_detected
                self.audio_manager.on_speech_recognized = self._on_speech_recognized
                
            # Start the audio manager
            print(f"🎤 Listening for wake word: '{self.wake_word}'")
            self.audio_manager.start(wake_word=self.wake_word, interrupt_word=self.interrupt_word)
            
            # Announce startup
            print("\n🚀 Maxwell is ready!")
            self.speak(f"Hello, Maxwell here. Say '{self.wake_word}' to get my attention.")
            
            # Print some helpful info
            print(f"🔊 Say '{self.wake_word}' followed by your query. For example:")
            print(f"🗣️  '{self.wake_word}, what time is it?'")
            print(f"🗣️  '{self.wake_word}, tell me a joke'")
            print("")
            print("⌨️  Press SPACE to interrupt speech")
            print("⌨️  Press Ctrl+C to exit")
            
            # Main loop - just keep the process alive
            while not self.should_stop:
                time.sleep(0.1)
                
            return True
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            print("\n👋 Shutting down by keyboard interrupt...")
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ Error: {e}")
            self.cleanup()
            
        return False
    
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
        
        # Get available commands
        commands = []
        if hasattr(self, 'command_executor') and self.command_executor:
            try:
                commands = sorted([cmd for cmd in self.command_executor.available_commands.keys() 
                                if len(cmd.split()) == 1 and cmd not in ["what", "what's", "tell"]])
            except Exception as e:
                logger.error(f"Error getting available commands: {e}")
                commands = ["time", "date", "weather", "help"]
        else:
            # Fallback to basic commands
            commands = ["time", "date", "weather", "help"]
            
        print("You can use these direct commands:")
        print(", ".join(commands))
        print("="*60 + "\n")
        print("💡 To use a command, say: 'execute [command]'")
        print("💡 Example: 'execute time' or 'execute joke'")
        if check_command:
            print(f"💡 To check connectivity, use 'execute {check_command}' command.")

def main():
    """Main entry point for the streaming assistant"""
    import argparse
    import platform
    import logging
    import traceback  # Make sure traceback is imported
    import json
    import sys
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Streaming Audio Assistant")
        
        # Config options
        parser.add_argument('--config', help="Path to YAML config file")
        parser.add_argument('--env-file', help="Path to .env file")
        
        # Microphone options
        parser.add_argument('--list-mics', action='store_true', help="List available microphones and exit")
        parser.add_argument('--mic-index', type=int, help="Microphone index to use")
        parser.add_argument('--energy-threshold', type=int, help="Energy level for mic to detect")
        parser.add_argument('--record-path', help="Path to save recorded audio files")
        parser.add_argument('--offline', action='store_true', help="Use offline speech recognition (Vosk)")
        
        # TTS options
        parser.add_argument('--tts-provider', choices=['system', 'pyttsx3', 'none'], 
                          help="TTS provider to use")
        parser.add_argument('--voice', help="Voice to use for TTS")
        parser.add_argument('--speed', type=float, help="Speed for TTS")
        
        # LLM provider options
        parser.add_argument('--llm-provider', 
                          choices=['openai', 'ollama', 'none'], 
                          help="LLM provider to use")
        
        # OpenAI options
        parser.add_argument('--openai-api-key', help="OpenAI API key")
        parser.add_argument('--openai-base-url', help="OpenAI API base URL")
        parser.add_argument('--openai-model', help="OpenAI model name to use")
        
        # Ollama options
        parser.add_argument('--ollama-host', help="Ollama server host")
        parser.add_argument('--ollama-port', type=int, help="Ollama server port")
        parser.add_argument('--ollama-model', help="Ollama model to use")
        
        # Conversation options
        parser.add_argument('--wake-word', help="Wake word to listen for")
        parser.add_argument('--interrupt-word', help="Word to interrupt the assistant")
        parser.add_argument('--no-convo', action='store_true', 
                          help="Exit after handling one query")
        parser.add_argument('--test-mode', action='store_true', 
                          help="Immediately enter conversation mode for testing")
        parser.add_argument('--keyboard-mode', action='store_true', 
                          help="Use keyboard instead of microphone for input")
        parser.add_argument('--always-listen', action='store_true', 
                          help="Always listen mode - no wake word needed")
                          
        # MCP integration
        parser.add_argument('--use-mcp', action='store_true', help="Use MCP tool provider integration")
        parser.add_argument('--mcp-port', type=int, help="MCP API port")
        
        # Debug options
        parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose output")
        parser.add_argument('--debug', action='store_true', help="Enable debug mode")
        
        # Parse the arguments
        args = parser.parse_args()
        
        # Handle microphone listing
        if args.list_mics:
            list_microphones()
            return
            
        # Set up logging
        log_level = logging.INFO
        if args.verbose or args.debug:
            log_level = logging.DEBUG
            
        # Configure logger
        logger = setup_logger(log_level)
        logger.info(f"Starting Maxwell assistant v{__version__}")
        
        # Log platform information
        logger.info(f"Platform: {platform.system()} {platform.release()} ({platform.version()})")
        logger.info(f"Python version: {platform.python_version()}")
        
        # Load configuration using new config_loader system
        try:
            from config_loader import load_config, create_config_object
            
            # Load configuration from files
            logger.info("Loading configuration using new config system")
            config_dict = load_config(yaml_path=args.config, env_path=args.env_file)
            
            # Create config object
            config = create_config_object(config_dict)
            
            # Override with any command line arguments that were explicitly provided
            arg_dict = vars(args)
            for key, value in arg_dict.items():
                if value is not None and key != 'config' and key != 'env_file':
                    # Special case for some parameters that have different names
                    if key == 'ollama_host' and value:
                        config.ollama_host = value
                    elif key == 'ollama_port' and value:
                        config.ollama_port = value
                    elif key == 'ollama_model' and value:
                        config.model = value
                    else:
                        setattr(config, key, value)
                    logger.debug(f"[CONFIG] Overriding {key} with command line value: {value}")
            
            # Set defaults for critical parameters if not provided
            if not hasattr(config, 'ollama_base_url') and hasattr(config, 'ollama_host') and hasattr(config, 'ollama_port'):
                config.ollama_base_url = f"http://{config.ollama_host}:{config.ollama_port}"
                
            # Debug log the final config
            logger.debug(f"Final configuration: {config}")
            
        except ImportError:
            # Fall back to old configuration system if config_loader isn't available
            logger.warning("config_loader not found, falling back to old config system")
            
            # Create a config object from the arguments
            config = argparse.Namespace()
            
            # Set up the configuration from arguments
            # Microphone settings
            config.mic_index = args.mic_index
            config.energy_threshold = args.energy_threshold
            config.record_path = args.record_path
            config.offline = args.offline
            
            # TTS settings
            config.tts_provider = args.tts_provider
            config.voice = args.voice
            config.speed = args.speed
            
            # LLM settings
            config.llm_provider = args.llm_provider
            
            # OpenAI settings
            config.openai_api_key = args.openai_api_key
            config.openai_base_url = args.openai_base_url
            config.openai_model = args.openai_model
            
            # Ollama settings
            if args.ollama_host and args.ollama_port:
                config.ollama_base_url = f"http://{args.ollama_host}:{args.ollama_port}"
            config.ollama_model = args.ollama_model
            
            # Conversation settings
            config.wake_word = args.wake_word
            config.interrupt_word = args.interrupt_word
            config.continuous_conversation = not args.no_convo
            config.test_mode = args.test_mode
            config.keyboard_mode = args.keyboard_mode
            config.always_listen = args.always_listen
            
            # MCP settings
            config.use_mcp = args.use_mcp
            config.mcp_port = args.mcp_port
            
            # Support for getting configuration from a config.py file
            try:
                from config import CONFIG as user_config
                logger.info("Loading user configuration from config.py")
                
                # Apply user configuration without overwriting command line args
                for key, value in user_config.items():
                    if not hasattr(config, key) or getattr(config, key) is None:
                        setattr(config, key, value)
                        
            except ImportError:
                logger.info("No user config.py found, using defaults or command line args")
            except Exception as e:
                logger.error(f"Error loading config.py: {e}")
        
        # Add config properties that should be dictionary-accessible
        config.get = lambda key, default=None: getattr(config, key, default)
            
        # Log the effective configuration (excluding sensitive info)
        config_dict = vars(config).copy()
        
        # Redact sensitive information
        if hasattr(config, 'openai_api_key') and config.openai_api_key:
            config_dict['openai_api_key'] = '[REDACTED]'
            
        # Print important values for debugging purposes
        logger.info("=" * 60)
        logger.info("IMPORTANT CONFIGURATION VALUES:")
        logger.info(f"LLM Provider: {config.get('llm_provider', 'Not set')}")
        logger.info(f"OpenAI Base URL: {config.get('openai_base_url', 'Not set')}")
        logger.info(f"OpenAI API Key: {'[SET]' if config.get('openai_api_key') else '[NOT SET]'}")
        logger.info(f"TTS Model Path: {config.get('tts_model_path', 'Not set')}")
        logger.info(f"TTS Voices Path: {config.get('tts_voices_path', 'Not set')}")
        logger.info("=" * 60)
        
        # Create and initialize the assistant
        assistant = StreamingMaxwell(config)
        
        # Run the assistant
        result = assistant.run()
        return result
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        print("\nProgram interrupted by user. Exiting...")
        return False
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        # Ensure traceback is imported
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    main() 