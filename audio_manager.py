import threading
import queue
import time
import speech_recognition as sr
import numpy as np
import logging
import os
import traceback
import sys

logger = logging.getLogger("maxwell")

class AudioManager:
    """
    Centralized manager for all audio input and processing.
    Only ONE instance of this class should exist in the entire application.
    """
    def __init__(self, mic_index=None, energy_threshold=300):  # Lower default threshold
        self.mic_index = mic_index
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = False  # Disable dynamic threshold
        self.recognizer.pause_threshold = 0.5  # Shorter pause threshold for faster response
        
        # Audio processing thread
        self.running = False
        self.processing_thread = None
        
        # Queue for audio chunks
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Callbacks
        self.on_speech_detected = None
        self.on_speech_recognized = None
        
        # Wake word and interrupt word
        self.wake_word = None
        self.interrupt_word = None
        self.in_conversation = False
        
        # Interrupt signal
        self.interrupt_signal = threading.Event()
        
        # Debug info
        self.last_recognition_time = 0
        self.recognition_count = 0
        self.listen_attempts = 0
        self.listen_successes = 0
        self.recognition_attempts = 0
        self.recognition_successes = 0
        
        # List available microphones and validate the selected index
        self._list_microphones()
        
        logger.info(f"AudioManager initialized with energy threshold: {energy_threshold}")
        logger.info(f"Recognizer settings: dynamic_threshold={self.recognizer.dynamic_energy_threshold}, pause_threshold={self.recognizer.pause_threshold}")
        
    def _list_microphones(self):
        """List available microphones for debugging purposes and validate mic_index"""
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
                    logger.error("This will likely cause errors. Please select a valid microphone index.")
                    # Don't exit here, let the user see the error and available mics
            else:
                logger.info("Using default microphone")
                
        except Exception as e:
            logger.error(f"Error listing microphones: {e}")
            logger.error(traceback.format_exc())
        
    def start(self, wake_word=None, interrupt_word=None):
        """Start the audio processing thread"""
        if self.running:
            logger.warning("AudioManager is already running")
            return
            
        self.wake_word = wake_word.lower() if wake_word else None
        self.interrupt_word = interrupt_word.lower() if interrupt_word else None
        logger.info(f"Starting AudioManager with wake_word='{self.wake_word}', interrupt_word='{self.interrupt_word}'")
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()
        logger.info("AudioManager started")
        
    def stop(self):
        """Stop the audio processing thread"""
        logger.info("Stopping AudioManager...")
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            self.processing_thread = None
        logger.info("AudioManager stopped")
        
    def set_conversation_mode(self, enabled):
        """Set whether we're in conversation mode"""
        self.in_conversation = enabled
        logger.info(f"Conversation mode: {enabled}")
        
    def is_in_conversation(self):
        """Check if we're in conversation mode"""
        return self.in_conversation
        
    def send_interrupt_signal(self):
        """Send a signal to interrupt any ongoing speech"""
        self.interrupt_signal.set()
        logger.info("Interrupt signal sent")
        
    def clear_interrupt_signal(self):
        """Clear the interrupt signal"""
        self.interrupt_signal.clear()
        
    def is_interrupt_signaled(self):
        """Check if an interrupt has been signaled"""
        return self.interrupt_signal.is_set()
        
    def _process_audio_loop(self):
        """Main audio processing loop"""
        logger.info("Audio processing loop started")
        
        while self.running:
            try:
                # Get microphone
                if self.mic_index is not None:
                    logger.info(f"Opening microphone with index {self.mic_index}")
                    try:
                        # Try to create the microphone with the specified index
                        mic = sr.Microphone(device_index=self.mic_index)
                        
                        # Test if the microphone can be opened
                        logger.info("Testing if microphone can be opened...")
                        with mic as source:
                            # Just open and close to test
                            pass
                        logger.info(f"Successfully opened microphone with index {self.mic_index}")
                    except Exception as e:
                        logger.error(f"Failed to open microphone with index {self.mic_index}: {e}")
                        logger.error(traceback.format_exc())
                        
                        # Try to list available microphones again for debugging
                        try:
                            mics = sr.Microphone.list_microphone_names()
                            logger.info(f"Available microphones ({len(mics)}):")
                            for i, mic_name in enumerate(mics):
                                logger.info(f"  {i}: {mic_name}")
                        except Exception as list_err:
                            logger.error(f"Error listing microphones: {list_err}")
                        
                        # Fall back to default microphone
                        logger.info("Falling back to default microphone")
                        mic = sr.Microphone()
                        
                        # Test if the default microphone can be opened
                        try:
                            logger.info("Testing if default microphone can be opened...")
                            with mic as source:
                                # Just open and close to test
                                pass
                            logger.info("Default microphone opened successfully")
                        except Exception as default_err:
                            logger.error(f"Failed to open default microphone: {default_err}")
                            logger.error("No working microphone found. Waiting before retrying...")
                            time.sleep(5)  # Wait before retrying
                            continue  # Skip to the next iteration of the loop
                else:
                    logger.info("Opening default microphone")
                    mic = sr.Microphone()
                    
                    # Test if the default microphone can be opened
                    try:
                        logger.info("Testing if default microphone can be opened...")
                        with mic as source:
                            # Just open and close to test
                            pass
                        logger.info("Default microphone opened successfully")
                    except Exception as default_err:
                        logger.error(f"Failed to open default microphone: {default_err}")
                        logger.error("No working microphone found. Waiting before retrying...")
                        time.sleep(5)  # Wait before retrying
                        continue  # Skip to the next iteration of the loop
                
                logger.info("Microphone opened successfully")
                
                # Use the microphone in a with statement
                with mic as source:
                    # Adjust for ambient noise once at the beginning
                    logger.info("Adjusting for ambient noise (this may take a moment)...")
                    try:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                        logger.info(f"Energy threshold after adjustment: {self.recognizer.energy_threshold}")
                    except Exception as e:
                        logger.error(f"Error adjusting for ambient noise: {e}")
                        logger.error(traceback.format_exc())
                        # Continue with the current threshold
                        logger.info(f"Continuing with current energy threshold: {self.recognizer.energy_threshold}")
                    
                    # Continuous listening loop
                    logger.info("Starting continuous listening loop...")
                    listen_count = 0
                    
                    # Keep listening as long as we're running and the microphone is open
                    while self.running:
                        try:
                            # Log periodically to show we're still listening
                            listen_count += 1
                            if listen_count % 10 == 0:
                                logger.debug(f"Still listening... (attempt {listen_count})")
                            
                            self.listen_attempts += 1
                            logger.debug(f"Listening for audio... (attempt {self.listen_attempts})")
                            
                            # Listen for audio with a short timeout
                            start_time = time.time()
                            audio = self.recognizer.listen(source, timeout=3)
                            duration = time.time() - start_time
                            
                            self.listen_successes += 1
                            logger.debug(f"Audio captured after {duration:.2f}s (success {self.listen_successes}/{self.listen_attempts})")
                            
                            # Process the audio in a separate thread to avoid blocking
                            logger.debug("Starting audio processing thread")
                            threading.Thread(
                                target=self._process_audio,
                                args=(audio,),
                                daemon=True,
                                name=f"AudioProcessor-{self.listen_successes}"
                            ).start()
                            
                        except sr.WaitTimeoutError:
                            # This is normal, just continue
                            logger.debug("No speech detected in timeout period")
                            continue
                        except Exception as e:
                            logger.error(f"Error in listening loop: {e}")
                            logger.error(traceback.format_exc())
                            # If we get an error with the microphone, break out of the inner loop
                            # to reinitialize the microphone
                            if "Audio source must be entered" in str(e) or "NoneType" in str(e):
                                logger.error("Microphone error detected, reinitializing...")
                                break
                            time.sleep(0.1)  # Prevent tight loop on error
            
            except Exception as e:
                logger.error(f"Error in audio processing main loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1)  # Wait before retrying
                
        logger.info("Audio processing loop stopped")
        
    def _process_audio(self, audio):
        """Process an audio chunk"""
        try:
            # Log audio properties
            logger.debug(f"Processing audio: duration={len(audio.frame_data)/audio.sample_rate:.2f}s, " +
                        f"sample_rate={audio.sample_rate}Hz, frame_count={len(audio.frame_data)}")
            
            # Try to recognize the speech
            self.recognition_attempts += 1
            logger.debug(f"Attempting speech recognition (attempt {self.recognition_attempts})...")
            
            start_time = time.time()
            text = self.recognizer.recognize_google(audio)
            duration = time.time() - start_time
            
            self.recognition_successes += 1
            logger.info(f"Speech recognized in {duration:.2f}s: '{text}'")
            
            # Update debug info
            self.last_recognition_time = time.time()
            self.recognition_count += 1
            
            # Check for wake word if not in conversation
            if not self.in_conversation and self.wake_word:
                logger.debug(f"Checking for wake word '{self.wake_word}' in '{text}'")
                if self._check_wake_word(text):
                    logger.info(f"Wake word detected in: '{text}'")
                    self.in_conversation = True
                    if self.on_speech_detected:
                        logger.info("Calling wake word callback")
                        self.on_speech_detected("wake_word", text)
                    return
                else:
                    logger.debug("Wake word not detected")
            
            # Check for interrupt word
            if self.interrupt_word and self.interrupt_word in text.lower():
                logger.info(f"Interrupt word detected in: '{text}'")
                self.send_interrupt_signal()
                if self.on_speech_detected:
                    logger.info("Calling interrupt callback")
                    self.on_speech_detected("interrupt", text)
                return
            
            # Process regular speech if in conversation
            if self.in_conversation:
                logger.info(f"Processing speech in conversation mode: '{text}'")
                if self.on_speech_recognized:
                    logger.info("Calling speech recognized callback")
                    self.on_speech_recognized(text)
                else:
                    logger.warning("No speech recognized callback registered")
            else:
                logger.debug(f"Ignoring speech (not in conversation mode): '{text}'")
            
        except sr.UnknownValueError:
            # This is normal, just continue
            logger.debug("Speech recognition could not understand audio")
            pass
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition service error: {e}")
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            logger.error(traceback.format_exc())
    
    def _check_wake_word(self, text):
        """Check if the wake word is in the text"""
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
        
    def get_debug_info(self):
        """Get debug information about the audio manager"""
        return {
            "running": self.running,
            "in_conversation": self.in_conversation,
            "last_recognition_time": self.last_recognition_time,
            "recognition_count": self.recognition_count,
            "energy_threshold": self.recognizer.energy_threshold,
            "interrupt_signaled": self.is_interrupt_signaled(),
            "listen_attempts": self.listen_attempts,
            "listen_successes": self.listen_successes,
            "recognition_attempts": self.recognition_attempts,
            "recognition_successes": self.recognition_successes
        } 