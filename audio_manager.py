import time
import speech_recognition as sr
import logging
import os
import traceback
import threading
from utils import setup_logger

# Get the logger instance
logger = logging.getLogger("maxwell")
if not logger.handlers:
    logger = setup_logger()

class AudioManager:
    """
    Centralized manager for all audio input and processing.
    Only ONE instance of this class should exist in the entire application.
    """
    def __init__(self, mic_index=None, energy_threshold=300):
        self.mic_index = mic_index
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.5
        
        # Callbacks
        self.on_speech_detected = None
        self.on_speech_recognized = None
        
        # Wake word and interrupt word
        self.wake_word = None
        self.interrupt_word = None
        self.in_conversation = False
        
        # Background listening
        self.listening_thread = None
        self.should_stop = False
        self.microphone = None
        self.last_audio_timestamp = 0
        
        # Debug info
        self.last_recognition_time = 0
        self.recognition_count = 0
        self.recognition_attempts = 0
        self.recognition_successes = 0
        
        # List available microphones
        self._list_microphones()
        
        logger.info(f"AudioManager initialized with energy threshold: {energy_threshold}")
        
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
    
    def _print_status_periodically(self):
        """Periodically print listening status"""
        now = time.time()
        if now - self.last_audio_timestamp > 5:  # Every 5 seconds with no audio
            print("👂 Still listening... (no speech detected)")
            logger.info("Still listening for speech input...")
            self.last_audio_timestamp = now
    
    def _background_listening_thread(self):
        """Background thread that continuously listens for speech"""
        logger.info("Background listening thread started")
        print("🎤 Microphone activated and listening")
        
        # Create and configure microphone
        try:
            with self.microphone as source:
                # Adjust for ambient noise once at the beginning
                logger.info("Adjusting for ambient noise...")
                print("⚙️ Calibrating microphone for ambient noise...")
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    logger.info(f"Energy threshold after adjustment: {self.recognizer.energy_threshold}")
                    print(f"✅ Microphone calibrated (sensitivity: {self.recognizer.energy_threshold:.1f})")
                except Exception as e:
                    logger.error(f"Error adjusting for ambient noise: {e}")
                    print("❌ Failed to calibrate microphone")
                    return
                
                self.last_audio_timestamp = time.time()
                listen_count = 0
                    
                # Main listening loop
                while not self.should_stop:
                    try:
                        # Use a very short timeout to allow checking the stop flag frequently
                        listen_count += 1
                        if listen_count % 3 == 0:  # Every 3 listen attempts
                            self._print_status_periodically()
                            
                        logger.debug("Listening for speech...")
                        logger.info(f"👂 Waiting for speech... (listen attempt #{listen_count})")
                        
                        # This is where speech recognition happens
                        audio = self.recognizer.listen(source, timeout=1)
                        
                        # Audio was detected!
                        self.last_audio_timestamp = time.time()
                        print("🔊 Audio detected! Processing...")
                        logger.info(f"Audio detected (length: {len(audio.frame_data)/audio.sample_rate:.2f}s)")
                        
                        # Check if we should stop
                        if self.should_stop:
                            logger.info("Stop flag detected after audio capture, exiting loop")
                            break
                        
                        # Process the audio
                        self._speech_callback(self.recognizer, audio)
                        
                    except sr.WaitTimeoutError:
                        # This is normal, just continue
                        logger.debug("No speech detected in timeout period")
                        # Explicitly check the stop flag here
                        if self.should_stop:
                            logger.info("Stop flag detected in timeout handler, exiting loop")
                            break
                        continue
                    except Exception as e:
                        if self.should_stop:
                            # We're shutting down, so this is expected
                            logger.info("Stop flag detected in exception handler, exiting loop")
                            break
                        
                        logger.error(f"Error in listening thread: {e}")
                        logger.error(traceback.format_exc())
                        print(f"❌ Error while listening: {str(e)}")
                        
                        # Small delay to prevent tight loop on error
                        time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in background thread: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ Critical error in listening thread: {str(e)}")
                    
        logger.info("Background listening thread stopped")
        print("🛑 Microphone listening stopped")
    
    def _speech_callback(self, recognizer, audio):
        """Callback function for processing audio"""
        try:
            # Check if we should stop
            if self.should_stop:
                logger.info("Stop flag detected in speech callback, skipping processing")
                return
                
            # Try to recognize the speech
            self.recognition_attempts += 1
            logger.debug(f"Attempting speech recognition (attempt {self.recognition_attempts})...")
            print("🧠 Recognizing speech...")
            
            start_time = time.time()
            text = recognizer.recognize_google(audio)
            duration = time.time() - start_time
            
            self.recognition_successes += 1
            logger.info(f"Speech recognized in {duration:.2f}s: '{text}'")
            print(f"✓ Recognized: \"{text}\" ({duration:.1f}s)")
            
            # Update debug info
            self.last_recognition_time = time.time()
            self.recognition_count += 1
            
            # Check for wake word if not in conversation
            if not self.in_conversation and self.wake_word:
                logger.debug(f"Checking for wake word '{self.wake_word}' in '{text}'")
                if self._check_wake_word(text):
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
            
            # Check for interrupt word
            if self.interrupt_word and self.interrupt_word in text.lower():
                logger.info(f"Interrupt word detected in: '{text}'")
                print(f"🛑 Interrupt word detected!")
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
                print("💤 Not in conversation mode - say the wake word to activate")
            
        except sr.UnknownValueError:
            # This is normal, just continue
            logger.debug("Speech recognition could not understand audio")
            print("❓ Couldn't understand audio")
            pass
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition service error: {e}")
            print(f"⚠️ Speech recognition service error: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ Error processing audio: {str(e)}")
        
    def start(self, wake_word=None, interrupt_word=None):
        """Start the background listening"""
        if self.listening_thread and self.listening_thread.is_alive():
            logger.warning("AudioManager is already running")
            return
            
        self.wake_word = wake_word.lower() if wake_word else None
        self.interrupt_word = interrupt_word.lower() if interrupt_word else None
        logger.info(f"Starting AudioManager with wake_word='{self.wake_word}', interrupt_word='{self.interrupt_word}'")
        print(f"🎤 Starting speech recognition with wake word: '{self.wake_word}'")
        
        try:
            # Initialize microphone
            if self.mic_index is not None:
                logger.info(f"Opening microphone with index {self.mic_index}")
                try:
                    self.microphone = sr.Microphone(device_index=self.mic_index)
                except Exception as e:
                    logger.error(f"Failed to open microphone with index {self.mic_index}: {e}")
                    logger.info("Falling back to default microphone")
                    print(f"⚠️ Failed to open microphone {self.mic_index}, falling back to default")
                    self.microphone = sr.Microphone()
            else:
                logger.info("Opening default microphone")
                self.microphone = sr.Microphone()
            
            # Reset stop flag
            self.should_stop = False
            
            # Start background listening thread
            logger.info("Starting background listening thread...")
            self.listening_thread = threading.Thread(
                target=self._background_listening_thread,
                daemon=True
            )
            self.listening_thread.start()
            
            logger.info("AudioManager started successfully")
            print("✅ Speech recognition started successfully")
            
        except Exception as e:
            logger.error(f"Error starting AudioManager: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ Failed to start speech recognition: {str(e)}")
            raise
    
    def stop(self):
        """Stop the background listening"""
        logger.info("Stopping AudioManager...")
        print("🛑 Stopping speech recognition...")
        
        if not self.listening_thread or not self.listening_thread.is_alive():
            logger.info("No active listening thread to stop")
            return
            
        # Signal thread to stop
        self.should_stop = True
        logger.info("Set should_stop flag to True")
        
        # Wait for thread to terminate
        logger.info("Waiting for listening thread to stop...")
        self.listening_thread.join(timeout=2)
        
        if self.listening_thread.is_alive():
            logger.warning("Listening thread did not stop within timeout, continuing anyway")
            print("⚠️ Listening thread did not stop cleanly (timeout)")
        else:
            logger.info("Listening thread stopped successfully")
            print("✅ Listening thread stopped successfully")
            
        logger.info("AudioManager stopped")
        
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
            "running": self.listening_thread is not None and self.listening_thread.is_alive(),
            "in_conversation": self.in_conversation,
            "last_recognition_time": self.last_recognition_time,
            "recognition_count": self.recognition_count,
            "energy_threshold": self.recognizer.energy_threshold,
            "recognition_attempts": self.recognition_attempts,
            "recognition_successes": self.recognition_successes
        } 