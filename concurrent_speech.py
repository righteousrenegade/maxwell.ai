#!/usr/bin/env python3
"""
Concurrent speech recognition module for the voice assistant.

This module contains classes for non-blocking speech recognition that can run
in a separate thread while the assistant is speaking.
"""

# Standard library imports
import os
import sys
import time
import json
import queue
import logging
import threading

# Third-party imports
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import webrtcvad

# Optional imports for offline speech recognition
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# Import utility functions and global variables
from utils import (
    logger, is_speaking, last_speaking_time, speaking_cooldown, 
    listen_while_speaking, interrupt_word, command_prefix, is_interrupted,
    print_listening_message, print_heard_message, is_interrupt_command
)

class ConcurrentSpeechRecognizer:
    """Speech recognizer that runs in a separate thread for non-blocking operation."""
    
    def __init__(self, vad_aggressiveness=3, sample_rate=16000, language="en-US", 
                 use_offline=False, vosk_model_path="vosk-model-small-en-us", energy_threshold=300,
                 pause_threshold=2.0, phrase_time_limit=15.0, listen_timeout=15.0):
        """Initialize the concurrent speech recognizer.
        
        Args:
            vad_aggressiveness: VAD aggressiveness level (0-3)
            sample_rate: Audio sample rate
            language: Language for speech recognition
            use_offline: Whether to use offline recognition
            vosk_model_path: Path to Vosk model directory
            energy_threshold: Minimum energy threshold for speech detection
            pause_threshold: Pause threshold in seconds
            phrase_time_limit: Maximum phrase time limit in seconds
            listen_timeout: Timeout for listening in seconds
        """
        self.use_offline = use_offline
        self.sample_rate = sample_rate
        self.language = language
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        self.phrase_time_limit = phrase_time_limit
        self.listen_timeout = listen_timeout
        
        # Initialize the speech queue for communication between threads
        self.speech_queue = queue.Queue()
        
        # Flag to control the listening thread
        self.stop_listening = False
        
        # Flag to indicate if the recognizer is currently listening
        self.is_listening = False
        
        # Initialize the recognizer based on the mode
        if use_offline:
            # Check if Vosk is available
            if not VOSK_AVAILABLE:
                logger.error("Vosk is not available. Install it with: pip install vosk")
                sys.exit(1)
                
            # Check if model exists
            if not os.path.exists(vosk_model_path):
                logger.error(f"Vosk model not found: {vosk_model_path}")
                logger.info("Download a model from: https://alphacephei.com/vosk/models")
                sys.exit(1)
                
            try:
                # Initialize Vosk model and recognizer
                self.model = Model(vosk_model_path)
                self.recognizer = KaldiRecognizer(self.model, sample_rate)
                
                # Initialize VAD
                self.vad = webrtcvad.Vad(vad_aggressiveness)
                
                # Audio parameters
                self.chunk_size = int(sample_rate * 0.1)  # 100ms chunks
                self.audio_buffer = queue.Queue()
                
                logger.info(f"Vosk speech recognizer initialized with model: {vosk_model_path}")
            except Exception as e:
                logger.error(f"Error initializing Vosk: {e}")
                sys.exit(1)
        else:
            # Initialize online recognizer
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = energy_threshold
            self.recognizer.pause_threshold = pause_threshold
            self.recognizer.phrase_time_limit = phrase_time_limit
            
            # Set up a separate recognizer with lower energy threshold for keyword spotting
            self.keyword_recognizer = sr.Recognizer()
            # Use a lower energy threshold for keywords to make them easier to detect
            self.keyword_recognizer.energy_threshold = max(energy_threshold * 0.7, 200)
            self.keyword_recognizer.pause_threshold = 0.5  # Shorter pause for keywords
            self.keyword_recognizer.phrase_time_limit = 3.0  # Shorter phrase time for keywords
            
            logger.info(f"Online speech recognizer initialized with language: {language}")
            logger.info(f"Energy threshold: {energy_threshold}")
            logger.info(f"Pause threshold: {pause_threshold}s")
            logger.info(f"Phrase time limit: {phrase_time_limit}s")
            logger.info(f"Keyword spotting configured with energy threshold: {self.keyword_recognizer.energy_threshold}")
        
        # Initialize the listening thread
        self.listening_thread = None
    
    def start_listening(self):
        """Start the listening thread."""
        if self.listening_thread is not None and self.listening_thread.is_alive():
            logger.info("Listening thread is already running")
            return
        
        # Reset the stop flag
        self.stop_listening = False
        
        # Create and start the listening thread
        self.listening_thread = threading.Thread(target=self._listening_loop)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        
        logger.info("Started continuous listening thread")
    
    def stop(self):
        """Stop the listening thread."""
        self.stop_listening = True
        
        if self.listening_thread is not None and self.listening_thread.is_alive():
            logger.info("Stopping listening thread...")
            self.listening_thread.join(timeout=2.0)
            logger.info("Listening thread stopped")
    
    def get_speech(self, block=False, timeout=None):
        """Get recognized speech from the queue.
        
        Args:
            block: Whether to block until speech is available
            timeout: Timeout in seconds for blocking
            
        Returns:
            Recognized speech text or None if no speech is available
        """
        try:
            return self.speech_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream in offline mode.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time: Time info
            status: Status flags
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        # Add audio data to buffer
        self.audio_buffer.put(bytes(indata))
    
    def _is_speech(self, audio_chunk):
        """Check if audio chunk contains speech for offline mode.
        
        Args:
            audio_chunk: Audio chunk to check
            
        Returns:
            True if speech detected, False otherwise
        """
        # Convert to 16-bit PCM
        audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Calculate energy
        energy = np.sqrt(np.mean(np.square(audio_chunk, dtype=np.float32)))
        
        # Check if energy is above threshold
        if energy > self.energy_threshold:
            return True
            
        # If energy is low, use VAD for more accurate detection
        try:
            # Ensure we have the right number of samples for VAD
            # VAD requires 10, 20, or 30 ms frames at 8, 16, or 32 kHz
            frame_duration_ms = 30  # 30 ms frames
            samples_per_frame = int(self.sample_rate * frame_duration_ms / 1000)
            
            # If we don't have enough samples, return False
            if len(audio_chunk) < samples_per_frame * 2:
                return False
                
            # Use only the first frame for VAD
            frame = audio_chunk[:samples_per_frame * 2]
            
            # Check if frame contains speech
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            logger.debug(f"Error in VAD: {e}")
            return False
    
    def _offline_listening_loop(self):
        """Continuous listening loop for offline recognition."""
        global is_speaking, last_speaking_time, speaking_cooldown, is_interrupted, waiting_for_user_input
        
        # Start audio stream
        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype='int16',
                channels=1,
                callback=self._audio_callback
            ):
                logger.debug("Started continuous offline listening loop")
                
                # Variables for VAD
                silent_chunks = 0
                voiced_frames = 0
                has_speech = False
                speech_start_time = None
                min_speech_duration = 0.3  # Minimum speech duration in seconds to consider it valid
                max_silence_chunks = 30  # About 3 seconds of silence
                post_speech_buffer = 20  # Continue listening for 2 seconds after speech ends
                
                # For partial recognition
                partial_text = ""
                last_partial_check_time = time.time()
                partial_check_interval = 0.3  # Check for partial results every 0.3 seconds
                
                # Reset the recognizer
                self.recognizer.Reset()
                
                # Main listening loop
                while not self.stop_listening:
                    # Get audio chunk from buffer
                    try:
                        audio_chunk = self.audio_buffer.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    # Check for voice activity
                    is_speech = self._is_speech(audio_chunk)
                    
                    if is_speech:
                        # If this is the first speech chunk, log it
                        if not has_speech and not self.is_listening:
                            self.is_listening = True
                            logger.debug("Speech detected, started listening")
                        
                        silent_chunks = 0
                        voiced_frames += 1
                        
                        # Record when speech started
                        if not has_speech:
                            speech_start_time = time.time()
                            logger.debug("Speech started")
                            
                        has_speech = True
                    elif has_speech:
                        silent_chunks += 1
                    
                    # Process audio with recognizer
                    if self.recognizer.AcceptWaveform(audio_chunk):
                        result = json.loads(self.recognizer.Result())
                        if result.get("text", ""):
                            logger.debug(f"Intermediate result: {result['text']}")
                    
                    # Check for partial results periodically
                    current_time = time.time()
                    if is_speaking and current_time - last_partial_check_time > partial_check_interval:
                        last_partial_check_time = current_time
                        
                        # Get partial result
                        partial_result = json.loads(self.recognizer.PartialResult())
                        if partial_result.get("partial", ""):
                            new_partial = partial_result["partial"].lower()
                            
                            # Check if the partial result contains the interrupt word
                            if is_interrupt_command(new_partial):
                                logger.info(f"Interrupt command detected in partial result: {new_partial}")
                                is_interrupted = True
                                
                                # Add the partial text to the queue
                                self.speech_queue.put(new_partial)
                                
                                # Reset for next speech
                                has_speech = False
                                voiced_frames = 0
                                silent_chunks = 0
                                self.recognizer.Reset()
                                self.is_listening = False
                                continue
                            
                            # Update partial text
                            partial_text = new_partial
                    
                    # End if we've had enough silence after speech
                    if has_speech and silent_chunks > max_silence_chunks + post_speech_buffer:
                        # But only if we've had a reasonable amount of speech
                        if voiced_frames > 10:  # Ensure we have enough speech frames
                            logger.debug(f"Ending listening after {silent_chunks} silent chunks")
                            
                            # Get final result
                            result = json.loads(self.recognizer.FinalResult())
                            text = result.get("text", "").strip()
                            
                            # Check if speech duration was long enough
                            speech_duration = 0
                            if speech_start_time is not None:
                                speech_duration = time.time() - speech_start_time
                                logger.debug(f"Speech duration: {speech_duration:.2f}s")
                                
                            if speech_duration < min_speech_duration:
                                logger.debug(f"Speech too short ({speech_duration:.2f}s < {min_speech_duration}s), ignoring")
                            else:
                                # Filter out very short responses (likely noise)
                                if text and len(text.strip()) < 2:
                                    logger.debug(f"Filtered out too short response: '{text}'")
                                elif text:
                                    logger.info(f"Recognized (Vosk): {text}")
                                    
                                    # Print and log what was heard
                                    print_heard_message(text)
                                    
                                    # Check for interrupt word if we're speaking
                                    if is_speaking and is_interrupt_command(text):
                                        logger.info(f"Interrupt command detected: {text}")
                                        is_interrupted = True
                                    
                                    # Add the recognized text to the queue
                                    self.speech_queue.put(text)
                            
                            # Reset for next speech
                            has_speech = False
                            voiced_frames = 0
                            silent_chunks = 0
                            self.recognizer.Reset()
                            self.is_listening = False
                
                logger.debug("Exiting offline listening loop")
        except Exception as e:
            logger.error(f"Error in offline listening loop: {e}")
        finally:
            self.is_listening = False
    
    def _online_listening_loop(self):
        """Continuous listening loop for online recognition."""
        global is_speaking, last_speaking_time, speaking_cooldown, is_interrupted, waiting_for_user_input
        
        # Initialize microphone
        mic = sr.Microphone()
        
        # Adjust for ambient noise
        logger.info("Adjusting for ambient noise...")
        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.keyword_recognizer.adjust_for_ambient_noise(source)
            # Keep keyword recognizer threshold lower
            self.keyword_recognizer.energy_threshold = max(self.recognizer.energy_threshold * 0.7, 200)
        
        logger.debug("Started continuous online listening loop")
        
        # Main listening loop
        while not self.stop_listening:
            try:
                # If we're speaking, prioritize listening for the interrupt word
                if is_speaking:
                    try:
                        with mic as source:
                            # Use the keyword recognizer with a short timeout
                            logger.debug("Listening for interrupt word...")
                            
                            # Use a much lower energy threshold when speaking
                            original_energy = self.keyword_recognizer.energy_threshold
                            self.keyword_recognizer.energy_threshold = max(original_energy * 0.6, 150)
                            
                            audio = self.keyword_recognizer.listen(
                                source,
                                timeout=0.2,  # Very short timeout for better responsiveness
                                phrase_time_limit=0.5  # Very short phrase time for quick detection
                            )
                            
                            # Restore original energy threshold
                            self.keyword_recognizer.energy_threshold = original_energy
                            
                            # Try to recognize with a shorter timeout
                            try:
                                text = self.keyword_recognizer.recognize_google(audio, language=self.language)
                                
                                # Check if the text contains the interrupt word
                                if is_interrupt_command(text):
                                    logger.info(f"Interrupt command detected: {text}")
                                    is_interrupted = True
                                    
                                    # Add the text to the queue with high priority
                                    self.speech_queue.put(text)
                                    
                                    # No delay for interrupt commands - we want immediate response
                                    continue
                            except (sr.UnknownValueError, sr.RequestError):
                                # No speech detected or error, continue with normal listening
                                pass
                    except (sr.WaitTimeoutError, Exception) as e:
                        # Timeout or other error, continue with normal listening
                        if isinstance(e, Exception) and not isinstance(e, sr.WaitTimeoutError):
                            logger.debug(f"Error in keyword spotting: {e}")
                        pass
                
                # Normal listening for commands or conversation
                with mic as source:
                    # Set dynamic energy threshold for better noise filtering
                    self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, self.energy_threshold)
                    self.recognizer.dynamic_energy_threshold = True
                    self.recognizer.dynamic_energy_adjustment_damping = 0.15
                    self.recognizer.dynamic_energy_ratio = 1.5
                    
                    # If we're speaking, use a lower energy threshold
                    if is_speaking:
                        original_energy = self.recognizer.energy_threshold
                        self.recognizer.energy_threshold = max(original_energy * 0.8, 200)
                    
                    # Set listening parameters
                    self.recognizer.pause_threshold = self.pause_threshold
                    self.recognizer.non_speaking_duration = self.pause_threshold
                    self.recognizer.operation_timeout = None
                    
                    # Listen for audio with timeout
                    logger.debug("Listening for speech...")
                    self.is_listening = True
                    
                    try:
                        audio = self.recognizer.listen(
                            source, 
                            timeout=1.0,  # Shorter timeout for more responsive listening
                            phrase_time_limit=self.phrase_time_limit
                        )
                        
                        # Restore original energy threshold if we modified it
                        if is_speaking:
                            self.recognizer.energy_threshold = original_energy
                        
                        # Recognize speech
                        try:
                            text = self.recognizer.recognize_google(audio, language=self.language)
                            
                            # Filter out very short responses (likely noise)
                            if len(text.strip()) < 2:
                                logger.debug(f"Filtered out too short response: '{text}'")
                                self.is_listening = False
                                continue
                                
                            logger.info(f"Recognized (Google): {text}")
                            
                            # Print and log what was heard
                            print_heard_message(text)
                            
                            # Check for interrupt word if we're speaking
                            if is_speaking and is_interrupt_command(text):
                                logger.info(f"Interrupt command detected: {text}")
                                is_interrupted = True
                            
                            # Add the recognized text to the queue
                            self.speech_queue.put(text)
                            
                            # Small delay to avoid rapid recognition
                            time.sleep(0.2)
                        except sr.UnknownValueError:
                            logger.debug("Google Speech Recognition could not understand audio")
                        except sr.RequestError as e:
                            logger.error(f"Could not request results from Google Speech Recognition service: {e}")
                    except sr.WaitTimeoutError:
                        # Timeout, continue listening
                        pass
                    finally:
                        self.is_listening = False
            except Exception as e:
                logger.error(f"Error in online listening loop: {e}")
                # Small delay to avoid rapid error loops
                time.sleep(0.2)
        
        logger.debug("Exiting online listening loop")
    
    def _listening_loop(self):
        """Main listening loop that runs in a separate thread."""
        # Choose the appropriate listening loop based on the recognition mode
        if self.use_offline:
            self._offline_listening_loop()
        else:
            self._online_listening_loop() 