#!/usr/bin/env python3
"""
Speech recognition module for the voice assistant.

This module contains classes for speech recognition:
- SpeechRecognizer: Main speech recognition class
- VoskSpeechRecognizer: Speech recognizer using Vosk for offline recognition
"""

# Standard library imports
import os
import sys
import time
import json
import queue
import logging

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
    print_listening_message, print_heard_message
)

class VoskSpeechRecognizer:
    """Speech recognizer using Vosk for offline recognition."""
    
    def __init__(self, model_path="vosk-model-small-en-us", sample_rate=16000, energy_threshold=300):
        """Initialize Vosk speech recognizer.
        
        Args:
            model_path: Path to Vosk model directory
            sample_rate: Audio sample rate
            energy_threshold: Minimum energy threshold for speech detection
        """
        if not VOSK_AVAILABLE:
            logger.error("Vosk is not available. Install it with: pip install vosk")
            sys.exit(1)
            
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Vosk model not found: {model_path}")
            logger.info("Download a model from: https://alphacephei.com/vosk/models")
            sys.exit(1)
            
        try:
            # Initialize Vosk model and recognizer
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, sample_rate)
            
            # Initialize VAD
            self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
            
            # Audio parameters
            self.sample_rate = sample_rate
            self.chunk_size = int(sample_rate * 0.1)  # 100ms chunks
            self.buffer = queue.Queue()
            self.silence_threshold = 5  # Number of silent chunks to end recording
            self.is_listening = False
            self.energy_threshold = energy_threshold
            
            # Initialize keyword spotting for interrupt word
            self.interrupt_word = interrupt_word
            self.command_prefix = command_prefix
            
            logger.info(f"Vosk speech recognizer initialized with model: {model_path}")
        except Exception as e:
            logger.error(f"Error initializing Vosk: {e}")
            sys.exit(1)
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time: Time info
            status: Status flags
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        # Add audio data to buffer
        self.buffer.put(bytes(indata))
    
    def _is_speech(self, audio_chunk):
        """Check if audio chunk contains speech.
        
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
    
    def listen_for_speech(self, timeout=10):
        """Listen for speech and return the recognized text.
        
        Args:
            timeout: Maximum time to listen in seconds
            
        Returns:
            Recognized text or None if no speech detected
        """
        global is_speaking, last_speaking_time, speaking_cooldown, is_interrupted
        
        # Don't listen while speaking
        if is_speaking and not listen_while_speaking:
            return None
            
        # Don't listen during cooldown period after speaking
        time_since_speaking = time.time() - last_speaking_time
        if not is_speaking and time_since_speaking < speaking_cooldown:
            logger.debug(f"In cooldown period ({time_since_speaking:.2f}s < {speaking_cooldown}s), skipping listening")
            return None
            
        self.is_listening = True
        self.recognizer.Reset()  # Reset the recognizer
        
        # Print and log a clear message that we're listening
        print_listening_message()
        
        # Start audio stream
        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype='int16',
                channels=1,
                callback=self._audio_callback
            ):
                logger.debug("Listening for speech...")
                
                # Variables for VAD
                silent_chunks = 0
                voiced_frames = 0
                start_time = time.time()
                has_speech = False
                speech_start_time = None
                min_speech_duration = 0.3  # Minimum speech duration in seconds to consider it valid
                max_silence_chunks = 30  # Increased from 15 to allow more pauses in speech (about 3 seconds)
                post_speech_buffer = 20  # Continue listening for 2 seconds after speech ends
                
                # For partial recognition
                partial_text = ""
                last_partial_check_time = time.time()
                partial_check_interval = 0.3  # Check for partial results every 0.3 seconds
                
                # Listen until timeout or silence after speech
                while time.time() - start_time < timeout:
                    # Check if we should stop
                    if is_speaking and not listen_while_speaking:
                        break
                        
                    # Get audio chunk from buffer
                    try:
                        audio_chunk = self.buffer.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    # Check for voice activity with more aggressive filtering
                    is_speech = self._is_speech(audio_chunk)
                    
                    if is_speech:
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
                            if (self.interrupt_word in new_partial or 
                                "stop" in new_partial or 
                                "shut up" in new_partial):
                                logger.info(f"Interrupt word detected in partial result: {new_partial}")
                                is_interrupted = True
                                
                                # Return the partial text with the interrupt word
                                self.is_listening = False
                                return new_partial
                            
                            # Update partial text
                            partial_text = new_partial
                    
                    # End if we've had enough silence after speech
                    if has_speech and silent_chunks > max_silence_chunks + post_speech_buffer:
                        # But only if we've had a reasonable amount of speech
                        if voiced_frames > 10:  # Ensure we have enough speech frames
                            logger.debug(f"Ending listening after {silent_chunks} silent chunks")
                            break
                
                # Get final result
                result = json.loads(self.recognizer.FinalResult())
                text = result.get("text", "").strip()
                
                # Check if speech duration was long enough
                speech_duration = 0
                if has_speech and speech_start_time is not None:
                    speech_duration = time.time() - speech_start_time
                    logger.debug(f"Speech duration: {speech_duration:.2f}s")
                    
                if speech_duration < min_speech_duration:
                    logger.debug(f"Speech too short ({speech_duration:.2f}s < {min_speech_duration}s), ignoring")
                    return None
                
                # Filter out very short responses (likely noise)
                if text and len(text.strip()) < 2:
                    logger.debug(f"Filtered out too short response: '{text}'")
                    return None
                
                if text:
                    logger.info(f"Recognized (Vosk): {text}")
                    
                    # Print and log what was heard
                    print_heard_message(text)
                    
                    # Check for interrupt word in final result
                    text_lower = text.lower()
                    if is_speaking and (self.interrupt_word in text_lower or 
                                        "stop" in text_lower or 
                                        "shut up" in text_lower):
                        logger.info(f"Interrupt word detected in final result: {text_lower}")
                        is_interrupted = True
                    
                    return text
                else:
                    logger.debug("No speech detected")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in Vosk speech recognition: {e}")
            return None
        finally:
            self.is_listening = False

class SpeechRecognizer:
    """Speech recognizer using Google Speech Recognition or Vosk."""
    
    def __init__(self, vad_aggressiveness=3, sample_rate=16000, language="en-US", 
                 use_offline=False, vosk_model_path="vosk-model-small-en-us", energy_threshold=300,
                 pause_threshold=2.0, phrase_time_limit=15.0, listen_timeout=15.0):
        """Initialize the speech recognizer.
        
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
        
        if use_offline:
            # Initialize offline recognizer
            self.offline_recognizer = VoskSpeechRecognizer(
                model_path=vosk_model_path,
                sample_rate=sample_rate,
                energy_threshold=energy_threshold
            )
            logger.info(f"Using offline speech recognition with Vosk model: {vosk_model_path}")
        else:
            # Initialize online recognizer
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = energy_threshold
            self.recognizer.pause_threshold = pause_threshold
            self.recognizer.phrase_time_limit = phrase_time_limit
            logger.info(f"Using online speech recognition with language: {language}")
            logger.info(f"Energy threshold: {energy_threshold}")
            logger.info(f"Pause threshold: {pause_threshold}s")
            logger.info(f"Phrase time limit: {phrase_time_limit}s")
        
        # Initialize keyword spotting for interrupt word and command prefix
        self.interrupt_word = interrupt_word
        self.command_prefix = command_prefix
        
        # Set up a separate recognizer with lower energy threshold for keyword spotting
        if not use_offline:
            self.keyword_recognizer = sr.Recognizer()
            # Use a lower energy threshold for keywords to make them easier to detect
            self.keyword_recognizer.energy_threshold = max(energy_threshold * 0.7, 200)
            self.keyword_recognizer.pause_threshold = 0.5  # Shorter pause for keywords
            self.keyword_recognizer.phrase_time_limit = 3.0  # Shorter phrase time for keywords
            logger.info(f"Keyword spotting configured with energy threshold: {self.keyword_recognizer.energy_threshold}")
    
    def adjust_for_ambient_noise(self, source, duration=1):
        """Adjust for ambient noise.
        
        Args:
            source: Audio source
            duration: Duration to sample ambient noise
        """
        if self.use_offline:
            # For offline recognition, we don't need to adjust for ambient noise
            logger.info("Skipping ambient noise adjustment for offline recognition")
            return
        
        logger.info(f"Adjusting for ambient noise (duration: {duration}s)...")
        self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        logger.info(f"Adjusted energy threshold: {self.recognizer.energy_threshold}")
        
        # Also adjust the keyword recognizer
        if hasattr(self, 'keyword_recognizer'):
            self.keyword_recognizer.adjust_for_ambient_noise(source, duration=duration)
            # Keep it lower than the main recognizer
            self.keyword_recognizer.energy_threshold = max(self.recognizer.energy_threshold * 0.7, 200)
            logger.info(f"Adjusted keyword energy threshold: {self.keyword_recognizer.energy_threshold}")
    
    def _check_for_interrupt_word(self, text):
        """Check if the text contains an interrupt word.
        
        Args:
            text: Text to check
            
        Returns:
            True if the text contains an interrupt word, False otherwise
        """
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Check for various interrupt phrases
        interrupt_phrases = [
            self.interrupt_word,
            "stop",
            "shut up",
            "be quiet",
            "silence",
            "quiet",
            "enough",
            "pause",
            "hold on"
        ]
        
        return any(phrase in text_lower for phrase in interrupt_phrases)
    
    def listen_for_speech(self, source=None, timeout=None):
        """Listen for speech and return the recognized text.
        
        Args:
            source: Audio source (for online recognition)
            timeout: Timeout in seconds
            
        Returns:
            Recognized text or None if no speech detected
        """
        global is_speaking, last_speaking_time, speaking_cooldown, listen_while_speaking, is_interrupted
        
        # Don't listen while speaking if the feature is disabled
        if is_speaking and not listen_while_speaking:
            return None
        
        # Don't listen during cooldown period after speaking, unless we're currently speaking
        # This allows listening while speaking but prevents immediate listening after speech ends
        time_since_speaking = time.time() - last_speaking_time
        if not is_speaking and time_since_speaking < speaking_cooldown:
            logger.debug(f"In cooldown period ({time_since_speaking:.2f}s < {speaking_cooldown}s), skipping listening")
            return None
        
        # If we're speaking, prioritize listening for the interrupt word
        if is_speaking and not self.use_offline and source is not None:
            try:
                # Use the keyword recognizer with a short timeout
                logger.debug("Listening for interrupt word...")
                
                # Use a much lower energy threshold when speaking to make it easier to detect interrupts
                original_energy = self.keyword_recognizer.energy_threshold
                self.keyword_recognizer.energy_threshold = max(original_energy * 0.6, 150)
                
                audio = self.keyword_recognizer.listen(
                    source,
                    timeout=0.5,  # Very short timeout for responsiveness
                    phrase_time_limit=1.5  # Short phrase time for quick detection
                )
                
                # Restore original energy threshold
                self.keyword_recognizer.energy_threshold = original_energy
                
                # Try to recognize with a shorter timeout
                try:
                    text = self.keyword_recognizer.recognize_google(audio, language=self.language, show_all=False)
                    text = text.lower()
                    
                    # Check if the text contains the interrupt word
                    if self._check_for_interrupt_word(text):
                        logger.info(f"Interrupt word detected: {text}")
                        is_interrupted = True
                        return text
                except (sr.UnknownValueError, sr.RequestError):
                    # No speech detected or error, continue with normal listening
                    pass
            except (sr.WaitTimeoutError, Exception) as e:
                # Timeout or other error, continue with normal listening
                if isinstance(e, Exception) and not isinstance(e, sr.WaitTimeoutError):
                    logger.debug(f"Error in keyword spotting: {e}")
                pass
        
        # Print and log a clear message that we're listening
        print_listening_message()
        
        try:
            if self.use_offline:
                # Use offline recognition
                if self.offline_recognizer is None:
                    logger.error("Offline recognition requested but Vosk recognizer not initialized")
                    return None
                
                # Use the provided timeout, fall back to self.listen_timeout, or use 15.0 as default
                effective_timeout = timeout if timeout is not None else self.listen_timeout
                logger.debug(f"Using timeout: {effective_timeout}s for offline recognition")
                
                # Get the speech text
                speech_text = self.offline_recognizer.listen_for_speech(timeout=effective_timeout)
                
                # Check for interrupt word if we're speaking
                if is_speaking and speech_text and self._check_for_interrupt_word(speech_text):
                    logger.info(f"Interrupt word detected in offline mode: {speech_text}")
                    is_interrupted = True
                
                return speech_text
            else:
                # Use online recognition
                if source is None:
                    logger.error("Source is required for online recognition")
                    return None
                
                # Adjust for ambient noise to improve recognition
                self.adjust_for_ambient_noise(source)
                
                # Set dynamic energy threshold for better noise filtering
                self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, self.energy_threshold)
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.dynamic_energy_adjustment_damping = 0.15
                self.recognizer.dynamic_energy_ratio = 1.5
                
                # Increase phrase timeout and pause threshold for longer sentences
                self.recognizer.pause_threshold = self.pause_threshold  # Use configured pause threshold
                self.recognizer.non_speaking_duration = self.pause_threshold  # Same as pause threshold
                self.recognizer.operation_timeout = None  # No timeout for operations
                
                # Use the provided timeout, fall back to self.listen_timeout
                effective_timeout = timeout if timeout is not None else self.listen_timeout
                logger.debug(f"Using timeout: {effective_timeout}s for online recognition")
                
                # If we're speaking, use a lower energy threshold to make it easier to detect speech
                if is_speaking:
                    original_energy = self.recognizer.energy_threshold
                    self.recognizer.energy_threshold = max(original_energy * 0.8, 200)
                
                # Listen for audio with phrase timeout
                logger.debug(f"Listening for speech with parameters: pause_threshold={self.pause_threshold}s, phrase_time_limit={self.phrase_time_limit}s")
                audio = self.recognizer.listen(
                    source, 
                    timeout=effective_timeout,
                    phrase_time_limit=self.phrase_time_limit,  # Use configured phrase time limit
                    snowboy_configuration=None
                )
                
                # Restore original energy threshold if we modified it
                if is_speaking:
                    self.recognizer.energy_threshold = original_energy
                
                # Recognize speech
                try:
                    # Use a longer timeout for recognition
                    text = self.recognizer.recognize_google(audio, language=self.language, show_all=False)
                    
                    # Filter out very short responses (likely noise)
                    if len(text.strip()) < 2:
                        logger.debug(f"Filtered out too short response: '{text}'")
                        return None
                        
                    logger.info(f"Recognized (Google): {text}")
                    
                    # Print and log what was heard
                    print_heard_message(text)
                    
                    # Check for interrupt word if we're speaking
                    if is_speaking and self._check_for_interrupt_word(text):
                        logger.info(f"Interrupt word detected: {text}")
                        is_interrupted = True
                    
                    return text
                except sr.UnknownValueError:
                    logger.debug("Google Speech Recognition could not understand audio")
                    return None
                except sr.RequestError as e:
                    logger.error(f"Could not request results from Google Speech Recognition service: {e}")
                    return None
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return None 