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
import collections

# Version
__version__ = "1.0.0"

# Import MCP tools if available
try:
    from mcp_tools import MCPToolProvider
    HAS_MCP_TOOLS = True
except ImportError as e:
    HAS_MCP_TOOLS = False
    print(f"⚠️ MCP tools import error: {str(e)}")
    print("⚠️ This will disable search and other web-based features.")
    import traceback
    traceback.print_exc()
    logger.error(f"Error importing MCP tools: {e}")
    logger.error(traceback.format_exc())

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
    """Audio manager that uses PyAudio directly for more reliable audio capture.
    
    This implementation uses a fully continuous listening approach:
    
    1. The microphone is ALWAYS listening, regardless of whether the assistant is speaking
    2. Audio is continuously recorded when above a dynamic energy threshold
    3. When the assistant is speaking, the energy threshold is raised to filter out self-speech
    4. Detected speech is added to a buffer for asynchronous processing
    5. A separate thread processes the speech buffer independently from the listening thread
    
    This approach allows the assistant to hear the user even while speaking,
    while minimizing self-triggering by dynamically adjusting energy thresholds.
    """
    def __init__(self, mic_index=None, energy_threshold=300):
        self.mic_index = mic_index
        self.energy_threshold = energy_threshold
        self.default_energy_threshold = energy_threshold
        self.speaking_energy_threshold = energy_threshold * 3.0  # Increased ratio to better filter self-speech
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
        
        # Speech buffer for continuous processing
        self.speech_buffer = []
        self.speech_buffer_lock = threading.Lock()
        self.speech_processing_thread = None
        self.is_assistant_speaking = False
        
        # Track speaking state with additional delay
        self.speaking_end_time = 0
        self.post_speaking_delay = 1.5  # Increased delay after speaking to avoid self-triggering
        
        # Recent patterns detection to avoid self-triggering
        self.recent_recognitions = collections.deque(maxlen=5)  # Store recent recognitions
        self.self_speech_filter = []  # Initialize self_speech_filter as an empty list
        
        # Audio characteristics tracking for self-speech detection
        self.audio_stats = {
            "assistant_speech": {
                "volume_avg": [],      # Store average volumes during assistant speech
                "volume_peak": [],     # Store peak volumes during assistant speech
                "spectrum_profile": [] # Store spectrum profiles of assistant speech
            },
            "last_spoken_time": 0
        }
        
        # Anti-feedback measures
        self.consecutive_recognitions = 0
        self.last_recognition_time = 0
        self.feedback_detection_window = 5.0  # seconds
        self.max_consecutive_recognitions = 3
        
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
    
    def _analyze_audio_characteristics(self, audio_data):
        """Analyze audio characteristics to help distinguish self-speech"""
        try:
            # Convert audio data to numpy array if it's not already
            if isinstance(audio_data, bytes):
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_np = audio_data
                
            # Basic volume statistics
            volume_avg = np.abs(audio_np).mean()
            volume_peak = np.abs(audio_np).max()
            
            # Frequency domain analysis (simple spectral profile)
            if len(audio_np) > self.CHUNK:
                # Get a segment for frequency analysis
                segment = audio_np[:self.CHUNK]
                # Calculate FFT
                spectrum = np.abs(np.fft.rfft(segment))
                # Simplify to a few bands (low, mid, high)
                bands = 10
                spectrum_profile = []
                band_size = len(spectrum) // bands
                for i in range(bands):
                    start = i * band_size
                    end = (i + 1) * band_size if i < bands - 1 else len(spectrum)
                    band_energy = np.sum(spectrum[start:end])
                    spectrum_profile.append(band_energy)
                
                # Normalize the profile
                if sum(spectrum_profile) > 0:
                    spectrum_profile = [e / sum(spectrum_profile) for e in spectrum_profile]
            else:
                spectrum_profile = []
                
            return {
                "volume_avg": volume_avg,
                "volume_peak": volume_peak,
                "spectrum_profile": spectrum_profile
            }
        except Exception as e:
            logger.error(f"Error analyzing audio characteristics: {e}")
            return {
                "volume_avg": 0,
                "volume_peak": 0,
                "spectrum_profile": []
            }
    
    def _is_audio_self_speech(self, audio_data):
        """Determine if audio matches characteristics of assistant's speech"""
        # Only use this method if we have collected enough samples
        if len(self.audio_stats["assistant_speech"]["volume_avg"]) < 3:
            return False
            
        # Get stats for this audio
        stats = self._analyze_audio_characteristics(audio_data)
        
        # Compare with assistant speech characteristics
        assistant_avg_volume = np.mean(self.audio_stats["assistant_speech"]["volume_avg"])
        assistant_peak_volume = np.mean(self.audio_stats["assistant_speech"]["volume_peak"])
        
        # Check if the volume characteristics are very close to assistant speech
        # This is a stronger check for when the assistant is currently speaking
        if self.is_assistant_speaking:
            # During speaking, if volume levels are similar, likely self-speech
            vol_ratio = stats["volume_avg"] / assistant_avg_volume if assistant_avg_volume > 0 else 1.0
            peak_ratio = stats["volume_peak"] / assistant_peak_volume if assistant_peak_volume > 0 else 1.0
            
            # If volume characteristics are within 30% of assistant speech, likely self-speech
            if 0.7 < vol_ratio < 1.3 and 0.7 < peak_ratio < 1.3:
                logger.info(f"Self-speech detected during speaking by volume similarity - " +
                          f"vol_ratio={vol_ratio:.2f}, peak_ratio={peak_ratio:.2f}")
                return True
        
        # Volume similarity check
        volume_similarity = abs(stats["volume_avg"] - assistant_avg_volume) / assistant_avg_volume if assistant_avg_volume > 0 else 1.0
        peak_similarity = abs(stats["volume_peak"] - assistant_peak_volume) / assistant_peak_volume if assistant_peak_volume > 0 else 1.0
        
        # Spectrum comparison (if available)
        spectrum_similarity = 1.0  # Default to high dissimilarity
        if stats["spectrum_profile"] and self.audio_stats["assistant_speech"]["spectrum_profile"]:
            # Use the most recent spectrum profile
            assistant_spectrum = self.audio_stats["assistant_speech"]["spectrum_profile"][-1]
            
            # Only compare if both profiles have values
            if assistant_spectrum and stats["spectrum_profile"] and len(assistant_spectrum) == len(stats["spectrum_profile"]):
                # Calculate Euclidean distance between spectrum profiles
                try:
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(assistant_spectrum, stats["spectrum_profile"])))
                    spectrum_similarity = min(distance, 1.0)  # Normalize between 0-1
                except Exception as e:
                    logger.error(f"Error calculating spectrum similarity: {e}")
        
        # Calculate overall similarity score (weighted) - lower weights to make less sensitive
        similarity_score = (volume_similarity * 0.3) + (peak_similarity * 0.2) + (spectrum_similarity * 0.2)
        
        # Apply time decay - less likely to be self-speech as time passes
        time_since_speech = time.time() - self.audio_stats["last_spoken_time"]
        time_factor = max(0, 1.0 - (time_since_speech / 2.0))  # Faster decay over 2 seconds
        
        # Final probability score
        self_speech_probability = similarity_score * time_factor
        
        # Log detailed analysis if in debug mode
        if self.debug_mode:
            logger.debug(f"Audio self-speech analysis: " +
                        f"vol_sim={volume_similarity:.2f}, peak_sim={peak_similarity:.2f}, " +
                        f"spec_sim={spectrum_similarity:.2f}, time_factor={time_factor:.2f}, " +
                        f"probability={self_speech_probability:.2f}")
        
        # Higher threshold during and just after speaking - more permissive thresholds
        threshold = 0.55 if self._should_use_elevated_threshold() else 0.80
        
        return self_speech_probability > threshold
    
    def _processing_thread_func(self):
        """Background thread that continuously processes audio"""
        logger.info("Audio processing thread started")
        print("🎤 Microphone activated and listening")
        
        self.listen_count = 0
        last_stats_collect_time = 0
        
        # VAD (Voice Activity Detection) window
        vad_window = []
        vad_window_size = 20  # Number of frames to keep in the VAD window
        
        # Open a continuous microphone stream
        try:
            # Initialize the stream with more robust error handling
            try:
                logger.info("Opening microphone stream...")
                stream = self.p.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK,
                    input_device_index=self.mic_index
                )
                
                # Flush initial buffer to prevent stale data
                logger.info("Flushing initial microphone buffer...")
                for _ in range(5):  # Read a few chunks to clear any stale data
                    stream.read(self.CHUNK, exception_on_overflow=False)
                time.sleep(0.1)  # Short pause after flushing
                
                logger.info("Microphone stream opened successfully")
            except Exception as e:
                logger.error(f"Error opening microphone stream: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Buffer to collect audio when above threshold
            frames = []
            is_recording = False
            silent_frames = 0
            max_silent_frames = int(self.RATE / self.CHUNK * 1.5)  # 1.5 seconds of silence
            recording_start = 0
            max_record_time = self.RECORD_SECONDS
            last_speaking_state = False
            last_keypress_time = time.time()
            min_keypress_interval = 1.0  # Minimum time between keypresses
            
            while not self.should_stop:
                self.listen_count += 1
                
                if self.listen_count % 30 == 0:  # Throttle status messages
                    if not self.in_conversation:
                        print("👂 Waiting for wake word...")
                    else:
                        print("👂 Listening for commands...")
                
                # Always read from microphone
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                except Exception as read_error:
                    logger.error(f"Error reading from microphone: {read_error}")
                    # Try to recover by sleeping briefly
                    time.sleep(0.1)
                    continue
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Get current volume level
                volume = np.abs(audio_data).mean()
                self.last_volume = volume
                
                # Add to voice activity detection window
                vad_window.append(volume)
                if len(vad_window) > vad_window_size:
                    vad_window.pop(0)
                
                # Calculate VAD statistics
                if len(vad_window) >= 5:  # Need some minimum history
                    avg_volume = np.mean(vad_window)
                    std_volume = np.std(vad_window)
                    
                    # Check for feedback loop pattern (consistent volumes with low variation)
                    if self.is_assistant_speaking and std_volume < avg_volume * 0.15:
                        # Very consistent volume during speaking likely indicates feedback
                        if std_volume > 0 and avg_volume > self.energy_threshold * 0.8:
                            logger.warning(f"Possible feedback loop detected: consistent volume with low variation")
                            logger.warning(f"avg={avg_volume:.1f}, std={std_volume:.1f}, ratio={std_volume/avg_volume:.3f}")
                            # Skip this frame to avoid feedback
                            continue
                
                # Check if speaking state changed - this will help us reset recording when speaking starts
                if self.is_assistant_speaking != last_speaking_state:
                    if self.is_assistant_speaking:
                        logger.debug("Speaking state changed to speaking - resetting any ongoing recording")
                        # Reset recording if we just started speaking
                        if is_recording:
                            is_recording = False
                            frames = []
                            silent_frames = 0
                        # Clear VAD window when starting to speak
                        vad_window = []
                    last_speaking_state = self.is_assistant_speaking
                
                # Collect audio statistics when assistant is speaking
                current_time = time.time()
                if self.is_assistant_speaking and current_time - last_stats_collect_time > 0.2:  # Every 200ms
                    # Analyze audio characteristics
                    stats = self._analyze_audio_characteristics(audio_data)
                    self.audio_stats["assistant_speech"]["volume_avg"].append(stats["volume_avg"])
                    self.audio_stats["assistant_speech"]["volume_peak"].append(stats["volume_peak"])
                    if stats["spectrum_profile"]:
                        self.audio_stats["assistant_speech"]["spectrum_profile"].append(stats["spectrum_profile"])
                    last_stats_collect_time = current_time
                    
                    if self.debug_mode and self.listen_count % 20 == 0:
                        logger.debug(f"Collected assistant speech audio stats: vol_avg={stats['volume_avg']:.1f}, vol_peak={stats['volume_peak']:.1f}")
                
                # Debug volume occasionally
                if self.debug_mode and self.listen_count % 10 == 0:
                    current_threshold = self.energy_threshold
                    logger.debug(f"Current volume: {volume:.1f} (threshold: {current_threshold})")
                
                # Determine current threshold based on speaking state and delay
                # Use lower threshold when waiting for wake word to make it easier to activate
                if not self.in_conversation:
                    # When not in conversation, use even lower threshold for wake word detection
                    current_threshold = self.speaking_energy_threshold * 0.8 if self._should_use_elevated_threshold() else self.energy_threshold * 0.9
                else:
                    # Normal thresholds during conversation
                    current_threshold = self.speaking_energy_threshold if self._should_use_elevated_threshold() else self.energy_threshold
                
                # If speaking and we detect a sudden very loud sound, it might be the user interrupting
                if self.is_assistant_speaking and volume > self.speaking_energy_threshold * 2.0:
                    logger.info(f"Detected potential interruption (loud sound while speaking): volume={volume:.1f}")
                
                if volume > current_threshold:
                    # If we weren't recording before, start a new recording
                    if not is_recording:
                        # If speaking, apply additional validation
                        if self.is_assistant_speaking:
                            # When assistant is speaking, we want to be more sure it's not picking up itself
                            # Only start recording if the volume is significantly higher
                            if volume > current_threshold * 1.5:
                                # Additional VAD check - ensure the volume pattern indicates real speech
                                if len(vad_window) >= 5 and not self._is_likely_feedback(vad_window):
                                    is_recording = True
                                    frames = [data]  # Start with this chunk
                                    recording_start = time.time()
                                    logger.debug(f"Started recording while speaking (higher volume: {volume:.1f}, threshold: {current_threshold})")
                                else:
                                    logger.debug(f"Rejecting audio - likely feedback pattern detected")
                        else:
                            is_recording = True
                            frames = [data]  # Start with this chunk
                            recording_start = time.time()
                            logger.debug(f"Started recording (volume: {volume:.1f}, threshold: {current_threshold})")
                            
                        # Only show audio detection if not too frequent and we started recording
                        if is_recording and self.listen_count % 5 == 0:
                            print("🔊 Sound detected...")
                    else:
                        # Continue recording
                        frames.append(data)
                        silent_frames = 0
                else:
                    # Below threshold
                    if is_recording:
                        frames.append(data)  # Still append to get smooth endings
                        silent_frames += 1
                        
                        # Check if we've been silent long enough to stop recording
                        if silent_frames > max_silent_frames:
                            # If we were recording but never broke threshold by much, it might be noise
                            if len(frames) < 10:
                                logger.debug("Recording too short, likely noise - discarding")
                                is_recording = False
                                frames = []
                                silent_frames = 0
                                continue
                                
                            # We've detected enough silence - process this audio segment
                            audio_data = b''.join(frames)
                            
                            # Skip if it's likely to be assistant's own speech
                            should_process = True
                            
                            # For audio recorded during assistant speech, check if it's self-speech
                            # But be more cautious about filtering - we still want to process potential wake words
                            if self.is_assistant_speaking or time.time() < self.speaking_end_time:
                                # When we're speaking or just stopped speaking, be very cautious
                                # Check if this audio is likely self-speech by audio characteristics
                                if self._is_audio_self_speech(audio_data):
                                    logger.info("Detected likely self-speech by audio characteristics, skipping")
                                    should_process = False
                                    
                            if should_process:
                                # Save to temp file and process in separate thread to avoid blocking
                                temp_file = os.path.join(self.temp_dir, f"audio_{int(time.time())}.wav")
                                self._save_audio(audio_data, temp_file)
                                
                                # Add to buffer for processing
                                with self.speech_buffer_lock:
                                    self.speech_buffer.append(temp_file)
                                    logger.debug(f"Added audio to speech buffer (length: {len(self.speech_buffer)})")
                            
                            # Reset recording state
                            is_recording = False
                            frames = []
                            silent_frames = 0
                        
                        # Check if we've been recording too long
                        elif time.time() - recording_start > max_record_time:
                            logger.info("Maximum recording time reached")
                            # Save what we have so far
                            audio_data = b''.join(frames)
                            
                            # Skip if it's likely to be assistant's own speech
                            should_process = True
                            
                            # Be more cautious about filtering while speaking
                            if self.is_assistant_speaking or time.time() < self.speaking_end_time:
                                # Check if this audio is likely self-speech by audio characteristics
                                if self._is_audio_self_speech(audio_data):
                                    logger.info("Detected likely self-speech by audio characteristics, skipping")
                                    should_process = False
                                    
                            if should_process:
                                # Save to temp file and process in separate thread
                                temp_file = os.path.join(self.temp_dir, f"audio_{int(time.time())}.wav")
                                self._save_audio(audio_data, temp_file)
                                
                                # Add to buffer for processing
                                with self.speech_buffer_lock:
                                    self.speech_buffer.append(temp_file)
                                    logger.debug(f"Added audio to speech buffer (length: {len(self.speech_buffer)})")
                            
                            # Reset recording state
                            is_recording = False
                            frames = []
                            silent_frames = 0
                
                # Small sleep to prevent tight loop
                time.sleep(0.01)
                
            # Clean up
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Error in audio processing thread: {e}")
            logger.error(traceback.format_exc())
            
        logger.info("Audio processing thread stopped")
        print("🛑 Microphone listening stopped")
    
    def _is_likely_feedback(self, vad_window):
        """Analyze volume pattern to detect feedback loop patterns
        
        Feedback loops tend to have consistent volume patterns with low variation,
        while natural speech has more variation in volume.
        """
        if len(vad_window) < 5:
            return False
            
        # Calculate statistics on the volume window
        mean_vol = np.mean(vad_window)
        std_vol = np.std(vad_window)
        
        # Very consistent volume is suspicious especially if it's not very quiet
        if mean_vol > self.energy_threshold * 0.5 and std_vol < mean_vol * 0.2:
            logger.debug(f"Suspicious volume pattern: mean={mean_vol:.1f}, std={std_vol:.1f}, ratio={std_vol/mean_vol:.3f}")
            return True
            
        return False
    
    def _speech_processing_thread_func(self):
        """Separate thread that processes the speech buffer"""
        logger.info("Speech processing thread started")
        
        last_processed_time = 0
        min_processing_interval = 0.5  # Min seconds between processing to avoid feedback loops
        
        while not self.should_stop:
            # Check for audio files in the buffer
            with self.speech_buffer_lock:
                if self.speech_buffer:
                    # Get the next file
                    audio_file = self.speech_buffer.pop(0)
                else:
                    audio_file = None
            
            if audio_file:
                current_time = time.time()
                
                # Skip processing if we've processed audio too recently
                if current_time - last_processed_time < min_processing_interval:
                    logger.debug(f"Skipping processing - too soon after last process ({current_time - last_processed_time:.2f}s)")
                    time.sleep(0.1)
                    # Put back at the end of the queue if it's valid
                    if os.path.exists(audio_file):
                        with self.speech_buffer_lock:
                            self.speech_buffer.append(audio_file)
                    continue
                
                # Apply additional delays if needed
                if self.is_assistant_speaking:
                    # While speaking, add more delay to avoid feedback
                    delay_needed = 0.6
                    logger.debug(f"Adding delay of {delay_needed}s because assistant is speaking")
                    time.sleep(delay_needed)
                elif current_time < self.speaking_end_time:
                    # Just stopped speaking - add a shorter delay
                    time_since_speaking = current_time - self.speaking_end_time
                    if time_since_speaking < 0.5:  # Within 0.5s of speaking end
                        delay_needed = 0.3
                        logger.debug(f"Adding delay of {delay_needed}s after speaking")
                        time.sleep(delay_needed)
                
                # If we should stop, don't process this file
                if self.should_stop:
                    try:
                        # Try to remove the audio file
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                    except Exception as e:
                        logger.error(f"Error removing audio file: {e}")
                    break
                
                # Process the audio file
                recognized_text = self._recognize_speech(audio_file)
                
                # Update the last processed time
                last_processed_time = time.time()
                
                # Try to remove the audio file after processing
                try:
                    os.remove(audio_file)
                except Exception as e:
                    logger.error(f"Error removing audio file: {e}")
                
                # Skip if no text was recognized
                if not recognized_text:
                    continue
                    
                # Apply text-based filtration of self-speech
                if self._is_likely_self_speech(recognized_text):
                    if not self._check_wake_word(recognized_text):  # Only skip if it's not a wake word
                        logger.info(f"Filtered out likely self-speech: '{recognized_text}'")
                        continue
                
                # Check if this audio has the acoustic characteristics of self-speech
                # This is an additional check that happens after text recognition
                if self.is_assistant_speaking or time.time() < self.speaking_end_time + 1.0:
                    # Only apply this check if we're speaking or just finished speaking
                    if self._check_self_speech_by_timing(recognized_text):
                        logger.warning(f"Detected probable self-speech based on timing: '{recognized_text}'")
                        # Don't skip wake words even if they seem like self-speech
                        if not self._check_wake_word(recognized_text):
                            continue
                
                # Process the text
                self._process_speech(recognized_text)
            
            # Small delay to prevent tight loops
            time.sleep(0.1)
        
        logger.info("Speech processing thread stopped")
    
    def _check_self_speech_by_timing(self, text):
        """Check if recognized text is likely to be self-speech based on timing and content patterns
        
        This uses a combination of timing (how soon after speaking), the text pattern,
        and common indicators of self-speech that might not be in the text filter.
        """
        # If not speaking and not just finished speaking, it's unlikely to be self-speech
        if not self.is_assistant_speaking and time.time() > self.speaking_end_time + 2.0:
            return False
            
        # Check for very short texts that are common responses
        text_lower = text.lower().strip()
        common_responses = ["yes", "okay", "sure", "no", "thanks", "thank you", "hello", "hi", "hey"]
        
        # If it's a very common short response AND we just spoke, it's highly suspicious
        if text_lower in common_responses and time.time() < self.speaking_end_time + 1.0:
            return True
            
        # Check if the text contains any phrases from our self-speech log
        # This is for cases where the exact text isn't in the filter
        # but it's very similar to something we just said
        time_since_speaking = time.time() - self.speaking_end_time
        
        if time_since_speaking < 1.5:  # Only check this for 1.5 seconds after speaking
            # Check against recent recognitions instead of self_speech_filter
            for phrase in self.recent_recognitions:
                # For short texts, require a stronger match
                if len(text_lower) < 10:
                    if text_lower in phrase.lower() or phrase.lower() in text_lower:
                        return True
                # For longer texts, check for substantial overlap
                else:
                    # Calculate the length of the longest common substring
                    # as a rough measure of similarity
                    common_words = set(text_lower.split()).intersection(set(phrase.lower().split()))
                    if common_words and len(common_words) >= 2:
                        # If we share 2+ words AND we just spoke, it's suspicious
                        return True
        
        return False
    
    def set_assistant_speaking(self, is_speaking):
        """Set whether the assistant is currently speaking to adjust thresholds"""
        self.is_assistant_speaking = is_speaking
        if is_speaking:
            # Set the higher threshold when speaking
            logger.info(f"Assistant is speaking - increasing energy threshold to {self.speaking_energy_threshold}")
            # Reset assistant speech audio characteristics collection
            if len(self.audio_stats["assistant_speech"]["volume_avg"]) > 10:
                # Keep only the most recent 10 samples
                self.audio_stats["assistant_speech"]["volume_avg"] = self.audio_stats["assistant_speech"]["volume_avg"][-10:]
                self.audio_stats["assistant_speech"]["volume_peak"] = self.audio_stats["assistant_speech"]["volume_peak"][-10:]
                self.audio_stats["assistant_speech"]["spectrum_profile"] = self.audio_stats["assistant_speech"]["spectrum_profile"][-10:]
            
            # Reset feedback detection when we start speaking
            self.consecutive_recognitions = 0
        else:
            # Set time to keep higher threshold for a short delay after speaking
            self.speaking_end_time = time.time() + self.post_speaking_delay
            self.audio_stats["last_spoken_time"] = time.time()
            logger.info(f"Assistant stopped speaking - will restore threshold after {self.post_speaking_delay} seconds")
    
    def _should_use_elevated_threshold(self):
        """Determine whether to use the elevated threshold based on speaking status and timing"""
        # Always use elevated threshold if actively speaking
        if self.is_assistant_speaking:
            return True
            
        # Use elevated threshold during the post-speaking delay period
        if time.time() < self.speaking_end_time:
            return True
            
        # Otherwise use normal threshold
        return False
    
    def _is_likely_self_speech(self, text):
        """Check if the recognized text is likely to be self-speech (the assistant hearing itself)"""
        # Skip empty text
        if not text:
            return False
            
        text_lower = text.lower().strip()
        
        # Check for consecutive recognitions in a short time window
        current_time = time.time()
        time_since_last = current_time - self.last_recognition_time
        
        if time_since_last < self.feedback_detection_window:
            self.consecutive_recognitions += 1
            if self.consecutive_recognitions > self.max_consecutive_recognitions:
                logger.info(f"Detected {self.consecutive_recognitions} consecutive recognitions in a short time frame - possible feedback loop")
                self._reset_after_feedback_detection()
                return True
        else:
            # Reset counter if enough time has passed
            self.consecutive_recognitions = 1
            
        self.last_recognition_time = current_time
        
        # Don't filter out commands starting with "execute" or search queries
        # as these are likely legitimate user commands
        if text_lower.startswith('execute ') or text_lower.startswith('search '):
            return False
            
        # Check for significant overlap with recent recognitions (the assistant's speech)
        # but be less aggressive with the filtering
        for recent_text in self.recent_recognitions:
            if recent_text and len(recent_text) > 5:  # Only check substantial texts
                # Calculate similarity score using character overlap
                similarity = self._calculate_text_similarity(text_lower, recent_text.lower())
                
                # If very high similarity (over 80%), it's likely self-speech
                if similarity > 0.8:
                    logger.info(f"Detected significant self-speech overlap: '{text_lower}' vs '{recent_text.lower()}'")
                    logger.info(f"Filtered out likely self-speech: '{text}'")
                    return True
                    
        # Check for wake word / interrupt word repetition soon after speaking
        if self.wake_word and self.wake_word in text_lower:
            # If we just heard the wake word, and we immediately hear it again, it might be feedback
            time_since_speaking = current_time - self.speaking_end_time
            if time_since_speaking < 1.5:  # 1.5 seconds
                logger.debug("Wake word detected very soon after speaking - possible self-speech")
                return True
                
        # Not detected as self-speech
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
            
            # Start audio processing thread
            logger.info("Starting audio processing thread...")
            self.processing_thread = threading.Thread(
                target=self._processing_thread_func,
                daemon=True
            )
            self.processing_thread.start()
            
            # Start speech processing thread
            logger.info("Starting speech processing thread...")
            self.speech_processing_thread = threading.Thread(
                target=self._speech_processing_thread_func,
                daemon=True
            )
            self.speech_processing_thread.start()
            
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
        
        # Signal threads to stop
        self.should_stop = True
        self.running = False
        logger.info("Set should_stop flag to True")
        
        # Wait for threads to terminate
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Waiting for processing thread to stop...")
            self.processing_thread.join(timeout=2)
            
            if self.processing_thread.is_alive():
                logger.warning("Processing thread did not stop within timeout")
        
        if self.speech_processing_thread and self.speech_processing_thread.is_alive():
            logger.info("Waiting for speech processing thread to stop...")
            self.speech_processing_thread.join(timeout=2)
            
            if self.speech_processing_thread.is_alive():
                logger.warning("Speech processing thread did not stop within timeout")
        
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
        """Process recognized speech and handle wake word and commands"""
        if not text:
            return
        
        text = text.strip()
        
        # Add to recognition counters for debugging
        self.recognition_count += 1
        
        # Log the recognized text
        logger.info(f"Recognized: '{text}'")
        print(f"✓ Recognized: \"{text}\"")
        
        # If we have an active audio source, make a note of this speech
        self.last_audio_timestamp = time.time()

        # Check for known problematic patterns and feedback
        if self._is_likely_self_speech(text):
            return  # Skip processing this speech
            
        # Check for interrupt word
        if self.interrupt_word and self.interrupt_word.lower() in text.lower():
            # Call the speech_detected callback with interrupt event
            if self.on_speech_detected:
                self.on_speech_detected("interrupt", text)
            return
            
        # Special handling for "execute" commands - prioritize these
        if text.lower().startswith("execute "):
            # If we're in conversation mode, process this as a command immediately
            if self.in_conversation:
                logger.info(f"Processing direct execute command in conversation mode: '{text}'")
                if self.on_speech_recognized:
                    self.on_speech_recognized(text)
                return
                
        # If we're already in conversation mode, any speech is a command
        if self.in_conversation:
            logger.info(f"Processing speech in conversation mode: '{text}'")
            # Call the speech_recognized callback
            if self.on_speech_recognized:
                logger.info("Calling speech recognized callback")
                self.on_speech_recognized(text)
            return

        # Not in conversation mode yet - check for wake word
        self._check_wake_word(text)
    
    def _check_wake_word(self, text):
        """Check if text contains the wake word and respond if found"""
        # Skip if in conversation already
        if self.in_conversation:
            return False
            
        if not self.wake_word or not text:
            return False
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        wake_word_lower = self.wake_word.lower()
        
        # Check for exact matches and variations
        found = False
        
        # The actual wake word as a standalone or at the beginning
        if (text_lower == wake_word_lower or 
            text_lower.startswith(wake_word_lower + " ") or
            " " + wake_word_lower + " " in text_lower or
            text_lower.endswith(" " + wake_word_lower)):
            found = True
            
        # Common variations with "hey", "hi", "hello"
        variations = [
            f"hey {wake_word_lower}",
            f"hi {wake_word_lower}",
            f"hello {wake_word_lower}",
            f"ok {wake_word_lower}"
        ]
        
        if not found:
            for variation in variations:
                if (text_lower == variation or
                    text_lower.startswith(variation + " ") or
                    " " + variation + " " in text_lower):
                    found = True
                    break
                    
        if found:
            # Wake word detected - but DON'T set conversation mode here
            # Let the speech_detected callback handle that
            logger.info(f"Wake word detected in: '{text}'")
            
            # Call the callback without changing state here
            if self.on_speech_detected:
                logger.info("Calling wake word callback from _check_wake_word")
                self.on_speech_detected("wake_word", text)
                
            return True
        else:
            logger.debug("Wake word not detected")
            return False

    def get_debug_info(self):
        """Get debug information about the audio manager"""
        return {
            "running": self.running,
            "in_conversation": self.in_conversation,
            "last_audio_timestamp": self.last_audio_timestamp,
            "recognition_count": self.recognition_count,
            "energy_threshold": self.energy_threshold,
            "recognition_attempts": self.recognition_attempts,
            "recognition_successes": self.recognition_successes,
            "buffer_size": len(self.speech_buffer) if hasattr(self, 'speech_buffer') else 0
        }
    
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
    
    # Backward compatibility method
    def pause_listening(self, pause=True):
        """
        DEPRECATED - Backward compatibility method
        Use set_assistant_speaking instead for adjusting thresholds
        """
        logger.warning("pause_listening is deprecated, use set_assistant_speaking instead")
        # No-op - we don't pause recording anymore, just adjust thresholds
    
    def _record_audio(self):
        """
        DEPRECATED - Kept for backwards compatibility
        Use the continuous recording in _processing_thread_func instead
        """
        logger.warning("_record_audio called directly - this method is deprecated")
        return None

    def _reset_after_feedback_detection(self):
        """Reset counters and adjust thresholds after feedback detection"""
        logger.info("Resetting thresholds after feedback detection")
        self.consecutive_recognitions = 0
        self.last_recognition_time = 0
        # Increase energy threshold temporarily
        self.energy_threshold += 100
        logger.info(f"Temporarily increasing energy threshold to {self.energy_threshold} due to feedback detection")
        print("✅ Listening resumed with normal sensitivity")
        
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two text strings
        
        Returns a value between 0.0 (completely different) and 1.0 (identical)
        """
        # For very short texts, simple character comparison
        if len(text1) < 3 or len(text2) < 3:
            return 1.0 if text1 == text2 else 0.0
            
        # For longer texts, use character set overlap ratio
        set1 = set(text1)
        set2 = set(text2)
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        set_similarity = intersection / union if union > 0 else 0
        
        # Calculate word overlap for better accuracy
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        word_intersection = len(words1.intersection(words2))
        word_union = len(words1.union(words2))
        word_similarity = word_intersection / word_union if word_union > 0 else 0
        
        # Calculate substring similarity
        substring_match = 0.0
        if text1 in text2 or text2 in text1:
            # Calculate how much of the shorter string is contained in the longer one
            min_len = min(len(text1), len(text2))
            max_len = max(len(text1), len(text2))
            substring_match = min_len / max_len if max_len > 0 else 0.0
            
        # Weighted combination of similarity metrics
        similarity = (0.3 * set_similarity) + (0.5 * word_similarity) + (0.2 * substring_match)
        
        return min(1.0, similarity)  # Cap at 1.0

class StreamingMaxwell:
    def __init__(self, config):
        self.config = config
        self.running = True
        self.in_conversation = False
        self.speaking = False
        self.cleaned_up = False
        self.should_stop = False
        
        # Debug the config object
        logger.info(f"Config object type: {type(config)}")
        logger.info(f"Config object dir: {dir(config)}")
        logger.info(f"Config has use_mcp: {hasattr(config, 'use_mcp')}")
        if hasattr(config, 'use_mcp'):
            logger.info(f"Config use_mcp value: {config.use_mcp}")
        logger.info(f"Config.get('use_mcp'): {config.get('use_mcp', 'NOT_FOUND')}")
        
        # Command mode flag - initially False
        self.command_mode = False
        
        # Initialize audio manager and mcp provider
        self.audio_manager = None
        self.mcp_tool_provider = None
        
        # Initialize LLM Provider directly
        self.llm_provider = None
        self._initialize_llm_provider()
        
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
            
            # Add callbacks for speech state
            if hasattr(self.tts, '_is_speaking'):
                # Add speech start/stop callbacks if supported
                logger.info("Adding speech start/stop callbacks")
                
                # These will be set after audio_manager is created
                self.tts.on_speaking_started = None
                self.tts.on_speaking_stopped = None
        
        # Initialize keyboard handler for space bar interruption
        logger.info("Initializing keyboard handler...")
        self.keyboard_handler = KeyboardHandler(self.tts)
        
        # MCP Tool Provider setup - simplified approach
        self.enable_mcp = getattr(config, 'use_mcp', True)
        print("config was\n============================\n", config, "\n============================\n")
        logging.warning(f"MCP Tool Provider enabled: {self.enable_mcp}")
        if self.enable_mcp and HAS_MCP_TOOLS:
            logger.info("Initializing MCP Tool Provider...")
            self.mcp_tool_provider = MCPToolProvider()
        else:
            if self.enable_mcp:
                logger.warning("MCP tools requested but not available")
            self.mcp_tool_provider = None
        
        # Initialize Command Executor (will be set after initialization)
        logger.info("Initializing command executor...")
        from commands import CommandExecutor
        # logger.error(f"Self.mcp_tool_provider: {self.mcp_tool_provider}")
        self.mcp_tool_provider = MCPToolProvider()
        self.command_executor = CommandExecutor(mcp_tool_provider=self.mcp_tool_provider, config=self.config)
        logger.info("Command executor initialized")
    
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"StreamingMaxwell initialized with wake word: '{self.wake_word}'")
    
    def _initialize_llm_provider(self):
        """Initialize the LLM provider based on the configuration"""
        from llm_provider import create_llm_provider
        
        llm_provider_type = self.config.get('llm_provider', 'none').lower()
        
        try:
            if llm_provider_type != 'none':
                logger.info(f"Initializing LLM provider: {llm_provider_type}")
                self.llm_provider = create_llm_provider(self.config)
                logger.info(f"LLM provider initialized with model: {self.llm_provider.get_model_name()}")
            else:
                logger.info("No LLM provider configured")
                self.llm_provider = None
        except Exception as e:
            logger.error(f"Error initializing LLM provider: {e}")
            logger.error(traceback.format_exc())
            print(f"⚠️ Warning: Failed to initialize LLM provider: {e}")
            self.llm_provider = None
    
    def _on_speech_detected(self, event_type, text):
        """Callback for speech detection events"""
        if event_type == "wake_word":
            logger.info(f"Wake word detected: {text}")
            self._play_listening_sound()
            
            # Set conversation mode first, before speaking
            self.in_conversation = True
            self.audio_manager.set_conversation_mode(True)
            
            # Speak only once
            self.speak("Yes?")
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
    
    def _on_speaking_started(self):
        """Callback for when the assistant starts speaking"""
        logger.info("Assistant started speaking - adjusting energy threshold")
        self.speaking = True
        # Adjust the energy threshold to avoid self-triggering
        # This doesn't stop the mic from listening, just raises the threshold
        if hasattr(self.audio_manager, 'set_assistant_speaking'):
            self.audio_manager.set_assistant_speaking(True)
    
    def _on_speaking_stopped(self):
        """Callback for when the assistant stops speaking"""
        try:
            logger.info("Assistant stopped speaking - restoring energy threshold")
            self.speaking = False
            # Restore the energy threshold to normal level
            # The mic has been listening the whole time, just with a higher threshold
            if self.running and hasattr(self.audio_manager, 'set_assistant_speaking') and self.audio_manager:
                try:
                    self.audio_manager.set_assistant_speaking(False)
                except Exception as e:
                    logger.error(f"Error restoring energy threshold: {e}")
                    # Don't re-raise, continue execution
        except Exception as e:
            logger.error(f"Uncaught exception in _on_speaking_stopped: {e}")
            logger.error(traceback.format_exc())
            # Don't propagate any exceptions to prevent program termination
    
    def signal_handler(self, sig, frame):
        """Handle interrupt signals"""
        try:
            signal_name = {
                signal.SIGINT: "SIGINT (Ctrl+C)",
                signal.SIGTERM: "SIGTERM",
                signal.SIGABRT: "SIGABRT",
                signal.SIGBREAK: "SIGBREAK (Windows Ctrl+Break)" if hasattr(signal, 'SIGBREAK') else "UNKNOWN",
            }.get(sig, f"Signal {sig}")
            
            logger.info(f"=== SIGNAL DEBUG: Received {signal_name}, initiating graceful shutdown ===")
            print(f"\n🛑 Received {signal_name}, shutting down gracefully...")
            
            # CRITICAL: First ensure TTS is stopped to prevent any deadlocks
            try:
                if hasattr(self, 'tts') and self.tts:
                    logger.info("=== SIGNAL DEBUG: Emergency stopping TTS ===")
                    if hasattr(self.tts, 'stop'):
                        self.tts.stop()
                    logger.info("=== SIGNAL DEBUG: TTS stopped ===")
            except Exception as tts_error:
                logger.error(f"=== SIGNAL DEBUG: Error stopping TTS: {tts_error} ===")
            
            # Set running flag to false to initiate graceful shutdown
            self.running = False
            
            # SIGTERM needs more immediate action
            if sig == signal.SIGTERM:
                logger.info("=== SIGNAL DEBUG: SIGTERM received, cleaning up and exiting ===")
                # SIGTERM should be handled more carefully
                try:
                    self.cleanup()
                    logger.info("=== SIGNAL DEBUG: Cleanup completed, exiting ===")
                except Exception as cleanup_error:
                    logger.error(f"=== SIGNAL DEBUG: Error during cleanup: {cleanup_error} ===")
                
                # Don't exit directly, set should_stop flag
                self.should_stop = True
                return
            
            # For all other signals, set the stop flag so the main loop handles shutdown
            # This prevents abrupt termination that could leave resources in a bad state
            logger.info(f"=== SIGNAL DEBUG: {signal_name} received, setting stop flag but continuing execution ===")
            self.should_stop = True
            
            # The main loop will check should_stop and exit cleanly
        except Exception as e:
            logger.error(f"=== SIGNAL DEBUG: Error in signal handler: {e} ===")
            logger.error(traceback.format_exc())
            # Don't exit here, let the main loop handle it
    
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
                try:
                    logger.info("Stopping keyboard handler...")
                    self.keyboard_handler.stop()
                    
                    # Verify the thread has stopped
                    if hasattr(self.keyboard_handler, 'thread') and self.keyboard_handler.thread:
                        if self.keyboard_handler.thread.is_alive():
                            logger.warning("Keyboard handler thread is still alive, waiting...")
                            try:
                                self.keyboard_handler.thread.join(timeout=2.0)
                            except Exception as join_error:
                                logger.error(f"Error joining keyboard handler thread: {join_error}")
                except Exception as kh_error:
                    logger.error(f"Error stopping keyboard handler: {kh_error}")
                    # Continue with cleanup even if this fails
            
            # Clean up TTS
            if hasattr(self, 'tts'):
                try:
                    logger.info("Cleaning up TTS...")
                    self.tts.cleanup()
                except Exception as tts_error:
                    logger.error(f"Error cleaning up TTS: {tts_error}")
                    # Continue with cleanup even if this fails
                
            # Clean up AudioManager
            if hasattr(self, 'audio_manager') and self.audio_manager:
                try:
                    logger.info("Stopping audio manager...")
                    self.audio_manager.stop()
                except Exception as am_error:
                    logger.error(f"Error stopping audio manager: {am_error}")
                    # Continue with cleanup even if this fails
            
            # Clean up MCP Tools if enabled
            if hasattr(self, 'mcp_tool_provider') and self.mcp_tool_provider:
                try:
                    logger.info("Stopping MCP Tool Provider...")
                    self.mcp_tool_provider.stop_server()
                except Exception as mcp_error:
                    logger.error(f"Error stopping MCP tool provider: {mcp_error}")
                    # Continue with cleanup even if this fails
                
            # Clean up LLM provider if it has a cleanup method
            if hasattr(self, 'llm_provider') and self.llm_provider:
                try:
                    logger.info("Cleaning up LLM provider...")
                    if hasattr(self.llm_provider, 'cleanup'):
                        self.llm_provider.cleanup()
                except Exception as llm_error:
                    logger.error(f"Error cleaning up LLM provider: {llm_error}")
                    # Continue with cleanup even if this fails
                
            # Mark as cleaned up
            self.cleaned_up = True
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            logger.error(traceback.format_exc())
            # Set cleaned_up flag even if there were errors
            self.cleaned_up = True
            logger.info("Cleanup completed with errors")
    
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
            try:
                self._play_sound(frequency=900, duration=70)  # Medium-pitched beep
            except Exception as beep_error:
                logger.error(f"Error playing beep sound: {beep_error}")
                # Continue even if beep fails
                
            # Tell audio manager that we're speaking to adjust threshold
            if hasattr(self.audio_manager, 'set_assistant_speaking') and self.audio_manager:
                try:
                    self.audio_manager.set_assistant_speaking(True)
                    
                    # Add only distinctive phrases to the self-speech filter
                    # Don't add greetings, wake word responses, or other common phrases
                    # that might filter out legitimate user commands
                    if hasattr(self.audio_manager, 'recent_recognitions'):
                        # Skip adding very short responses or common responses to the filter
                        common_phrases = ["yes", "hello", "hi", "hey", "okay", "sure", 
                                        "yes?", "I'm listening", "I'm here", "go ahead"]
                        
                        if len(text.strip()) > 8 and not any(text.lower().startswith(phrase.lower()) for phrase in common_phrases):
                            # Split long responses into shorter phrases
                            sentences = re.split(r'[.!?]\s+', text)
                            for sentence in sentences:
                                # Only add substantial sentences, not simple acknowledgments
                                if sentence and len(sentence.strip()) > 10:
                                    self.audio_manager.recent_recognitions.append(sentence.strip())
                                    logger.debug(f"Added to speech filter: '{sentence.strip()}'")
                except Exception as am_error:
                    logger.error(f"Error setting assistant speaking state: {am_error}")
                    # Continue even if setting speaking state fails
            
            # Add a small delay to ensure thresholds are adjusted
            try:
                time.sleep(0.2)
            except Exception:
                # Ignore interrupts during the delay
                pass
                
            # Use direct audio playback - will be stopped by space bar handler
            try:
                # Check if self.tts is valid before calling speak
                if hasattr(self, 'tts') and self.tts:
                    self.tts.speak(text)
                    
                    # Wait for speech to complete
                    while hasattr(self, 'tts') and self.tts and self.tts.is_speaking():
                        try:
                            time.sleep(0.1)
                        except KeyboardInterrupt:
                            logger.warning("KeyboardInterrupt during speech wait - continuing")
                            break
                        except Exception as wait_error:
                            logger.error(f"Error during speech wait: {wait_error}")
                            break
                else:
                    logger.error("TTS not available for speech")
            except KeyboardInterrupt as k:
                # Handle Ctrl+C directly during speech
                print("\n🛑 Speech interrupted by KeyboardInterrupt (Ctrl+C)")
                try:
                    if hasattr(self, 'tts') and self.tts:
                        self.tts.stop()
                except Exception as stop_error:
                    logger.error(f"Error stopping TTS during KeyboardInterrupt: {stop_error}")
                
                # Don't re-raise the KeyboardInterrupt as it could terminate the program
                logger.warning("Captured KeyboardInterrupt during speech - continuing execution")
            except Exception as speak_error:
                logger.error(f"Error during TTS speak: {speak_error}")
                logger.error(traceback.format_exc())
                # Continue execution despite speech errors
            
            # Play a beep when Maxwell finishes speaking normally (if not interrupted)
            try:
                if hasattr(self, 'tts') and self.tts and not self.tts.is_speaking():
                    self._play_sound(frequency=850, duration=70)  # Medium-pitched beep
            except Exception as beep_error:
                logger.error(f"Error playing end beep: {beep_error}")
                # Continue even if end beep fails
            
            # Add a shorter pause after speaking to avoid cutting off
            try:
                time.sleep(0.3)
            except Exception:
                # Ignore interrupts during the delay
                pass
            
        except KeyboardInterrupt:
            # Catch and handle Ctrl+C again as a safety net
            print("\n🛑 Speech interrupted by user")
            logger.warning("Handling KeyboardInterrupt at outer level")
            try:
                if hasattr(self, 'tts') and self.tts:
                    self.tts.stop()
            except Exception as e:
                logger.error(f"Error stopping TTS: {e}")
            
            # Play a beep when speech is interrupted
            try:
                self._play_sound(frequency=600, duration=120)  # Lower-pitched longer beep for interruption
            except Exception as beep_error:
                logger.error(f"Error playing interrupt beep: {beep_error}")
        except Exception as e:
            logger.error(f"Error in speak: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Restore normal energy threshold
            try:
                if hasattr(self.audio_manager, 'set_assistant_speaking') and self.audio_manager:
                    self.audio_manager.set_assistant_speaking(False)
            except Exception as threshold_error:
                logger.error(f"Error restoring energy threshold: {threshold_error}")
                
            # Clear speaking flag
            self.speaking = False
            
            # Log that speaking has finished
            logger.debug("Speaking completed, energy threshold restored")
    
    def handle_query(self, query):
        """Handle a user query using MCP tools when possible"""
        if not query:
            return
            
        # Log the query
        query_lower = query.lower().strip()
        logger.info(f"Handling query: '{query}'")
        
        # Command mode check
        if self.command_mode:
            if not query_lower.startswith("execute "):
                logger.info(f"Command mode - adding execute prefix to: '{query}'")
                query = f"execute {query}"
                query_lower = query.lower()  # Update the lowercase version
                
        # If we have a command executor, try to use it first
        if query_lower.startswith("execute "):
            # Extract the actual command part
            command_text = query[8:].strip()
            logger.info(f"Execute command detected: '{command_text}'")
            
            # Directly use execute_command for better handling
            if self.command_executor:
                logger.info(f"Using direct command execution for: '{command_text}'")
                print(f"⚡ Executing command: '{command_text}'")
                response = self.command_executor.execute_command(command_text)
                if response:
                    self.speak(response)
                return
                
            # Special case for "execute search" pattern if command executor is not available
            if command_text.lower().startswith("search ") and self.enable_mcp and self.mcp_tool_provider:
                search_term = command_text[7:].strip()
                logger.info(f"Execute search command detected: '{search_term}'")
                print(f"🔍 Searching for: '{search_term}'")
                response = self.mcp_tool_provider.execute_tool("search_web", query=search_term)
                if response:
                    self.speak(response)
                return
        
        # MCP tools integration - check if tools are available
        if self.enable_mcp and self.mcp_tool_provider:
            # Start the tool provider if not already started
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
            
            # Handle search commands directly - prioritize these for better user experience
            if query_lower.startswith("search "):
                search_term = query[7:].strip()
                logger.info(f"Search command detected - search term: '{search_term}'")
                print(f"🔍 Processing search for: '{search_term}'")
                response = self.mcp_tool_provider.execute_tool("search_web", query=search_term)
                if response:
                    self.speak(response)
                return
            
            # Handle other search patterns
            search_keywords = ["search for", "look up", "find", "find information about"]
            is_search_command = False
            search_term = ""
            
            for keyword in search_keywords:
                if query_lower.startswith(keyword + " ") or keyword + " " in query_lower:
                    is_search_command = True
                    if keyword + " " in query_lower:
                        search_term = query_lower.split(keyword + " ", 1)[1].strip()
                    else:
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
    
        # If we get here and have a command executor, use it
        if self.command_executor:
            try:
                logger.info("Using command executor for query")
                response = self.command_executor.execute(query)
                if response:
                    self.speak(response)
                    return
            except Exception as e:
                logger.error(f"Error executing command: {e}")
                logger.error(traceback.format_exc())
                
        # If command executor didn't handle it, try direct LLM provider
        if hasattr(self, 'llm_provider') and self.llm_provider:
            try:
                # Check if LLM provider is available
                if self.llm_provider.check_connection():
                    logger.info(f"Querying LLM provider directly: {query}")
                    print("💭 Thinking...")
                    
                    # Format the query into messages
                    messages = self.llm_provider.format_user_query(query)
                    
                    # Send to LLM provider and get response
                    response = self.llm_provider.chat(messages)
                    
                    if response:
                        logger.info("Got response from LLM provider")
                        self.speak(response)
                        return
                    else:
                        logger.warning("Empty response from LLM provider")
                else:
                    logger.warning("LLM provider is not available")
            except Exception as e:
                logger.error(f"Error querying LLM provider: {e}")
                logger.error(traceback.format_exc())
                self.speak(f"I'm sorry, I encountered an error processing your request: {str(e)}")
                return
        else:
            # No command executor or LLM provider available
            logger.error("No command executor or LLM provider available to handle the query")
            self.speak("I'm sorry, I'm not able to process that request right now.")
    
    def run(self):
        """Main loop for the assistant"""
        initial_greeting_complete = False
        self.space_key_hit = False
        
        logger.info("StreamingMaxwell is running")
        print("\n🤖 Maxwell Assistant is starting up...")
        
        try:
            # Start the keyboard handler with special space key callback
            try:
                # Add a custom space key callback to handle space during startup
                self.space_key_callback = self._handle_space_key_during_startup
                self.keyboard_handler.set_space_key_callback(self.space_key_callback)
                
                # Start the keyboard handler
                self.keyboard_handler.start(self.tts)
                logger.info("Keyboard handler started - SPACE key will interrupt speech")
            except Exception as kh_error:
                logger.error(f"Failed to start keyboard handler: {kh_error}")
                logger.error(traceback.format_exc())
                print("⚠️ Warning: Keyboard interruption may not work properly")
            
            # Start MCP tool provider if enabled
            if self.enable_mcp and self.mcp_tool_provider:
                try:
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
                except Exception as mcp_error:
                    logger.error(f"Failed to start MCP tool provider: {mcp_error}")
                    logger.error(traceback.format_exc())
                    print("⚠️ Warning: MCP tools may not be available")
            
            # Print LLM provider information if available
            llm_provider_type = self.config.get('llm_provider', 'none').lower()
            if hasattr(self, 'llm_provider') and self.llm_provider:
                # Print LLM provider information based on the actual provider instance
                model_name = self.llm_provider.get_model_name()
                if llm_provider_type == 'openai':
                    base_url = getattr(self.config, 'openai_base_url', 'unknown')
                    print(f"🧠 Using OpenAI API ({model_name})")
                    print(f"   API URL: {base_url}")
                elif llm_provider_type == 'ollama':
                    base_url = getattr(self.config, 'ollama_base_url', 'unknown')
                    print(f"🧠 Using Ollama ({model_name})")
                    logger.info(f"Using Ollama with model {model_name} at {base_url}")
                else:
                    print(f"🧠 Using {llm_provider_type} provider ({model_name})")
            else:
                print("🧠 No LLM provider configured, running in command-only mode")
            
            # Ensure the wake word is set
            if not self.wake_word and hasattr(self.config, 'wake_word'):
                self.wake_word = self.config.wake_word
                
            if not self.wake_word:
                self.wake_word = "maxwell"  # Default wake word
                
            # Initialize the audio manager if not already done
            if not self.audio_manager:
                try:
                    mic_index = getattr(self.config, 'mic_index', None)
                    energy_threshold = getattr(self.config, 'energy_threshold', 300)
                    
                    self.audio_manager = StreamingAudioManager(
                        mic_index=mic_index,
                        energy_threshold=energy_threshold
                    )
                    
                    # Set callbacks
                    self.audio_manager.on_speech_detected = self._on_speech_detected
                    self.audio_manager.on_speech_recognized = self._on_speech_recognized
                    
                    # Connect the speech callbacks if available
                    if hasattr(self.tts, 'on_speaking_started'):
                        self.tts.on_speaking_started = self._on_speaking_started
                        self.tts.on_speaking_stopped = self._on_speaking_stopped
                        logger.info("Connected speech state callbacks")
                except Exception as am_error:
                    logger.error(f"Failed to initialize audio manager: {am_error}")
                    logger.error(traceback.format_exc())
                    print("❌ Error: Failed to initialize audio manager")
                    return False
                
            # Start the audio manager
            try:
                print(f"🎤 Listening for wake word: '{self.wake_word}'")
                self.audio_manager.start(wake_word=self.wake_word, interrupt_word=self.interrupt_word)
            except Exception as start_error:
                logger.error(f"Failed to start audio manager: {start_error}")
                logger.error(traceback.format_exc())
                print("❌ Error: Failed to start audio manager")
                return False
            
            # Announce startup - wrap this in extra protection since it's where space bar is often pressed
            print("\n🚀 Maxwell is ready!")
            try:
                # Special handling for initial greeting
                self.speaking = True
                # Signal that we're doing the initial greeting
                logger.info("Starting initial greeting - applying special protection")
                
                # Directly call TTS to avoid complex speak method during initial greeting
                if hasattr(self, 'tts') and self.tts:
                    # Print message but use a simpler speak routine
                    print(f"🔊 Maxwell: \"Hello, Maxwell here. Say '{self.wake_word}' to get my attention.\"")
                    print("(Press SPACE to interrupt speech)")
                    
                    try:
                        # Directly generate audio without the full speak method
                        logger.info(f"Speaking: Hello, Maxwell here. Say '{self.wake_word}' to get my attention.")
                        self.tts.speak(f"Hello, Maxwell here. Say '{self.wake_word}' to get my attention.")
                        
                        # Wait for speech to complete or be interrupted
                        try:
                            while self.tts.is_speaking():
                                time.sleep(0.1)
                        except Exception as wait_error:
                            logger.warning(f"Initial greeting was interrupted or errored: {wait_error}")
                    except Exception as speak_error:
                        logger.error(f"Error during initial greeting: {speak_error}")
                        logger.error(traceback.format_exc())
                    finally:
                        # Ensure speaking flag is cleared
                        self.speaking = False
                        
                        # Ensure energy threshold is restored
                        try:
                            if hasattr(self.audio_manager, 'set_assistant_speaking') and self.audio_manager:
                                self.audio_manager.set_assistant_speaking(False)
                        except Exception as am_error:
                            logger.error(f"Error restoring energy threshold after initial greeting: {am_error}")
                        
                        # Mark greeting as complete
                        initial_greeting_complete = True
                        logger.info("Initial greeting completed or interrupted, continuing...")
                else:
                    logger.error("TTS not available for initial greeting")
            except Exception as greeting_error:
                logger.error(f"Error during initial greeting: {greeting_error}")
                logger.error(traceback.format_exc())
                # Continue even if greeting fails
                
            # Print some helpful info
            print(f"🔊 Say '{self.wake_word}' followed by your query. For example:")
            print(f"🗣️  '{self.wake_word}, what time is it?'")
            print(f"🗣️  '{self.wake_word}, tell me a joke'")
            print("")
            print("⌨️  Press SPACE to interrupt speech")
            print("⌨️  Press Ctrl+C to exit")
            
            # Main loop - just keep the process alive
            max_errors = 5
            error_count = 0
            
            # Check if space was hit before reaching the main loop
            if self.space_key_hit:
                logger.warning("=== SPACE KEY DEBUG: Space key was hit during startup, but continuing ===")
                # Reset the flag to continue
                self.space_key_hit = False
                
            while not self.should_stop:
                try:
                    # Check if space was hit
                    if self.space_key_hit:
                        logger.info("=== SPACE KEY DEBUG: Space key hit detected in main loop, resetting flag ===")
                        self.space_key_hit = False
                        
                    time.sleep(0.1)
                    # Reset error count during normal operation
                    error_count = 0
                except KeyboardInterrupt:
                    # Handle keyboard interrupt directly in the main loop
                    logger.info("Keyboard interrupt received in main loop")
                    print("\n👋 Keyboard interrupt received. Press again to exit.")
                    
                    # Give user a chance to press Ctrl+C again to exit
                    try:
                        time.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("Second keyboard interrupt, initiating shutdown...")
                        print("\n👋 Shutting down...")
                        self.should_stop = True
                except Exception as e:
                    # Handle other exceptions in the main loop
                    logger.error(f"Error in main loop: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Count errors and exit if too many occur in succession
                    error_count += 1
                    if error_count > max_errors:
                        logger.error(f"Too many consecutive errors ({error_count}), shutting down")
                        self.should_stop = True
                    
                    # Sleep a bit longer after an error to avoid tight error loops
                    time.sleep(0.5)
                
            logger.info("Main loop exited, performing cleanup...")
            self.cleanup()
            return True
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            print("\n👋 Shutting down by keyboard interrupt...")
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ Error: {e}")
            self.cleanup()
            
        return False
    
    def _handle_space_key_during_startup(self):
        """Special handler for space key during startup to prevent program termination"""
        logger.warning("=== SPACE KEY DEBUG: Space key pressed during startup/greeting ===")
        
        # Set a flag that we can check in the main loop
        self.space_key_hit = True
        
        # Try to stop TTS safely with multiple layers of protection
        try:
            # First check if TTS attribute exists and is not None
            if not hasattr(self, 'tts') or self.tts is None:
                logger.warning("=== SPACE KEY DEBUG: No TTS object available to stop ===")
                # Even without a TTS object, try to force stop any audio output
                try:
                    import sounddevice as sd
                    sd.stop()
                    logger.info("=== SPACE KEY DEBUG: Called sd.stop() directly (no TTS object) ===")
                except Exception as sd_err:
                    logger.error(f"=== SPACE KEY DEBUG: Error calling direct sd.stop(): {sd_err} ===")
                return
            
            # Now we know self.tts exists and is not None
            logger.info("=== SPACE KEY DEBUG: TTS object exists ===")
            
            # Check if it has a stop method
            if not hasattr(self.tts, 'stop'):
                logger.warning("=== SPACE KEY DEBUG: TTS object has no stop method ===")
                # Try the direct sounddevice approach as fallback
                try:
                    import sounddevice as sd
                    sd.stop()
                    logger.info("=== SPACE KEY DEBUG: Called sd.stop() directly (no stop method) ===")
                except Exception as sd_err:
                    logger.error(f"=== SPACE KEY DEBUG: Error calling direct sd.stop(): {sd_err} ===")
                return
                
            # We have a TTS object with a stop method, try to use it
            logger.info("=== SPACE KEY DEBUG: Stopping TTS during startup ===")
            
            # Use a separate thread with timeout for extra safety
            def safe_stop_thread():
                try:
                    self.tts.stop()
                    logger.info("=== SPACE KEY DEBUG: TTS.stop() completed in thread ===")
                except Exception as stop_err:
                    logger.error(f"=== SPACE KEY DEBUG: Error in threaded TTS.stop(): {stop_err} ===")
                    # Try as a last resort
                    try:
                        import sounddevice as sd
                        sd.stop()
                        logger.info("=== SPACE KEY DEBUG: Called sd.stop() as last resort ===")
                    except Exception:
                        pass
            
            # Create and start the thread
            stop_thread = threading.Thread(target=safe_stop_thread)
            stop_thread.daemon = True
            stop_thread.start()
            
            # Wait with timeout
            stop_thread.join(timeout=1.0)
            if stop_thread.is_alive():
                logger.warning("=== SPACE KEY DEBUG: TTS.stop() thread timed out ===")
                # Try the direct approach as an additional fallback
                try:
                    import sounddevice as sd
                    sd.stop()
                    logger.info("=== SPACE KEY DEBUG: Called sd.stop() after thread timeout ===")
                except Exception:
                    pass
            else:
                logger.info("=== SPACE KEY DEBUG: TTS stopped successfully ===")
            
        except Exception as e:
            logger.error(f"=== SPACE KEY DEBUG: Error in space key handler: {e} ===")
            # As an absolute last resort, try to stop any audio playback directly
            try:
                import sounddevice as sd
                sd.stop()
                logger.info("=== SPACE KEY DEBUG: Called sd.stop() after exception ===")
            except Exception:
                pass
        
        # Mark the assistant as not speaking, regardless of what happened above
        try:
            if hasattr(self, 'audio_manager') and self.audio_manager:
                self.audio_manager.set_assistant_speaking(False)
                logger.info("=== SPACE KEY DEBUG: Reset assistant speaking state ===")
        except Exception as am_err:
            logger.error(f"=== SPACE KEY DEBUG: Error resetting speaking state: {am_err} ===")
        
        # Indicate that we've handled it
        logger.info("=== SPACE KEY DEBUG: Space key during startup handled safely ===")
        
        # Play a sound to indicate interruption, but with extra protection
        try:
            self._play_sound(frequency=600, duration=120)
        except Exception as sound_err:
            logger.error(f"=== SPACE KEY DEBUG: Error playing interrupt sound: {sound_err} ===")
            # Don't let sound errors propagate
    
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
            # config.use_mcp = args.use_mcp  # Removed to prevent override - get from config.yaml only
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