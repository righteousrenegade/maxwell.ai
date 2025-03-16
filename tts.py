#!/usr/bin/env python3
"""
Text-to-speech module for the voice assistant.

This module contains the KokoroTTS class for text-to-speech functionality.
"""

# Standard library imports
import os
import sys
import time
import logging
import threading
import queue

# Third-party imports
import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flags for control
is_speaking = False
last_speaking_time = 0
waiting_for_user_input = False
pending_user_input = None
is_interrupted = False

class KokoroTTS:
    """Text-to-speech using Kokoro."""
    
    def __init__(self, model_path="kokoro-v1.0.onnx", voices_path="voices-v1.0.bin", 
                 voice="bm_lewis", language="en-us", speed=1.0, skip_init=False,
                 auto_download=False):
        """Initialize Kokoro TTS."""
        self.on_speech_finished = None  # Callback for when speech is finished
        self.on_speech_interrupted = None  # Callback for when speech is interrupted
        self.current_stream = None  # Reference to the current audio stream
        
        if skip_init:
            logger.warning("Skipping TTS initialization (for testing only)")
            self.kokoro = None
            self.voice = voice
            self.language = language
            self.speed = speed
            return
        
        # URLs for downloading model files
        model_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
        voices_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
            
        # Check if model files exist and download if needed
        files_to_download = []
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            files_to_download.append(("model", model_path, model_url))
            
        if not os.path.exists(voices_path):
            logger.warning(f"Voices file not found: {voices_path}")
            files_to_download.append(("voices", voices_path, voices_url))
        
        # If files are missing, handle download
        if files_to_download:
            # Print information about missing files
            print("\n" + "="*50)
            print("KOKORO TTS MODEL FILES MISSING")
            print("="*50)
            print("\nThe following files are needed for text-to-speech:")
            for file_type, path, url in files_to_download:
                print(f"- {file_type.capitalize()} file: {path}")
            
            # Approximate file sizes
            print("\nApproximate file sizes:")
            print("- Model file: ~40 MB")
            print("- Voices file: ~15 MB")
            
            if auto_download:
                print("\nAuto-download is enabled. Downloading files...")
                self._download_files(files_to_download)
            else:
                # Ask user for confirmation
                print("\nWould you like to download these files now? (y/n)")
                choice = input().strip().lower()
                
                if choice in ["y", "yes"]:
                    self._download_files(files_to_download)
                else:
                    print("\nDownload cancelled. TTS will be disabled.")
                    logger.warning("TTS model files download cancelled by user")
                    self.kokoro = None
                    self.voice = voice
                    self.language = language
                    self.speed = speed
                    return
        
        try:
            # Initialize Kokoro TTS
            logger.info(f"Initializing Kokoro TTS with model: {model_path}, voices: {voices_path}")
            self.kokoro = Kokoro(model_path, voices_path)
            self.voice = voice
            self.language = language
            self.speed = speed
            
            # Log available voices and languages
            logger.info(f"Using voice: {voice}, language: {language}, speed: {speed}")
            
            # Test TTS
            logger.info("Testing TTS...")
            try:
                samples, sample_rate = self.kokoro.create(
                    "TTS initialized successfully.", 
                    voice=self.voice, 
                    speed=self.speed, 
                    lang=self.language
                )
                logger.info("TTS test successful")
            except Exception as e:
                logger.warning(f"TTS test with parameters failed: {e}. Trying with default parameters.")
                try:
                    samples, sample_rate = self.kokoro.create("TTS initialized successfully.")
                    logger.info("TTS test with default parameters successful")
                except Exception as e2:
                    logger.error(f"TTS test with default parameters failed: {e2}")
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            self.kokoro = None
    
    def _download_files(self, files_to_download):
        """Download missing model files.
        
        Args:
            files_to_download: List of tuples (file_type, path, url)
        """
        import requests
        from tqdm import tqdm
        
        for file_type, path, url in files_to_download:
            try:
                print(f"\nDownloading {file_type} file from {url}...")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
                
                # Download with progress bar
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                
                with open(path, "wb") as f, tqdm(
                    desc=f"{file_type.capitalize()} file",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)
                
                print(f"{file_type.capitalize()} file downloaded successfully to {path}")
                logger.info(f"{file_type.capitalize()} file downloaded successfully to {path}")
            except Exception as e:
                print(f"Error downloading {file_type} file: {e}")
                logger.error(f"Error downloading {file_type} file: {e}")
                return False
        
        return True
    
    def list_available_voices(self):
        """List available voices."""
        if self.kokoro is None:
            print("TTS is not initialized")
            return
            
        try:
            # Get available voices
            voices = self.kokoro.voices
            
            print("\nAvailable voices:")
            for voice in voices:
                print(f"- {voice}")
        except Exception as e:
            print(f"Error listing voices: {e}")
    
    def list_available_languages(self):
        """List available languages."""
        if self.kokoro is None:
            print("TTS is not initialized")
            return
            
        try:
            # Get available languages
            languages = self.kokoro.languages
            
            print("\nAvailable languages:")
            for lang in languages:
                print(f"- {lang}")
        except Exception as e:
            print(f"Error listing languages: {e}")
    
    def _chunk_text(self, text, max_chunk_length=200):
        """Split text into chunks for better TTS handling.
        
        Args:
            text: Text to split
            max_chunk_length: Maximum chunk length
            
        Returns:
            List of text chunks
        """
        # If text is short enough, return as is
        if len(text) <= max_chunk_length:
            return [text]
            
        # Split by sentences with more careful handling
        import re
        
        # Define sentence-ending patterns
        sentence_end_pattern = r'(?<=[.!?])\s+'
        
        # Split text into sentences
        sentences = re.split(sentence_end_pattern, text)
        
        # Make sure sentence endings are preserved
        for i in range(len(sentences) - 1):
            # Find the ending punctuation from the original text
            if i < len(text):
                end_match = re.search(r'[.!?]', text[text.find(sentences[i]) + len(sentences[i]):])
                if end_match:
                    sentences[i] += end_match.group(0)
        
        # Ensure the last sentence has proper ending
        if sentences and not sentences[-1].rstrip().endswith(('.', '!', '?')):
            sentences[-1] = sentences[-1].rstrip() + '.'
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # If this sentence alone is longer than max_chunk_length, split it further
            if len(sentence) > max_chunk_length:
                # If we have accumulated text in current_chunk, add it first
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long sentence by phrases (commas, semicolons, etc.)
                phrase_pattern = r'(?<=[,;:])\s+'
                phrases = re.split(phrase_pattern, sentence)
                
                # Ensure phrase endings are preserved
                for i in range(len(phrases) - 1):
                    if i < len(sentence):
                        end_match = re.search(r'[,;:]', sentence[sentence.find(phrases[i]) + len(phrases[i]):])
                        if end_match:
                            phrases[i] += end_match.group(0)
                
                phrase_chunk = ""
                for phrase in phrases:
                    if len(phrase_chunk) + len(phrase) + 1 <= max_chunk_length:
                        if phrase_chunk:
                            phrase_chunk += " " + phrase
                        else:
                            phrase_chunk = phrase
                    else:
                        if phrase_chunk:
                            chunks.append(phrase_chunk.strip())
                        phrase_chunk = phrase
                
                if phrase_chunk:
                    chunks.append(phrase_chunk.strip())
                continue
            
            # If adding this sentence would make the chunk too long, start a new chunk
            if len(current_chunk) + len(sentence) + 1 > max_chunk_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Ensure no empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        # If we have only one chunk, return it
        if len(chunks) == 1:
            return chunks
            
        # For multiple chunks, ensure they're not too small (combine small chunks)
        min_chunk_length = 50  # Minimum chunk length to avoid too many small chunks
        combined_chunks = []
        current_combined = ""
        
        for chunk in chunks:
            if len(current_combined) + len(chunk) + 1 <= max_chunk_length:
                if current_combined:
                    current_combined += " " + chunk
                else:
                    current_combined = chunk
            else:
                if current_combined:
                    combined_chunks.append(current_combined)
                current_combined = chunk
        
        if current_combined:
            combined_chunks.append(current_combined)
        
        return combined_chunks
    
    def _prepare_all_audio(self, chunks):
        """Prepare audio for all chunks before playback.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Tuple of (concatenated_samples, sample_rate)
        """
        all_samples = []
        sample_rate = 16000  # Default sample rate
        
        for chunk in chunks:
            try:
                # Try to create audio with the specified parameters
                samples, chunk_sample_rate = self.kokoro.create(
                    chunk, 
                    voice=self.voice, 
                    speed=self.speed, 
                    lang=self.language
                )
                all_samples.append(samples)
                sample_rate = chunk_sample_rate  # Update sample rate (should be consistent)
            except TypeError as e:
                # If we get a TypeError, it might be because the create method has different parameters
                logger.warning(f"Error with TTS parameters: {e}. Trying with default parameters.")
                try:
                    # Try with just the text
                    samples, chunk_sample_rate = self.kokoro.create(chunk)
                    all_samples.append(samples)
                    sample_rate = chunk_sample_rate  # Update sample rate
                except Exception as e2:
                    logger.error(f"Error in text-to-speech with default parameters: {e2}")
                    # Add a short silence to maintain flow
                    all_samples.append(np.zeros(500, dtype=np.float32))
            except Exception as e:
                logger.error(f"Error creating audio for chunk: {e}")
                # Add a short silence to maintain flow
                all_samples.append(np.zeros(500, dtype=np.float32))
        
        # Concatenate all samples with a tiny silence between chunks for natural pauses
        if not all_samples:
            return np.zeros(1000, dtype=np.float32), sample_rate
            
        # Add a very small silence between chunks (10ms) for more natural speech
        silence = np.zeros(int(sample_rate * 0.01), dtype=np.float32)
        concatenated_samples = all_samples[0]
        
        for samples in all_samples[1:]:
            concatenated_samples = np.concatenate((concatenated_samples, silence, samples))
        
        return concatenated_samples, sample_rate
    
    def _check_for_interrupt(self, stream, interval=0.002):
        """Check for interrupt flag at regular intervals.
        
        Args:
            stream: Audio stream to stop if interrupted
            interval: Check interval in seconds (2ms for ultra-responsive interruption)
        """
        global is_interrupted, is_speaking, last_speaking_time, waiting_for_user_input, pending_user_input
        
        # Check very frequently for better responsiveness
        while stream.active and not is_interrupted:
            time.sleep(interval)
            
        # If interrupted, stop immediately
        if is_interrupted:
            logger.info("Interrupt detected, stopping audio stream IMMEDIATELY")
            try:
                # Stop the stream immediately if it's still active
                if stream.active:
                    stream.stop()
                    logger.info("Audio stream stopped due to interrupt")
                
                # Reset speaking flags immediately
                is_speaking = False
                last_speaking_time = time.time()
                waiting_for_user_input = True
                
                # Clear any pending input that might be an interrupt command
                if pending_user_input and is_interrupt_command(pending_user_input):
                    logger.info(f"Clearing pending interrupt command: {pending_user_input}")
                    pending_user_input = None
                
                # Call the speech interrupted callback if set
                if self.on_speech_interrupted:
                    self.on_speech_interrupted()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
                
            # Do not reset the interrupted flag here - let the callback handle it
    
    def _wait_for_stream(self, stream, samples, sample_rate, interval=0.05):
        """Wait for the audio stream to finish playing.
        
        Args:
            stream: Audio stream to wait for
            samples: Audio samples
            sample_rate: Sample rate
            interval: Check interval in seconds
        """
        # Calculate the expected duration of the audio
        duration = len(samples) / sample_rate
        
        # Wait for the expected duration plus a small buffer
        start_time = time.time()
        while stream.active and time.time() - start_time < duration + 0.5:
            time.sleep(interval)
    
    def speak(self, text):
        """Convert text to speech and play it.
        
        Args:
            text: Text to speak
        """
        global is_speaking, last_speaking_time, waiting_for_user_input, pending_user_input, is_interrupted
        
        if not text or not text.strip():
            return
            
        try:
            # Process text into chunks for better handling
            chunks = self._chunk_text(text)
            logger.debug(f"Split text into {len(chunks)} chunks")
            
            # Prepare all audio data as a single continuous stream
            samples, sample_rate = self._prepare_all_audio(chunks)
            
            # Check if we should stop speaking
            if is_interrupted:
                logger.info("Speech interrupted before playback")
                is_speaking = False  # Ensure flag is reset
                last_speaking_time = time.time()
                waiting_for_user_input = True
                return
            
            try:
                # Start a single audio stream for the entire text
                stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    callback=None,
                    finished_callback=None
                )
                stream.start()
                self.current_stream = stream
                
                # Start a separate thread to check for interrupts BEFORE writing samples
                interrupt_thread = threading.Thread(
                    target=self._check_for_interrupt,
                    args=(stream, 0.002)  # Check every 2ms for better responsiveness
                )
                interrupt_thread.daemon = True
                interrupt_thread.start()
                
                # Write all samples to the stream at once
                if not is_interrupted:  # Check again before writing
                    stream.write(samples)
                
                # Wait for the audio to finish or be interrupted
                self._wait_for_stream(stream, samples, sample_rate, 0.05)
                
                # Check if we were interrupted
                if is_interrupted:
                    logger.info("Speech interrupted during playback")
                    return
                    
            except KeyboardInterrupt:
                # Stop audio on keyboard interrupt
                logger.info("Speech interrupted by keyboard")
                if stream and stream.active:
                    stream.stop()
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
        finally:
            # Reset speaking flag and update last speaking time
            is_speaking = False
            last_speaking_time = time.time()  # Record when speaking finished
            waiting_for_user_input = True  # Ready for user input after speaking
            self.current_stream = None  # Clear the current stream reference
            
            # Call the speech finished callback if set
            if not is_interrupted and self.on_speech_finished:
                try:
                    self.on_speech_finished()
                except Exception as e:
                    logger.error(f"Error in speech finished callback: {e}")
            elif is_interrupted and self.on_speech_interrupted:
                # Ensure the interrupted callback is called if we were interrupted
                try:
                    self.on_speech_interrupted()
                except Exception as e:
                    logger.error(f"Error in speech interrupted callback: {e}")
    
    def stream_speak(self, text_chunk):
        """Convert a chunk of text to speech and play it immediately.
        
        This method is designed to be called repeatedly with small chunks of text
        as they become available from a streaming source.
        
        Args:
            text_chunk: Small chunk of text to speak
        """
        global is_speaking, last_speaking_time, waiting_for_user_input
        
        if not text_chunk or not text_chunk.strip():
            return
            
        # Skip if TTS is not initialized
        if self.kokoro is None:
            logger.info(f"Would speak chunk (TTS disabled): {text_chunk}")
            return
        
        # Set speaking flag if not already speaking
        if not is_speaking:
            is_speaking = True
            waiting_for_user_input = False  # Not waiting for input while speaking
        
        try:
            # Check if we should stop speaking
            if is_interrupted:
                return
            
            logger.debug(f"Speaking stream chunk: {text_chunk}")
            
            # Process text into smaller chunks if needed
            chunks = self._chunk_text(text_chunk)
            
            # Prepare all audio data as a single continuous stream
            samples, sample_rate = self._prepare_all_audio(chunks)
            
            # Check if we should stop speaking
            if is_interrupted:
                logger.info("Stream speech interrupted before playback")
                return
            
            try:
                # Start a single audio stream for the entire chunk
                stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    callback=None,
                    finished_callback=None
                )
                stream.start()
                self.current_stream = stream
                
                # Write all samples to the stream at once
                stream.write(samples)
                
                # Start a separate thread to check for interrupts
                interrupt_thread = threading.Thread(
                    target=self._check_for_interrupt,
                    args=(stream, 0.002)  # Check every 2ms for better responsiveness
                )
                interrupt_thread.daemon = True
                interrupt_thread.start()
                
                # Wait for the audio to finish or be interrupted
                self._wait_for_stream(stream, samples, sample_rate, 0.05)
                
                # Check if we were interrupted
                if is_interrupted:
                    logger.info("Stream speech interrupted during playback")
                    return
                    
            except KeyboardInterrupt:
                # Stop audio on keyboard interrupt
                logger.info("Stream speech interrupted by keyboard")
                if stream.active:
                    stream.stop()
                return
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            
    def stop_speaking(self):
        """Stop speaking immediately."""
        global is_speaking, is_interrupted, last_speaking_time, waiting_for_user_input, pending_user_input
        
        logger.info("Stopping speech IMMEDIATELY")
        # Set the interrupted flag first
        is_interrupted = True
        
        # Stop the current stream if it exists - do this FIRST
        if self.current_stream is not None:
            try:
                if self.current_stream.active:
                    # Force stop the stream immediately
                    self.current_stream.stop()
                    self.current_stream = None  # Clear the reference
                    logger.info("Audio stream forcefully stopped")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
        
        # Reset speaking flag immediately
        is_speaking = False
        last_speaking_time = time.time()
        waiting_for_user_input = True
        
        # Clear any pending input that might be an interrupt command
        if pending_user_input and is_interrupt_command(pending_user_input):
            logger.info(f"Clearing pending interrupt command: {pending_user_input}")
            pending_user_input = None
        
        # Call the speech interrupted callback if set
        if self.on_speech_interrupted:
            try:
                self.on_speech_interrupted()
            except Exception as e:
                logger.error(f"Error in speech interrupted callback: {e}")
                
        logger.info("Speech stopped successfully") 