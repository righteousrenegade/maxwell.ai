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
    
    def _chunk_text(self, text, max_chunk_length=150):
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
            
        # Split by sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would make the chunk too long, start a new chunk
            if len(current_chunk) + len(sentence) > max_chunk_length and current_chunk:
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
            
        return chunks
    
    def _check_for_interrupt(self, stream, interval=0.05):
        """Check for interrupt flag at regular intervals.
        
        Args:
            stream: Audio stream to stop if interrupted
            interval: Check interval in seconds
        """
        global is_interrupted
        
        while stream.active and not is_interrupted:
            time.sleep(interval)  # Check more frequently for better responsiveness
            
        if is_interrupted and stream.active:
            logger.info("Interrupt detected, stopping audio stream")
            stream.stop()
            
            # Call the speech interrupted callback if set
            if self.on_speech_interrupted:
                self.on_speech_interrupted()
    
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
            
        # Skip if TTS is not initialized
        if self.kokoro is None:
            logger.info(f"Would speak (TTS disabled): {text}")
            last_speaking_time = time.time()  # Update last speaking time even when TTS is disabled
            waiting_for_user_input = True  # Ready for user input after "speaking"
            
            # Call the speech finished callback if set
            if self.on_speech_finished:
                self.on_speech_finished()
            return
        
        # Set speaking flag
        is_speaking = True
        waiting_for_user_input = False  # Not waiting for input while speaking
        is_interrupted = False  # Reset interrupted flag
        
        try:
            # Process text in chunks for better handling
            chunks = self._chunk_text(text)
            
            for chunk in chunks:
                # Check if we should stop speaking
                if is_interrupted:
                    logger.info("Speech interrupted before chunk processing")
                    break
                
                logger.debug(f"Speaking chunk: {chunk[:50]}...")
                try:
                    # Try to create audio with the specified parameters
                    samples, sample_rate = self.kokoro.create(
                        chunk, 
                        voice=self.voice, 
                        speed=self.speed, 
                        lang=self.language
                    )
                    
                    # Play the audio with a way to interrupt
                    try:
                        # Start the audio stream
                        stream = sd.OutputStream(
                            samplerate=sample_rate,
                            channels=1,
                            callback=None,
                            finished_callback=None
                        )
                        stream.start()
                        self.current_stream = stream
                        
                        # Write the samples to the stream
                        stream.write(samples)
                        
                        # Start a separate thread to check for interrupts
                        interrupt_thread = threading.Thread(
                            target=self._check_for_interrupt,
                            args=(stream, 0.05)  # Check every 50ms for better responsiveness
                        )
                        interrupt_thread.daemon = True
                        interrupt_thread.start()
                        
                        # Wait for the audio to finish or be interrupted
                        self._wait_for_stream(stream, samples, sample_rate, 0.05)
                        
                        # Check if we were interrupted
                        if is_interrupted:
                            logger.info("Speech interrupted during playback")
                            break
                            
                    except KeyboardInterrupt:
                        # Stop audio on keyboard interrupt
                        logger.info("Speech interrupted by keyboard")
                        if stream.active:
                            stream.stop()
                        break
                except TypeError as e:
                    # If we get a TypeError, it might be because the create method has different parameters
                    logger.warning(f"Error with TTS parameters: {e}. Trying with default parameters.")
                    try:
                        # Try with just the text
                        samples, sample_rate = self.kokoro.create(chunk)
                        
                        # Play the audio with a way to interrupt
                        try:
                            # Start the audio stream
                            stream = sd.OutputStream(
                                samplerate=sample_rate,
                                channels=1,
                                callback=None,
                                finished_callback=None
                            )
                            stream.start()
                            self.current_stream = stream
                            
                            # Write the samples to the stream
                            stream.write(samples)
                            
                            # Start a separate thread to check for interrupts
                            interrupt_thread = threading.Thread(
                                target=self._check_for_interrupt,
                                args=(stream, 0.05)  # Check every 50ms for better responsiveness
                            )
                            interrupt_thread.daemon = True
                            interrupt_thread.start()
                            
                            # Wait for the audio to finish or be interrupted
                            self._wait_for_stream(stream, samples, sample_rate, 0.05)
                            
                            # Check if we were interrupted
                            if is_interrupted:
                                logger.info("Speech interrupted during playback")
                                break
                                
                        except KeyboardInterrupt:
                            # Stop audio on keyboard interrupt
                            logger.info("Speech interrupted by keyboard")
                            if stream.active:
                                stream.stop()
                            break
                    except Exception as e2:
                        logger.error(f"Error in text-to-speech with default parameters: {e2}")
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
                self.on_speech_finished()
            elif is_interrupted and self.on_speech_interrupted:
                # Ensure the interrupted callback is called if we were interrupted
                self.on_speech_interrupted()
    
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
            try:
                # Try to create audio with the specified parameters
                samples, sample_rate = self.kokoro.create(
                    text_chunk, 
                    voice=self.voice, 
                    speed=self.speed, 
                    lang=self.language
                )
                
                # Play the audio with a way to interrupt
                try:
                    # Start the audio stream
                    stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        callback=None,
                        finished_callback=None
                    )
                    stream.start()
                    self.current_stream = stream
                    
                    # Write the samples to the stream
                    stream.write(samples)
                    
                    # Start a separate thread to check for interrupts
                    interrupt_thread = threading.Thread(
                        target=self._check_for_interrupt,
                        args=(stream, 0.05)  # Check every 50ms for better responsiveness
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
            except TypeError as e:
                # If we get a TypeError, it might be because the create method has different parameters
                logger.warning(f"Error with TTS parameters: {e}. Trying with default parameters.")
                try:
                    # Try with just the text
                    samples, sample_rate = self.kokoro.create(text_chunk)
                    
                    # Play the audio with a way to interrupt
                    try:
                        # Start the audio stream
                        stream = sd.OutputStream(
                            samplerate=sample_rate,
                            channels=1,
                            callback=None,
                            finished_callback=None
                        )
                        stream.start()
                        self.current_stream = stream
                        
                        # Write the samples to the stream
                        stream.write(samples)
                        
                        # Start a separate thread to check for interrupts
                        interrupt_thread = threading.Thread(
                            target=self._check_for_interrupt,
                            args=(stream, 0.05)  # Check every 50ms for better responsiveness
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
                except Exception as e2:
                    logger.error(f"Error in text-to-speech with default parameters: {e2}")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            
    def stop_speaking(self):
        """Stop speaking immediately."""
        global is_speaking, is_interrupted
        
        if is_speaking:
            logger.info("Stopping speech")
            is_interrupted = True
            
            # Stop the current stream if it exists
            if self.current_stream is not None and self.current_stream.active:
                self.current_stream.stop()
            
            # Call the speech interrupted callback if set
            if self.on_speech_interrupted:
                self.on_speech_interrupted() 