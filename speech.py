import os
import sounddevice as sd
import numpy as np
import threading
import time
from kokoro_onnx import Kokoro
from utils import setup_logger
import logging

# Get the logger instance
logger = logging.getLogger("maxwell")
if not logger.handlers:
    logger = setup_logger()

class DummyTTS:
    """Dummy TTS class that doesn't actually speak - for when TTS is disabled"""
    def __init__(self):
        self._is_speaking = False
        logger.info("DummyTTS initialized - no speech will be produced")
    
    def speak(self, text):
        if text:
            logger.info(f"[DUMMY TTS] Would say: {text}")
            print(f"🔇 {text}")
    
    def stop(self):
        pass
        
    def is_speaking(self):
        return False
        
    def cleanup(self):
        pass

class TextToSpeech:
    """Simple Kokoro-based TTS"""
    def __init__(self, voice='default', speed=1.0, engine=None, config=None):
        self.speed = speed
        self._is_speaking = False
        self._speaking_lock = threading.Lock()
        self.voice = voice if voice != 'default' else 'en_us_1'  # Default to first voice if not specified
        
        # Add speech state callbacks
        self.on_speaking_started = None
        self.on_speaking_stopped = None
        
        try:
            # Get paths for Kokoro from config if available
            model_path = None
            voices_path = None
            
            if config:
                # Try to get paths from config
                if hasattr(config, 'tts_model_path'):
                    model_path = config.tts_model_path
                if hasattr(config, 'tts_voices_path'):
                    voices_path = config.tts_voices_path
            
            # Fall back to default paths if not in config
            if not model_path or not voices_path:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, "kokoro-v1.0.onnx")
                voices_path = os.path.join(base_dir, "voices-v1.0.bin")
            
            logger.info(f"Initializing Kokoro TTS with model: {model_path}")
            logger.info(f"Voices file: {voices_path}")
            
            # Check if files exist
            if not os.path.exists(model_path):
                logger.error(f"TTS model file not found: {model_path}")
            if not os.path.exists(voices_path):
                logger.error(f"TTS voices file not found: {voices_path}")
            
            # Initialize Kokoro TTS with paths
            self.engine = Kokoro(model_path=model_path, voices_path=voices_path)
            
            # Check available voices
            try:
                available_voices = self.engine.get_voices()
                logger.info(f"Available voices: {', '.join(available_voices[:5])}...")
                
                # Use first available voice if specified voice not found
                if self.voice not in available_voices and len(available_voices) > 0:
                    original_voice = self.voice
                    self.voice = available_voices[0]
                    logger.info(f"Voice '{original_voice}' not found. Using '{self.voice}' instead.")
            except Exception as e:
                logger.warning(f"Could not get available voices: {e}")
            
            logger.info(f"Kokoro TTS initialized successfully with voice: {self.voice}")
            
            # Test audio output device
            try:
                audio_devices = sd.query_devices()
                default_output = sd.query_devices(kind='output')
                logger.info(f"Default audio output device: {default_output['name']}")
                logger.info(f"Audio output available: {'Yes' if len(audio_devices) > 0 else 'No'}")
            except Exception as e:
                logger.warning(f"Could not query audio devices: {e}")
            
            # Test TTS with a silent quick audio generation to ensure everything works
            try:
                logger.info("Testing TTS audio generation...")
                # Generate a short audio sample to test the system
                test_text = "Test."
                test_audio, test_rate = self.engine.create(text=test_text, voice=self.voice, speed=self.speed)
                logger.info(f"Test audio generated: shape={test_audio.shape}, sample_rate={test_rate}")
                logger.info("TTS test successful")
            except Exception as e:
                logger.error(f"TTS test failed: {e}")
            
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            print(f"⚠️ Failed to initialize Kokoro TTS: {e}")
            self.engine = None
            
    def speak(self, text):
        """Generate and play speech using Kokoro"""
        if not text or self.engine is None:
            logger.warning(f"Cannot speak: {'No text provided' if not text else 'TTS engine not initialized'}")
            return
            
        # Log longer text more intelligently
        if len(text) > 100:
            logger.info(f"Speaking longer text ({len(text)} chars): {text[:50]}...{text[-50:]}")
        else:
            logger.info(f"Speaking: {text}")
        
        # Set speaking state before starting thread
        with self._speaking_lock:
            self._is_speaking = True
            
        # Call the speaking started callback if it exists
        if self.on_speaking_started:
            try:
                self.on_speaking_started()
            except Exception as e:
                logger.error(f"Error in on_speaking_started callback: {e}")
            
        # Run in a separate thread to avoid blocking
        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()
        
        # Return quickly to not block the caller
        return thread
        
    def _speak_thread(self, text):
        """Thread function for speech generation and playback"""
        try:
            # Generate audio using the correct Kokoro API
            logger.info(f"Generating audio with voice '{self.voice}'...")
            audio, sample_rate = self.engine.create(
                text=text, 
                voice=self.voice,
                speed=self.speed
            )
            
            # Log audio information
            logger.info(f"Generated audio: shape={audio.shape}, sample_rate={sample_rate}, min={audio.min():.5f}, max={audio.max():.5f}")
            
            # If audio is all zeros or very small values, it won't be audible
            if np.abs(audio).max() < 0.01:
                logger.warning("Audio has very low amplitude - might not be audible")
            
            # Make sure audio is in the right format for sounddevice
            if audio.dtype != np.float32:
                logger.info(f"Converting audio from {audio.dtype} to float32")
                audio = audio.astype(np.float32)
            
            # Play audio with explicit error handling
            try:
                logger.info("Starting audio playback...")
                sd.play(audio, sample_rate)
                sd.wait()  # This will block until playback is finished
                logger.info("Audio playback completed")
            except Exception as playback_error:
                logger.error(f"Audio playback error: {playback_error}")
                # Try fallback playback method
                try:
                    logger.info("Trying fallback playback method...")
                    with sd.OutputStream(samplerate=sample_rate, channels=1) as stream:
                        stream.write(audio)
                    logger.info("Fallback playback completed")
                except Exception as fallback_error:
                    logger.error(f"Fallback playback also failed: {fallback_error}")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            if hasattr(e, "__traceback__"):
                logger.error(f"TTS error traceback: {str(e.__traceback__)}")
        finally:
            # Important: Clear speaking flag when done
            with self._speaking_lock:
                self._is_speaking = False
                
            # Call the speaking stopped callback if it exists
            if self.on_speaking_stopped:
                try:
                    self.on_speaking_stopped()
                except Exception as e:
                    logger.error(f"Error in on_speaking_stopped callback: {e}")
                    
            logger.info("Speech thread completed, speaking flag cleared")
            
    def stop(self):
        """Stop current speech"""
        if self._is_speaking:
            logger.info("Stopping TTS playback...")
            sd.stop()
            logger.info("TTS playback stopped")
            
            # Update speaking state
            with self._speaking_lock:
                self._is_speaking = False
                
            # Call the speaking stopped callback if it exists
            if self.on_speaking_stopped:
                try:
                    self.on_speaking_stopped()
                except Exception as e:
                    logger.error(f"Error in on_speaking_stopped callback: {e}")
            
    def is_speaking(self):
        """Check if currently speaking"""
        with self._speaking_lock:
            return self._is_speaking
        
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        logger.info("TTS engine cleaned up")
