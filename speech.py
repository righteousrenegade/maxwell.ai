import os
import tempfile
import time
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from kokoro_onnx import Kokoro
from utils import setup_logger

logger = setup_logger()

class TextToSpeech:
    def __init__(self, voice="bm_lewis", speed=1.25):
        self.voice = voice
        self.speed = speed
        
        # Paths to Kokoro model and voices files
        model_path = "./kokoro-v1.0.onnx"
        voices_path = "./voices-v1.0.bin"
        
        # Check if files exist
        if not os.path.exists(model_path):
            logger.error(f"Kokoro model file not found at {model_path}")
            raise FileNotFoundError(f"Kokoro model file not found at {model_path}")
            
        if not os.path.exists(voices_path):
            logger.error(f"Kokoro voices file not found at {voices_path}")
            raise FileNotFoundError(f"Kokoro voices file not found at {voices_path}")
        
        # Initialize Kokoro with required paths
        self.tts = Kokoro(model_path, voices_path)
        
        # Validate and set the voice
        self.voice_style = self.validate_voice(voice, self.tts)
        
        self.stop_event = threading.Event()
        self.temp_dir = tempfile.mkdtemp()
        
    def validate_voice(self, voice, kokoro):
        """Validate if the voice is supported and handle voice blending."""
        try:
            supported_voices = set(kokoro.get_voices())
            
            # Parse comma seperated voices for blend
            if ',' in voice:
                voices = []
                weights = []
                
                # Parse voice:weight pairs
                for pair in voice.split(','):
                    if ':' in pair:
                        v, w = pair.strip().split(':')
                        voices.append(v.strip())
                        weights.append(float(w.strip()))
                    else:
                        voices.append(pair.strip())
                        weights.append(50.0)  # Default to 50% if no weight specified
                
                if len(voices) != 2:
                    raise ValueError("voice blending needs two comma separated voices")
                     
                # Validate voice
                for v in voices:
                    if v not in supported_voices:
                        supported_voices_list = ', '.join(sorted(supported_voices))
                        raise ValueError(f"Unsupported voice: {v}\nSupported voices are: {supported_voices_list}")
                 
                # Normalize weights to sum to 100
                total = sum(weights)
                if total != 100:
                    weights = [w * (100/total) for w in weights]
                
                # Create voice blend style
                style1 = kokoro.get_voice_style(voices[0])
                style2 = kokoro.get_voice_style(voices[1])
                blend = np.add(style1 * (weights[0]/100), style2 * (weights[1]/100))
                return blend
                 
            # Single voice validation
            if voice not in supported_voices:
                supported_voices_list = ', '.join(sorted(supported_voices))
                raise ValueError(f"Unsupported voice: {voice}\nSupported voices are: {supported_voices_list}")
            return voice
        except Exception as e:
            logger.error(f"Error getting supported voices: {e}")
            raise ValueError(f"Voice validation error: {e}")
        
    def speak(self, text):
        if not text:
            return
            
        # Reset stop event
        self.stop_event.clear()
        
        try:
            # Generate audio using the 'create' method
            if isinstance(self.voice_style, np.ndarray):
                # Using a blended voice style
                audio_data = self.tts.create(text, style=self.voice_style)
            else:
                # Using a single voice
                audio_data = self.tts.create(text, voice=self.voice_style)
            
            # Handle different audio data formats
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                # If it's a tuple of (audio, sample_rate)
                audio, sample_rate = audio_data
            else:
                # Otherwise, assume it's just the audio data
                audio = audio_data
                sample_rate = 24000  # Default sample rate
            
            # Ensure audio is a 1D numpy array for soundfile
            if isinstance(audio, np.ndarray):
                if audio.ndim > 1:
                    # If it's a 2D array with multiple channels, take the first channel
                    if audio.shape[1] > 0:
                        audio = audio[:, 0]
                    else:
                        audio = audio.flatten()
            
            # Create a temporary file
            temp_file = os.path.join(self.temp_dir, "tts_output.wav")
            
            # Write audio to file
            sf.write(temp_file, audio, sample_rate)
            
            # Play audio
            self._play_audio(temp_file)
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Failed to speak: {text}")
            
    def _play_audio(self, audio_file):
        """Play audio file with interruption support"""
        try:
            # Load the audio file
            data, fs = sf.read(audio_file)
            
            # Ensure data is the right shape for sounddevice
            if len(data.shape) > 1 and data.shape[1] > 1:
                # Multi-channel audio - keep only first channel for simplicity
                data = data[:, 0]
            
            # Apply speed adjustment without using scipy
            if self.speed != 1.0:
                # Simple speed adjustment by skipping samples
                if self.speed > 1.0:
                    # For faster speech, skip samples
                    step = self.speed
                    indices = np.arange(0, len(data), step)
                    data = data[indices.astype(int)]
            
            # Play the audio
            sd.play(data, fs)
            
            # Wait for playback to finish or stop event
            while sd.get_stream().active and not self.stop_event.is_set():
                time.sleep(0.1)
            
            # Stop playback if interrupted
            if self.stop_event.is_set():
                sd.stop()
                logger.info("Audio playback stopped")
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    def stop(self):
        """Stop the current speech."""
        logger.info("Stopping speech...")
        self.stop_event.set()
        
    def cleanup(self):
        # Clean up temporary files
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except:
                    pass
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
                
    @staticmethod
    def list_available_voices():
        # Since we need model and voices paths for Kokoro, we need to initialize it first
        model_path = "./kokoro-v1.0.onnx"
        voices_path = "./voices-v1.0.bin"
        
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            print("Error: Kokoro model or voices file not found")
            return
            
        # Initialize with a default voice to get access to the model
        kokoro = Kokoro(model_path, voices_path)
        # Get available voices
        voices = kokoro.get_voices()
        print("Available voices:")
        for voice in sorted(voices):
            print(f"- {voice}")
