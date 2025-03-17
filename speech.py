import os
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
from kokoro_onnx import Kokoro
from utils import setup_logger
import logging
import platform
import traceback
import msvcrt  # Windows-specific module for keyboard input without blocking

# Get the logger instance
logger = logging.getLogger("maxwell")
if not logger.handlers:
    logger = setup_logger()

# Global flag for interruption
_interrupt_playback = False

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
        
        self.temp_dir = tempfile.mkdtemp()
        self._speech_queue = []
        self._is_speaking = False
        
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
            
        try:
            # Reset the global interrupt flag
            global _interrupt_playback
            _interrupt_playback = False
            
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
            
            # Print instructions for interrupting speech
            print("Press SPACE to interrupt speech")
            
            # Play audio
            self._play_audio(temp_file)
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Failed to speak: {text}")
            
    def _play_audio(self, audio_file):
        """Play audio file with support for interruption via keyboard"""
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
            
            # Set flag that we're speaking
            self._is_speaking = True
            
            # Use non-blocking play with manual keyboard interrupt detection
            try:
                # Convert to float32 to avoid dtype mismatch
                data = data.astype(np.float32)
                
                # Start playback non-blocking
                logger.info("Starting audio playback (press SPACE to interrupt)")
                sd.play(data, fs, blocking=False)
                
                # Monitor for keyboard interrupts while audio is playing
                while sd.get_stream().active:
                    # Check if a key is pressed without blocking
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        # Check if space key was pressed (byte code 32)
                        if key == b' ':
                            logger.info("SPACE key pressed, interrupting speech")
                            print("\n🛑 Speech interrupted!")
                            sd.stop()
                            
                            # Play an interruption beep
                            try:
                                if platform.system() == 'Windows':
                                    # Use Windows-specific beep
                                    import winsound
                                    winsound.Beep(600, 120)  # Lower-pitched longer beep for interruption
                                else:
                                    # Cross-platform alternative
                                    print('\a', end='', flush=True)  # ASCII bell character
                            except Exception as e:
                                logger.debug(f"Error playing interruption beep: {e}")
                                
                            break
                    # Brief sleep to prevent CPU hogging
                    time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Audio playback error: {e}")
                sd.stop()  # Make sure playback is stopped on error
            
            # Clear speaking flag when done
            self._is_speaking = False
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._is_speaking = False
            
    def stop(self):
        """Immediately stop any ongoing speech"""
        try:
            # Set the global flag to interrupt playback
            global _interrupt_playback
            _interrupt_playback = True
            logger.info("Setting interrupt flag to stop audio playback")
            
            # Additionally, try to stop any playing sound using sounddevice
            try:
                sd.stop()
                logger.info("Called sounddevice.stop()")
            except:
                pass
                
            # Additionally, try to stop any playing sound using platform-specific methods
            if platform.system() == 'Windows':
                import winsound
                # Stop any playing waveform
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                    logger.info("Called winsound.PlaySound(None, SND_PURGE)")
                except:
                    pass
                    
            # Clear any queue we might have
            self._speech_queue = []
            
            logger.info("TTS playback stop requested")
        except Exception as e:
            logger.error(f"Error stopping TTS: {e}")
            logger.error(traceback.format_exc())
        
    def cleanup(self):
        # Stop any ongoing playback
        self.stop()
        
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
