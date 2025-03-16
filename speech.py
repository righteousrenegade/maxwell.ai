import os
import tempfile
import time
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import speech_recognition as sr
from kokoro_onnx import Kokoro
from utils import setup_logger
import sys
import logging

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
        """Validate if the voice is supported and handle voice blending.
        
        Format for blended voices: "voice1:weight,voice2:weight"
        Example: "af_sarah:60,am_adam:40" for 60-40 blend
        """
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
            
            # Debug the audio data shape
            logger.debug(f"Audio data type: {type(audio_data)}, shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'no shape'}")
            
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
            
            # Play audio in chunks to allow for interruption
            self._play_audio_with_interruption(temp_file)
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Failed to speak: {text}")
            
    def _play_audio_with_interruption(self, audio_file):
        """Play audio file in chunks to allow for interruption."""
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
                # This is a very basic approach but doesn't require scipy
                if self.speed > 1.0:
                    # For faster speech, skip samples
                    step = self.speed
                    indices = np.arange(0, len(data), step)
                    data = data[indices.astype(int)]
                else:
                    # For slower speech, we'd need to interpolate
                    # But we'll just use the original speed to avoid complexity
                    pass
            
            # Use the simplest possible playback method
            sd.play(data, fs)
            
            # Check for interruption while audio is playing
            while sd.get_stream().active:
                if self.stop_event.is_set():
                    logger.info("Speech interrupted")
                    sd.stop()
                    break
                time.sleep(0.1)  # Check for interruption every 100ms
            
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

class SpeechRecognizer:
    def __init__(self, wake_word="hey maxwell", interrupt_word="stop talking", offline_mode=False):
        self.wake_word = wake_word.lower()
        self.interrupt_word = interrupt_word.lower()
        self.offline_mode = offline_mode
        self.recognizer = sr.Recognizer()
        
        # Adjust energy threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 3000  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        
        # For background listening
        self.background_listening = False
        self.interrupt_detected = threading.Event()
        self.background_thread = None
        
        # Get list of available microphones
        self.list_microphones()
        
        # Initialize Vosk for offline recognition if needed
        if offline_mode:
            try:
                from vosk import Model, KaldiRecognizer
                model_path = os.path.join(os.path.expanduser("~"), ".maxwell", "vosk_model")
                if not os.path.exists(model_path):
                    raise FileNotFoundError("Vosk model not found. Please download it first.")
                self.vosk_model = Model(model_path)
                self.offline_recognizer = KaldiRecognizer(self.vosk_model, 16000)
            except Exception as e:
                logger.error(f"Failed to initialize offline recognition: {e}")
                logger.info("Falling back to online recognition")
                self.offline_mode = False
    
    def list_microphones(self):
        """List available microphones for debugging purposes"""
        try:
            mics = sr.Microphone.list_microphone_names()
            # Remove duplicates while preserving order
            unique_mics = []
            seen = set()
            for mic in mics:
                if mic not in seen:
                    seen.add(mic)
                    unique_mics.append(mic)
            
            logger.info("Available microphones:")
            for i, mic in enumerate(unique_mics):
                logger.info(f"  {i}: {mic}")
            
            # Try to identify the default microphone
            default_mic = None
            try:
                # Try to find a good microphone by name
                jlab_mic_index = next((i for i, mic in enumerate(unique_mics) if "JLAB TALK PRO" in mic), None)
                if jlab_mic_index is not None:
                    logger.info(f"Using JLAB microphone: {jlab_mic_index}: {unique_mics[jlab_mic_index]}")
                    self.microphone_index = jlab_mic_index
                    return
                    
                # If no specific mic found, try to use the default
                with sr.Microphone() as source:
                    default_mic = source.device_index
                    self.microphone_index = default_mic
            except Exception as e:
                logger.error(f"Error determining default microphone: {e}")
            
            if default_mic is not None:
                logger.info(f"Default microphone: {default_mic}: {mics[default_mic]}")
            else:
                logger.warning("Could not determine default microphone, using first available")
                self.microphone_index = 0  # Use the first microphone as fallback
            
        except Exception as e:
            logger.error(f"Error listing microphones: {e}")
            self.microphone_index = None  # No specific microphone
        
    def listen(self):
        try:
            # Use the specific microphone if available
            if hasattr(self, 'microphone_index') and self.microphone_index is not None:
                mic = sr.Microphone(device_index=self.microphone_index)
            else:
                mic = sr.Microphone()
            
            with mic as source:
                logger.info("Listening...")
                # Adjust for ambient noise before each listen
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                except sr.WaitTimeoutError:
                    logger.info("Listen timeout - no speech detected")
                    return None
            
            try:
                if self.offline_mode:
                    # Use Vosk for offline recognition
                    raw_data = audio.get_raw_data()
                    self.offline_recognizer.AcceptWaveform(raw_data)
                    result = self.offline_recognizer.Result()
                    import json
                    text = json.loads(result)["text"]
                else:
                    # Use Google Speech Recognition
                    text = self.recognizer.recognize_google(audio)
                    
                logger.info(f"Recognized: {text}")
                return text
            except sr.UnknownValueError:
                logger.info("Could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Recognition error: {e}")
                return None
        except Exception as e:
            logger.error(f"Error in listen: {e}")
            return None
        
    def detect_wake_word(self):
        try:
            # Use the specific microphone if available
            if hasattr(self, 'microphone_index') and self.microphone_index is not None:
                mic = sr.Microphone(device_index=self.microphone_index)
            else:
                mic = sr.Microphone()
            
            with mic as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                except sr.WaitTimeoutError:
                    return False
            
            try:
                if self.offline_mode:
                    # Use Vosk for offline recognition
                    raw_data = audio.get_raw_data()
                    self.offline_recognizer.AcceptWaveform(raw_data)
                    result = self.offline_recognizer.Result()
                    import json
                    text = json.loads(result)["text"]
                else:
                    # Use Google Speech Recognition
                    text = self.recognizer.recognize_google(audio)
                    
                logger.debug(f"Wake word check: {text}")
                return self.wake_word in text.lower()
            except:
                return False
        except Exception as e:
            logger.error(f"Error in detect_wake_word: {e}")
            return False
    
    def start_background_listening(self):
        """Start listening for the interrupt word in the background."""
        if self.background_thread and self.background_thread.is_alive():
            return  # Already listening
            
        self.background_listening = True
        self.interrupt_detected.clear()
        
        def background_listen():
            logger.info("Background listening started")
            while self.background_listening:
                try:
                    # Use the specific microphone if available
                    if hasattr(self, 'microphone_index') and self.microphone_index is not None:
                        mic = sr.Microphone(device_index=self.microphone_index)
                    else:
                        mic = sr.Microphone()
                        
                    with mic as source:
                        # Quick adjustment for ambient noise
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                        try:
                            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=2)
                            
                            # Try to recognize
                            try:
                                if self.offline_mode:
                                    # Use Vosk for offline recognition
                                    raw_data = audio.get_raw_data()
                                    self.offline_recognizer.AcceptWaveform(raw_data)
                                    result = self.offline_recognizer.Result()
                                    import json
                                    text = json.loads(result)["text"]
                                else:
                                    # Use Google Speech Recognition
                                    text = self.recognizer.recognize_google(audio)
                                    
                                logger.debug(f"Background heard: {text}")
                                
                                # Check for interrupt word
                                if self.interrupt_word in text.lower():
                                    logger.info(f"Interrupt word detected: {text}")
                                    self.interrupt_detected.set()
                                    break
                            except:
                                # Ignore recognition errors in background
                                pass
                        except sr.WaitTimeoutError:
                            # Timeout is normal, just continue
                            pass
                except Exception as e:
                    logger.error(f"Error in background listening: {e}")
                    time.sleep(0.5)  # Prevent tight loop on error
            
            logger.info("Background listening stopped")
        
        self.background_thread = threading.Thread(target=background_listen, daemon=True)
        self.background_thread.start()
    
    def stop_background_listening(self):
        """Stop the background listening thread."""
        self.background_listening = False
        if self.background_thread:
            self.background_thread.join(timeout=2)
            self.background_thread = None
    
    def is_interrupt_detected(self):
        """Check if an interrupt was detected."""
        return self.interrupt_detected.is_set()
    
    def reset_interrupt(self):
        """Reset the interrupt flag."""
        self.interrupt_detected.clear()
            
    def detect_interrupt(self):
        """Legacy method - checks for interrupt directly (not in background)."""
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(source, phrase_time_limit=2)
                text = self.recognizer.recognize_google(audio)
                logger.debug(f"Interrupt check: {text}")
                return self.interrupt_word in text.lower()
            except:
                return False
                
    def cleanup(self):
        """Clean up resources."""
        self.stop_background_listening() 