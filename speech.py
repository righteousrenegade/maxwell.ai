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
            
            # The audio_data might be a tuple with (audio, sample_rate)
            # or it might need to be converted to the right format
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                # If it's a tuple of (audio, sample_rate)
                audio, sample_rate = audio_data
            else:
                # Otherwise, assume it's just the audio data
                audio = audio_data
                sample_rate = 24000  # Default sample rate
            
            # Make sure audio is a 1D numpy array
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                # If it's a 2D array, take the first channel
                audio = audio[:, 0] if audio.shape[1] > 0 else audio.flatten()
            
            # Create a temporary file
            temp_file = os.path.join(self.temp_dir, "tts_output.wav")
            
            # Write audio to file
            sf.write(temp_file, audio, sample_rate)
            
            # Play audio
            data, fs = sf.read(temp_file)
            sd.play(data, fs)
            
            # Wait for playback to finish or stop event
            while sd.get_stream().active and not self.stop_event.is_set():
                time.sleep(0.1)
            
            # Stop playback if interrupted
            if self.stop_event.is_set():
                sd.stop()
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            # Print more details about the audio_data for debugging
            if 'audio_data' in locals():
                logger.error(f"Audio data type: {type(audio_data)}")
                if isinstance(audio_data, np.ndarray):
                    logger.error(f"Audio data shape: {audio_data.shape}")
                    logger.error(f"Audio data dtype: {audio_data.dtype}")
            logger.error(f"Failed to speak: {text}")
        
    def stop(self):
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
            
        # Initialize Kokoro
        try:
            kokoro = Kokoro(model_path, voices_path)
            voices = kokoro.get_voices()
            print("Available voices:")
            for voice in sorted(voices):
                print(f"- {voice}")
        except Exception as e:
            print(f"Error listing voices: {e}")

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
            logger.info("Available microphones:")
            for i, mic in enumerate(mics):
                logger.info(f"  {i}: {mic}")
            
            # Try to identify the default microphone
            default_mic = None
            try:
                with sr.Microphone() as source:
                    default_mic = source.device_index
            except:
                pass
                
            if default_mic is not None:
                logger.info(f"Default microphone: {default_mic}: {mics[default_mic]}")
            else:
                logger.warning("Could not determine default microphone")
                
        except Exception as e:
            logger.error(f"Error listing microphones: {e}")
        
    def listen(self):
        with sr.Microphone() as source:
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
            
    def detect_wake_word(self):
        with sr.Microphone() as source:
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
            
    def detect_interrupt(self):
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(source, phrase_time_limit=2)
                text = self.recognizer.recognize_google(audio)
                logger.debug(f"Interrupt check: {text}")
                return self.interrupt_word in text.lower()
            except:
                return False
                
    def cleanup(self):
        pass  # No specific cleanup needed for speech recognition 