#!/usr/bin/env python3

# Standard library imports
import os
import sys
import time
import signal
import threading
import queue
import json
import re
import argparse
import logging
from datetime import datetime

# Third-party imports
import numpy as np
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr
import requests
import webrtcvad
from kokoro_onnx import Kokoro
# Import the ollama Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False

# Optional imports for offline speech recognition
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

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
stop_listening = False
is_speaking = False
conversation_history = []
audio_queue = queue.Queue()
last_speaking_time = 0  # Track when the assistant last finished speaking
speaking_cooldown = 1.0  # Cooldown period in seconds after speaking
waiting_for_user_input = False  # Flag to indicate when the assistant is waiting for user input
max_silence_before_prompt = 10.0  # Maximum silence time before prompting user again
wake_word = "hey maxwell"  # Wake word to activate the assistant
wake_word_active = False  # Flag to indicate if the wake word has been detected

class OllamaClient:
    """Client for interacting with Ollama API using the official Python client."""
    
    def __init__(self, model="dolphin-llama3:8b-v2.9-q4_0", base_url="http://localhost:11434"):
        """Initialize Ollama client.
        
        Args:
            model: The model to use for generation
            base_url: The base URL for the Ollama API
        """
        self.model = model
        self.base_url = base_url
        
        # Set Ollama host if specified
        if base_url != "http://localhost:11434":
            # Extract host and port from base_url
            import re
            match = re.match(r'https?://([^:/]+)(?::(\d+))?', base_url)
            if match:
                host = match.group(1)
                port = match.group(2) or "11434"
                os.environ["OLLAMA_HOST"] = f"{host}:{port}"
        
        # Check if the official client is available
        if not OLLAMA_CLIENT_AVAILABLE:
            logger.warning("Official Ollama Python client not found. Using REST API fallback.")
            self._use_official_client = False
            self.api_url = f"{base_url}/api/generate"
            # Test connection and ensure model is available using REST API
            self._ensure_ollama_running_with_model_rest()
        else:
            self._use_official_client = True
            # Test connection and ensure model is available using Python client
            self._ensure_ollama_running_with_model_client()
    
    def _ensure_ollama_running_with_model_client(self):
        """Ensure Ollama is running and the specified model is available using the Python client."""
        try:
            # List available models
            models_response = ollama.list()
            
            # Debug log to see the actual structure
            logger.debug(f"Ollama models response: {models_response}")
            
            # Extract model names safely - try different approaches
            available_models = []
            
            # Check if models_response is a list or has a models attribute that's a list
            if isinstance(models_response, list):
                # Direct list of models
                for model_info in models_response:
                    if hasattr(model_info, 'model'):
                        available_models.append(model_info.model)
                    elif isinstance(model_info, dict) and 'model' in model_info:
                        available_models.append(model_info['model'])
            elif hasattr(models_response, 'models') and isinstance(models_response.models, list):
                # Object with models attribute
                for model_info in models_response.models:
                    if hasattr(model_info, 'model'):
                        available_models.append(model_info.model)
                    elif isinstance(model_info, dict) and 'model' in model_info:
                        available_models.append(model_info['model'])
            elif isinstance(models_response, dict) and 'models' in models_response:
                # Dictionary with models key
                for model_info in models_response['models']:
                    if isinstance(model_info, dict) and 'model' in model_info:
                        available_models.append(model_info['model'])
            
            # If we still don't have models, try a different approach
            if not available_models:
                # Run the command line tool and parse the output
                logger.debug("Trying to get models from command line")
                try:
                    import subprocess
                    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                    if result.returncode == 0:
                        # Parse the output to extract model names
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:  # Skip header line
                            for line in lines[1:]:
                                if line.strip():
                                    # First column is the model name
                                    model_name = line.strip().split()[0]
                                    available_models.append(model_name)
                except Exception as e:
                    logger.debug(f"Error getting models from command line: {e}")
            
            logger.debug(f"Available models: {available_models}")
            
            # Check if the specified model is available
            if self.model not in available_models:
                logger.info(f"Model '{self.model}' not found in available models. Attempting to pull it...")
                try:
                    # Pull the model
                    logger.info(f"Pulling model '{self.model}'. This may take some time...")
                    ollama.pull(self.model)
                    logger.info(f"Successfully pulled model '{self.model}'")
                except Exception as e:
                    logger.error(f"Failed to pull model '{self.model}': {e}")
                
                # Check again after pulling
                try:
                    # Try the command line approach directly
                    import subprocess
                    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                    available_models = []
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:  # Skip header line
                            for line in lines[1:]:
                                if line.strip():
                                    model_name = line.strip().split()[0]
                                    available_models.append(model_name)
                except Exception as e:
                    logger.debug(f"Error getting models from command line after pull: {e}")
                
                logger.debug(f"Available models after pull: {available_models}")
                
                if self.model not in available_models:
                    logger.warning(f"Model '{self.model}' still not available after pull attempt.")
                    if available_models:
                        self.model = available_models[0]
                        logger.info(f"Using '{self.model}' instead")
                    else:
                        logger.error("No models available in Ollama")
                        sys.exit(1)
            
            logger.info(f"Connected to Ollama. Using model: {self.model}")
        except Exception as e:
            logger.error(f"Could not connect to Ollama: {e}")
            logger.info("Please start Ollama and try again.")
            sys.exit(1)
    
    def _ensure_ollama_running_with_model_rest(self):
        """Ensure Ollama is running and the specified model is available using REST API."""
        try:
            # Check if Ollama is running and get available models
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                response_data = response.json()
                logger.debug(f"Ollama REST API response: {response_data}")
                
                # Extract model names safely
                available_models = []
                if isinstance(response_data, dict) and "models" in response_data:
                    for model_info in response_data["models"]:
                        if isinstance(model_info, dict):
                            model_name = model_info.get("name") or model_info.get("model")
                            if model_name:
                                available_models.append(model_name)
                
                logger.debug(f"Available models (REST API): {available_models}")
                
                # Check if the specified model is available
                if self.model not in available_models:
                    logger.info(f"Model '{self.model}' not found in available models. Attempting to pull it...")
                    self._pull_model_rest()
                    
                    # Check again after pulling
                    response = requests.get(f"{self.base_url}/api/tags")
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        # Extract model names safely again
                        available_models = []
                        if isinstance(response_data, dict) and "models" in response_data:
                            for model_info in response_data["models"]:
                                if isinstance(model_info, dict):
                                    model_name = model_info.get("name") or model_info.get("model")
                                    if model_name:
                                        available_models.append(model_name)
                        
                        if self.model not in available_models:
                            logger.warning(f"Model '{self.model}' still not available after pull attempt.")
                            if available_models:
                                self.model = available_models[0]
                                logger.info(f"Using '{self.model}' instead")
                            else:
                                logger.error("No models available in Ollama")
                                sys.exit(1)
                
                logger.info(f"Connected to Ollama. Using model: {self.model}")
            else:
                logger.error(f"Failed to connect to Ollama API: {response.status_code}")
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama at {self.base_url}. Is Ollama running?")
            logger.info("Please start Ollama and try again.")
            sys.exit(1)
    
    def _pull_model_rest(self):
        """Pull the specified model from Ollama using REST API."""
        try:
            logger.info(f"Pulling model '{self.model}'. This may take some time...")
            
            # Use the Ollama API to pull the model
            pull_url = f"{self.base_url}/api/pull"
            payload = {"name": self.model}
            
            response = requests.post(pull_url, json=payload, stream=True)
            
            if response.status_code == 200:
                # Process the streaming response to show progress
                for line in response.iter_lines():
                    if line:
                        try:
                            progress_data = json.loads(line.decode('utf-8'))
                            if 'status' in progress_data:
                                if progress_data.get('completed'):
                                    logger.info(f"Model pull completed: {progress_data.get('status')}")
                                else:
                                    logger.info(f"Pulling model: {progress_data.get('status')}")
                        except json.JSONDecodeError:
                            pass
                
                logger.info(f"Successfully pulled model '{self.model}'")
                return True
            else:
                logger.error(f"Failed to pull model: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Exception when pulling model: {e}")
            return False
    
    def generate_response(self, prompt, system_prompt=None, max_tokens=500):
        """Generate a response from Ollama.
        
        Args:
            prompt: The user's input
            system_prompt: Optional system prompt to guide the model
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated response text
        """
        if self._use_official_client:
            return self._generate_response_client(prompt, system_prompt, max_tokens)
        else:
            return self._generate_response_rest(prompt, system_prompt, max_tokens)
    
    def _generate_response_client(self, prompt, system_prompt=None, max_tokens=500):
        """Generate a response using the Python client."""
        try:
            # Prepare the messages
            messages = [{"role": "user", "content": prompt}]
            
            # Add system message if provided
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            
            # Set options
            options = {"num_predict": max_tokens}
            
            # Generate response
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options=options
            )
            
            # Extract the response content
            return response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Exception when calling Ollama client: {e}")
            return "I'm sorry, I'm having trouble connecting to my thinking module."
    
    def _generate_response_rest(self, prompt, system_prompt=None, max_tokens=500):
        """Generate a response using the REST API."""
        # Prepare the request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
                return "I'm sorry, I encountered an error processing your request."
        except Exception as e:
            logger.error(f"Exception when calling Ollama API: {e}")
            return "I'm sorry, I'm having trouble connecting to my thinking module."

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
            
            logger.info(f"Vosk speech recognizer initialized with model: {model_path}")
        except Exception as e:
            logger.error(f"Error initializing Vosk: {e}")
            sys.exit(1)
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if not self.is_listening:
            return
            
        # Add audio data to buffer
        self.buffer.put(bytes(indata))
    
    def listen_for_speech(self, timeout=10):
        """Listen for speech and return the recognized text.
        
        Args:
            timeout: Maximum time to listen in seconds
            
        Returns:
            Recognized text or None if no speech detected
        """
        global is_speaking, last_speaking_time, speaking_cooldown
        
        # Don't listen while speaking
        if is_speaking:
            return None
            
        # Don't listen during cooldown period after speaking
        time_since_speaking = time.time() - last_speaking_time
        if time_since_speaking < speaking_cooldown:
            logger.debug(f"In cooldown period ({time_since_speaking:.2f}s < {speaking_cooldown}s), skipping listening")
            return None
            
        self.is_listening = True
        self.recognizer.Reset()  # Reset the recognizer
        
        # Print and log a clear message that we're listening
        listening_message = "LISTENING FOR SPEECH (VOSK)..."
        print("\n" + "*"*50)
        print(listening_message)
        print("*"*50)
        logger.info(listening_message)
        
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
                max_silence_chunks = 15  # Increased from 5 to allow more pauses in speech
                
                # Listen until timeout or silence after speech
                while time.time() - start_time < timeout:
                    # Check if we should stop
                    if stop_listening or is_speaking:
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
                            
                        has_speech = True
                    elif has_speech:
                        silent_chunks += 1
                        
                    # Process audio with recognizer
                    if self.recognizer.AcceptWaveform(audio_chunk):
                        result = json.loads(self.recognizer.Result())
                        if result.get("text", ""):
                            logger.debug(f"Intermediate result: {result['text']}")
                    
                    # End if we've had enough silence after speech
                    if has_speech and silent_chunks > max_silence_chunks:
                        # But only if we've had a reasonable amount of speech
                        if voiced_frames > 10:  # Ensure we have enough speech frames
                            break
                
                # Get final result
                result = json.loads(self.recognizer.FinalResult())
                text = result.get("text", "").strip()
                
                # Check if speech duration was long enough
                speech_duration = 0
                if has_speech and speech_start_time is not None:
                    speech_duration = time.time() - speech_start_time
                    
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
                    heard_message = f"HEARD (VOSK): \"{text}\""
                    print("\n" + "*"*50)
                    print(heard_message)
                    print("*"*50 + "\n")
                    logger.info(heard_message)
                    
                    return text
                else:
                    logger.debug("No speech detected")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in Vosk speech recognition: {e}")
            return None
        finally:
            self.is_listening = False
    
    def _is_speech(self, audio_chunk):
        """Check if audio chunk contains speech using VAD.
        
        Args:
            audio_chunk: Audio data to check
            
        Returns:
            True if speech is detected, False otherwise
        """
        # Ensure we have the right number of samples for VAD
        # VAD expects 10, 20, or 30ms frames at 8, 16, or 32kHz
        frame_duration_ms = 30
        samples_per_frame = int(self.sample_rate * frame_duration_ms / 1000)
        
        if len(audio_chunk) >= samples_per_frame * 2:  # Need at least 2 bytes per sample
            # Extract first frame for VAD
            frame = audio_chunk[:samples_per_frame * 2]
            
            # Additional energy-based filtering
            # Convert bytes to int16 array
            audio_array = np.frombuffer(frame, dtype=np.int16)
            
            # Calculate energy
            energy = np.mean(np.abs(audio_array))
            
            # Set a minimum energy threshold (adjust as needed)
            min_energy_threshold = self.energy_threshold
            
            # Check both VAD and energy threshold
            is_speech_vad = self.vad.is_speech(frame, self.sample_rate)
            is_speech_energy = energy > min_energy_threshold
            
            # Only consider it speech if both conditions are met
            is_speech = is_speech_vad and is_speech_energy
            
            if is_speech_vad and not is_speech_energy:
                logger.debug(f"VAD detected speech but energy too low: {energy:.2f} < {min_energy_threshold}")
            
            return is_speech
        
        return False

class SpeechRecognizer:
    """Speech recognizer using Google Speech Recognition or Vosk."""
    
    def __init__(self, vad_aggressiveness=3, sample_rate=16000, language="en-US", 
                 use_offline=False, vosk_model_path="vosk-model-small-en-us", energy_threshold=300):
        """Initialize speech recognizer.
        
        Args:
            vad_aggressiveness: VAD aggressiveness level (0-3)
            sample_rate: Audio sample rate
            language: Language for speech recognition
            use_offline: Whether to use offline recognition
            vosk_model_path: Path to Vosk model directory
            energy_threshold: Minimum energy threshold for speech detection
        """
        self.recognizer = sr.Recognizer()
        self.language = language
        self.use_offline = use_offline
        self.speech_timeout = 5  # Maximum speech duration in seconds
        self.energy_threshold = energy_threshold
        
        # Initialize offline recognizer if requested
        self.vosk_recognizer = None
        if use_offline:
            try:
                if not VOSK_AVAILABLE:
                    logger.warning("Vosk not available, falling back to online recognition")
                    self.use_offline = False
                else:
                    self.vosk_recognizer = VoskSpeechRecognizer(
                        model_path=vosk_model_path,
                        sample_rate=sample_rate,
                        energy_threshold=energy_threshold
                    )
            except Exception as e:
                logger.error(f"Error initializing offline recognizer: {e}")
                logger.warning("Falling back to online recognition")
                self.use_offline = False
        
        logger.info(f"Speech recognizer initialized with language: {language}, offline: {use_offline}")
        
    def listen_for_speech(self, source=None, timeout=None):
        """Listen for speech and return the recognized text.
        
        Args:
            source: Audio source (for online recognition)
            timeout: Maximum time to listen in seconds
            
        Returns:
            Recognized text or None if no speech detected
        """
        global is_speaking, last_speaking_time, speaking_cooldown
        
        # Don't listen while speaking
        if is_speaking:
            return None
            
        # Don't listen during cooldown period after speaking
        time_since_speaking = time.time() - last_speaking_time
        if time_since_speaking < speaking_cooldown:
            logger.debug(f"In cooldown period ({time_since_speaking:.2f}s < {speaking_cooldown}s), skipping listening")
            return None
        
        # Print and log a clear message that we're listening
        listening_message = "LISTENING FOR SPEECH..."
        print("\n" + "*"*50)
        print(listening_message)
        print("*"*50)
        logger.info(listening_message)
        
        try:
            if self.use_offline:
                # Use offline recognition
                if self.vosk_recognizer is None:
                    logger.error("Offline recognition requested but Vosk recognizer not initialized")
                    return None
                    
                return self.vosk_recognizer.listen_for_speech(timeout=timeout or 10.0)  # Longer timeout
            else:
                # Use online recognition
                if source is None:
                    logger.error("Source is required for online recognition")
                    return None
                
                # Adjust for ambient noise to improve recognition
                logger.debug("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Set dynamic energy threshold for better noise filtering
                self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, self.energy_threshold)
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.dynamic_energy_adjustment_damping = 0.15
                self.recognizer.dynamic_energy_ratio = 1.5
                
                # Increase phrase timeout and pause threshold for longer sentences
                self.recognizer.pause_threshold = 1.0  # Longer pause threshold (default is 0.8)
                self.recognizer.non_speaking_duration = 1.0  # Longer non-speaking duration
                
                # Listen for audio with phrase timeout
                logger.debug("Listening for speech with adjusted parameters...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout or 10.0,  # Longer timeout
                    phrase_time_limit=10.0,  # Longer phrase time limit (was 5.0)
                    snowboy_configuration=None
                )
                
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
                    heard_message = f"HEARD: \"{text}\""
                    print("\n" + "*"*50)
                    print(heard_message)
                    print("*"*50 + "\n")
                    logger.info(heard_message)
                    
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

class KokoroTTS:
    """Text-to-speech using Kokoro."""
    
    def __init__(self, model_path="kokoro-v1.0.onnx", voices_path="voices-v1.0.bin", 
                 voice="bm_lewis", language="en-us", speed=1.0, skip_init=False):
        """Initialize Kokoro TTS.
        
        Args:
            model_path: Path to Kokoro model file
            voices_path: Path to voices file
            voice: Voice to use
            language: Language to use
            speed: Speech speed
            skip_init: Skip initialization (for testing)
        """
        if skip_init:
            logger.warning("Skipping TTS initialization (for testing only)")
            self.kokoro = None
            self.voice = voice
            self.language = language
            self.speed = speed
            return
            
        # Check if model files exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            logger.info("Download the model with: wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx")
            logger.info("Or run with --no-tts to skip TTS initialization for testing")
            sys.exit(1)
            
        if not os.path.exists(voices_path):
            logger.error(f"Voices file not found: {voices_path}")
            logger.info("Download the voices with: wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin")
            logger.info("Or run with --no-tts to skip TTS initialization for testing")
            sys.exit(1)
        
        try:
            self.kokoro = Kokoro(model_path, voices_path)
            
            # Check if the Kokoro object has the expected methods
            if not hasattr(self.kokoro, 'get_languages'):
                logger.warning("Kokoro object doesn't have get_languages method. Using default language.")
                self._has_languages = False
                self.language = language
            else:
                self._has_languages = True
                self.language = self._validate_language(language)
                
            if not hasattr(self.kokoro, 'get_voices'):
                logger.warning("Kokoro object doesn't have get_voices method. Using default voice.")
                self._has_voices = False
                self.voice = voice
            else:
                self._has_voices = True
                self.voice = self._validate_voice(voice)
                
            self.speed = speed
            
            logger.info(f"Kokoro TTS initialized with voice: {voice}, language: {language}, speed: {speed}")
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            logger.info("Run with --no-tts to skip TTS initialization for testing")
            sys.exit(1)
    
    def _validate_voice(self, voice):
        """Validate if the voice is supported."""
        if self.kokoro is None or not self._has_voices:
            # Skip validation if TTS is disabled or no voices method
            return voice
            
        supported_voices = set(self.kokoro.get_voices())
        
        # Parse comma separated voices for blend
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
                logger.error("Voice blending needs two comma separated voices")
                sys.exit(1)
                 
            # Validate voice
            for v in voices:
                if v not in supported_voices:
                    logger.error(f"Unsupported voice: {v}")
                    logger.info(f"Supported voices: {', '.join(sorted(supported_voices))}")
                    sys.exit(1)
             
            # Normalize weights to sum to 100
            total = sum(weights)
            if total != 100:
                weights = [w * (100/total) for w in weights]
            
            # Create voice blend style
            style1 = self.kokoro.get_voice_style(voices[0])
            style2 = self.kokoro.get_voice_style(voices[1])
            blend = np.add(style1 * (weights[0]/100), style2 * (weights[1]/100))
            return blend
        
        # Single voice validation
        if voice not in supported_voices:
            logger.error(f"Unsupported voice: {voice}")
            logger.info(f"Supported voices: {', '.join(sorted(supported_voices))}")
            sys.exit(1)
        
        return voice
    
    def _validate_language(self, lang):
        """Validate if the language is supported."""
        if self.kokoro is None or not self._has_languages:
            # Skip validation if TTS is disabled or no languages method
            return lang
            
        supported_languages = set(self.kokoro.get_languages())
        if lang not in supported_languages:
            logger.error(f"Unsupported language: {lang}")
            logger.info(f"Supported languages: {', '.join(sorted(supported_languages))}")
            sys.exit(1)
        return lang
    
    def list_available_voices(self):
        """List all available voices."""
        if self.kokoro is None:
            logger.warning("TTS initialization was skipped, cannot list voices")
            return []
            
        if not self._has_voices:
            logger.warning("Kokoro object doesn't have get_voices method")
            return []
            
        voices = sorted(self.kokoro.get_voices())
        print("\nAvailable voices:")
        for idx, voice in enumerate(voices, 1):
            print(f"{idx}. {voice}")
        return voices
    
    def list_available_languages(self):
        """List all available languages."""
        if self.kokoro is None:
            logger.warning("TTS initialization was skipped, cannot list languages")
            return []
            
        if not self._has_languages:
            logger.warning("Kokoro object doesn't have get_languages method")
            return []
            
        languages = sorted(self.kokoro.get_languages())
        print("\nAvailable languages:")
        for idx, lang in enumerate(languages, 1):
            print(f"{idx}. {lang}")
        return languages
    
    def speak(self, text):
        """Convert text to speech and play it.
        
        Args:
            text: Text to speak
        """
        global is_speaking, last_speaking_time, waiting_for_user_input
        
        if not text or not text.strip():
            return
            
        # Skip if TTS is not initialized
        if self.kokoro is None:
            logger.info(f"Would speak (TTS disabled): {text}")
            last_speaking_time = time.time()  # Update last speaking time even when TTS is disabled
            waiting_for_user_input = True  # Ready for user input after "speaking"
            return
        
        # Set speaking flag
        is_speaking = True
        waiting_for_user_input = False  # Not waiting for input while speaking
        
        try:
            # Process text in chunks for better handling
            chunks = self._chunk_text(text)
            
            for chunk in chunks:
                # Check if we should stop speaking
                if stop_listening:
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
                        sd.play(samples, sample_rate)
                        
                        # Wait for audio to finish, but check for interrupts
                        start_time = time.time()
                        while sd.get_stream().active and time.time() - start_time < len(samples) / sample_rate + 0.5:
                            # Small sleep to reduce CPU usage
                            time.sleep(0.1)
                            
                            # Check for keyboard interrupt
                            if stop_listening:
                                logger.info("Interrupting speech")
                                sd.stop()
                                break
                    except KeyboardInterrupt:
                        # Stop audio on keyboard interrupt
                        logger.info("Speech interrupted by user")
                        sd.stop()
                        break
                except TypeError as e:
                    # If we get a TypeError, it might be because the create method has different parameters
                    logger.warning(f"Error with TTS parameters: {e}. Trying with default parameters.")
                    try:
                        # Try with just the text
                        samples, sample_rate = self.kokoro.create(chunk)
                        
                        # Play the audio with a way to interrupt
                        try:
                            sd.play(samples, sample_rate)
                            
                            # Wait for audio to finish, but check for interrupts
                            start_time = time.time()
                            while sd.get_stream().active and time.time() - start_time < len(samples) / sample_rate + 0.5:
                                # Small sleep to reduce CPU usage
                                time.sleep(0.1)
                                
                                # Check for keyboard interrupt
                                if stop_listening:
                                    logger.info("Interrupting speech")
                                    sd.stop()
                                    break
                        except KeyboardInterrupt:
                            # Stop audio on keyboard interrupt
                            logger.info("Speech interrupted by user")
                            sd.stop()
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
    
    def _chunk_text(self, text, max_chunk_size=1000):
        """Split text into chunks at sentence boundaries."""
        sentences = text.replace('\n', ' ').split('.')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue  # Skip empty sentences
            
            sentence = sentence.strip() + '.'
            sentence_size = len(sentence)
            
            # Start new chunk if current one would be too large
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class ConversationalAssistant:
    """Main class for the conversational assistant."""
    
    def __init__(self, ollama_model="dolphin-llama3:8b-v2.9-q4_0", tts_voice="bm_lewis", 
                 language="en-us", speed=1.0, vad_aggressiveness=3,
                 use_offline_recognition=False, vosk_model_path="vosk-model-small-en-us",
                 skip_tts=False, skip_speech=False, energy_threshold=300, 
                 use_wake_word=True, custom_wake_word=None, listen_timeout=10.0):
        """Initialize the conversational assistant.
        
        Args:
            ollama_model: Ollama model to use
            tts_voice: Voice for text-to-speech
            language: Language for speech recognition and TTS
            speed: Speech speed
            vad_aggressiveness: VAD aggressiveness level
            use_offline_recognition: Whether to use offline recognition
            vosk_model_path: Path to Vosk model directory
            skip_tts: Whether to skip text-to-speech
            skip_speech: Whether to skip speech recognition
            energy_threshold: Minimum energy threshold for speech detection
            use_wake_word: Whether to use wake word activation
            custom_wake_word: Custom wake word to use (default: "hey maxwell")
            listen_timeout: Timeout in seconds for speech recognition
        """
        # Initialize Ollama client
        self.ollama = OllamaClient(model=ollama_model)
        
        # Initialize TTS only if not skipping
        self.skip_tts = skip_tts
        if not skip_tts:
            self.tts = KokoroTTS(voice=tts_voice, language=language, speed=speed)
        else:
            # Create a dummy TTS object that does nothing
            logger.info("Skipping TTS initialization")
            self.tts = None
        
        # Initialize speech recognizer only if not skipping
        self.skip_speech = skip_speech
        if not skip_speech:
            self.recognizer = SpeechRecognizer(
                vad_aggressiveness=vad_aggressiveness,
                language=language,
                use_offline=use_offline_recognition,
                vosk_model_path=vosk_model_path,
                energy_threshold=energy_threshold
            )
        else:
            logger.info("Skipping speech recognition initialization")
            self.recognizer = None
        
        # System prompt for the assistant
        self.system_prompt = (
            "You are a helpful, friendly, and concise voice assistant named Maxwell. "
            "Keep your responses conversational but brief (1-3 sentences when possible). "
            "If you don't know something, say so clearly. "
            "Avoid lengthy introductions or unnecessary details."
        )
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Set energy threshold
        self.energy_threshold = energy_threshold
        
        # Wake word settings
        self.use_wake_word = use_wake_word
        global wake_word
        if custom_wake_word:
            wake_word = custom_wake_word.lower()
            
        # Set listen timeout
        self.listen_timeout = listen_timeout
        
        logger.info(f"Conversational assistant initialized{' with wake word: ' + wake_word if use_wake_word else ''}")
        logger.info(f"Listen timeout set to {self.listen_timeout} seconds")
    
    def process_speech_to_response(self, speech_text):
        """Process speech text to generate and speak a response.
        
        Args:
            speech_text: Recognized speech text
        """
        global waiting_for_user_input, wake_word, wake_word_active
        
        if not speech_text:
            waiting_for_user_input = True  # Reset to waiting for input if no speech
            return
        
        # Print and log the captured input in a clearly visible way
        input_message = f"USER INPUT: \"{speech_text}\""
        print("\n" + "="*50)
        print(input_message)
        print("="*50 + "\n")
        logger.info(input_message)
        
        # Check for wake word if wake word is enabled and not already active
        if self.use_wake_word:
            speech_lower = speech_text.lower()
            
            # If wake word is not active, check if the input contains the wake word
            if not wake_word_active:
                if wake_word in speech_lower:
                    # Wake word detected
                    wake_word_active = True
                    logger.info(f"Wake word detected: {wake_word}")
                    
                    # Remove the wake word from the speech text
                    speech_text = speech_text.lower().replace(wake_word, "").strip()
                    
                    # If there's no command after the wake word, just acknowledge and wait for command
                    if not speech_text:
                        response_text = "Yes, Maxwell here. How can I help you?"
                        logger.info(f"Speaking response: {response_text}")
                        
                        # Print and log the response in a clearly visible way
                        response_message = f"ASSISTANT RESPONSE: \"{response_text}\""
                        print("\n" + "-"*50)
                        print(response_message)
                        print("-"*50 + "\n")
                        logger.info(response_message)
                        
                        # Speak the response
                        waiting_for_user_input = False  # Not waiting for input while speaking
                        if not self.skip_tts and self.tts is not None:
                            self.tts.speak(response_text)
                        
                        # After speaking, we're waiting for user input again
                        waiting_for_user_input = True
                        return
                else:
                    # No wake word, ignore the input
                    logger.info(f"Ignoring input without wake word: {speech_text}")
                    waiting_for_user_input = True
                    return
            else:
                # Wake word is already active, check if the user is saying "goodbye" or similar
                if any(phrase in speech_lower for phrase in ["goodbye", "bye", "thank you", "thanks"]):
                    # Deactivate wake word
                    wake_word_active = False
                    logger.info("Wake word deactivated due to goodbye phrase")
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": speech_text})
        
        # Generate response
        logger.info("Generating response...")
        response_text = None
        
        try:
            if hasattr(self.ollama, '_use_official_client') and self.ollama._use_official_client:
                # Use the official client with proper message format
                try:
                    # Check if we need to add system message
                    messages = self.conversation_history[-10:]  # Use last 10 messages
                    
                    # Add system message as the first message if needed
                    if self.system_prompt:
                        # Check if we already have a system message
                        has_system_message = any(msg.get("role") == "system" for msg in messages)
                        if not has_system_message:
                            # Add system message at the beginning
                            messages = [{"role": "system", "content": self.system_prompt}] + messages
                    
                    # Debug log the messages being sent
                    logger.debug(f"Sending messages to Ollama: {json.dumps(messages, indent=2)}")
                    
                    # Generate response
                    response = ollama.chat(
                        model=self.ollama.model,
                        messages=messages
                    )
                    
                    # Debug log the raw response
                    logger.debug(f"Raw Ollama response: {response}")
                    
                    # Extract the response text
                    if isinstance(response, dict):
                        response_text = response.get("message", {}).get("content", "")
                    elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                        response_text = response.message.content
                    else:
                        logger.error(f"Unexpected response format: {type(response)}")
                        response_text = "I'm sorry, I received an unexpected response format."
                except Exception as e:
                    logger.error(f"Error generating response with Ollama client: {e}")
                    logger.error(f"Exception type: {type(e)}")
                    logger.error(f"Exception args: {e.args}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    response_text = "I'm sorry, I encountered an error processing your request."
            else:
                # Fall back to the old method
                prompt = self._format_conversation_for_ollama()
                logger.debug(f"Sending prompt to Ollama: {prompt}")
                response_text = self.ollama.generate_response(
                    prompt=prompt,
                    system_prompt=self.system_prompt
                )
        except Exception as e:
            logger.error(f"Unexpected error in response generation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            response_text = "I'm sorry, I encountered an unexpected error processing your request."
        
        # Fallback if we still don't have a response
        if not response_text:
            logger.error("No response text generated, using fallback")
            response_text = "I'm sorry, I wasn't able to generate a response. Please try again."
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Limit conversation history to last 10 exchanges (20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        # Print and log the response in a clearly visible way
        response_message = f"ASSISTANT RESPONSE: \"{response_text}\""
        print("\n" + "-"*50)
        print(response_message)
        print("-"*50 + "\n")
        logger.info(response_message)
        
        # Speak the response
        logger.info(f"Speaking response: {response_text}")
        waiting_for_user_input = False  # Not waiting for input while speaking
        if not self.skip_tts and self.tts is not None:
            self.tts.speak(response_text)
        
        # After speaking, we're waiting for user input again
        waiting_for_user_input = True
        
        # If wake word is active and the response contains a closing phrase, deactivate wake word
        if self.use_wake_word and wake_word_active:
            if any(phrase in response_text.lower() for phrase in ["goodbye", "bye", "have a good day"]):
                wake_word_active = False
                logger.info("Wake word deactivated due to assistant goodbye phrase")
    
    def _format_conversation_for_ollama(self):
        """Format conversation history for Ollama API.
        
        Returns:
            Formatted conversation string
        """
        formatted = ""
        for msg in self.conversation_history[-10:]:  # Use last 10 messages
            role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
            formatted += f"{role_prefix}{msg['content']}\n\n"
        
        # Add final prompt for the assistant to respond
        formatted += "Assistant: "
        return formatted
    
    def run(self, text_input=None):
        """Run the conversational assistant.
        
        Args:
            text_input: Optional text input to process (for testing)
        """
        global stop_listening, waiting_for_user_input, max_silence_before_prompt, wake_word_active
        
        # Initial greeting
        greeting = "Hello! I'm Maxwell, your voice assistant. Say 'Hey Maxwell' to activate me."
        if not self.use_wake_word:
            greeting = "Hello! I'm Maxwell, your voice assistant. How can I help you today?"
            
        logger.info(f"Speaking greeting: {greeting}")
        if not self.skip_tts and self.tts is not None:
            self.tts.speak(greeting)
        
        # If text input is provided, process it and exit
        if text_input:
            input_message = f"PROCESSING TEXT INPUT: \"{text_input}\""
            print("\n" + "="*50)
            print(input_message)
            print("="*50 + "\n")
            logger.info(input_message)
            
            # If using wake word, check if the text input contains the wake word
            if self.use_wake_word and wake_word not in text_input.lower():
                # Temporarily set wake_word_active to True to process this input
                wake_word_active = True
                
            self.process_speech_to_response(text_input)
            
            # Reset wake_word_active after processing text input
            if self.use_wake_word:
                wake_word_active = False
                
            return
            
        # If speech recognition is disabled, exit
        if self.skip_speech or self.recognizer is None:
            logger.info("Speech recognition is disabled. Use --text-input to process a single text input.")
            return
        
        # Initialize microphone if using online recognition
        mic = None
        if not self.recognizer.use_offline:
            mic = sr.Microphone()
            
            with mic as source:
                # Adjust for ambient noise
                logger.info("Adjusting for ambient noise... Please be quiet for a moment.")
                self.recognizer.recognizer.adjust_for_ambient_noise(source, duration=2)
                logger.info("Ready to listen!")
        else:
            logger.info("Using offline recognition. Ready to listen!")
        
        # Main loop
        waiting_for_user_input = True
        last_user_input_time = time.time()
        
        while not stop_listening:
            try:
                # Only listen when not speaking and we're waiting for user input
                if not is_speaking and waiting_for_user_input:
                    # Check if we've been silent for too long and prompting is enabled
                    time_since_last_input = time.time() - last_user_input_time
                    
                    # Only prompt if wake word is active or we're not using wake word
                    should_prompt = (not self.use_wake_word or wake_word_active) and \
                                   max_silence_before_prompt > 0 and \
                                   time_since_last_input > max_silence_before_prompt
                    
                    if should_prompt:
                        # Prompt the user again
                        prompt_message = "I'm still listening. Is there anything I can help you with?"
                        logger.info(f"Prompting user after {time_since_last_input:.1f}s of silence: {prompt_message}")
                        
                        if not self.skip_tts and self.tts is not None:
                            self.tts.speak(prompt_message)
                            
                        # Reset the timer
                        last_user_input_time = time.time()
                        
                    # Listen for speech with the configured timeout
                    speech_text = None
                    if self.recognizer.use_offline:
                        # Offline recognition doesn't need a source
                        speech_text = self.recognizer.listen_for_speech(timeout=self.listen_timeout)
                    else:
                        # Online recognition needs a source
                        with mic as source:
                            speech_text = self.recognizer.listen_for_speech(source, timeout=self.listen_timeout)
                    
                    if speech_text:
                        # Update the last input time
                        last_user_input_time = time.time()
                        
                        # We're no longer waiting for user input while processing
                        waiting_for_user_input = False
                        
                        # Process in a separate thread to allow interruption
                        processing_thread = threading.Thread(
                            target=self.process_speech_to_response,
                            args=(speech_text,)
                        )
                        processing_thread.daemon = True
                        processing_thread.start()
                        
                        # Wait for processing to complete
                        processing_thread.join()
                        
                        # After processing, we're waiting for user input again
                        waiting_for_user_input = True
                else:
                    # Small sleep when speaking or not waiting for input to reduce CPU usage
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
        
        logger.info("Conversational assistant stopped")

def handle_interrupt(signum, frame):
    """Handle interrupt signal."""
    global stop_listening, is_speaking, waiting_for_user_input
    
    logger.info("Interrupt received, shutting down...")
    
    # Stop any ongoing speech
    if is_speaking:
        logger.info("Stopping ongoing speech")
        sd.stop()
        is_speaking = False
        waiting_for_user_input = True
    
    # Set flag to stop listening
    stop_listening = True

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Conversational AI Assistant with Kokoro TTS and Ollama")
    
    parser.add_argument("--ollama-model", type=str, default="dolphin-llama3:8b-v2.9-q4_0",
                        help="Ollama model to use (default: dolphin-llama3:8b-v2.9-q4_0)")
    parser.add_argument("--voice", type=str, default="bm_lewis",
                        help="Voice for text-to-speech (default: bm_lewis)")
    parser.add_argument("--language", type=str, default="en-us",
                        help="Language for speech recognition and TTS (default: en-us)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed (default: 1.0)")
    parser.add_argument("--vad-level", type=int, choices=[0, 1, 2, 3], default=3,
                        help="VAD aggressiveness level (0-3, default: 3)")
    parser.add_argument("--offline", action="store_true",
                        help="Use offline speech recognition with Vosk")
    parser.add_argument("--vosk-model", type=str, default="vosk-model-small-en-us",
                        help="Path to Vosk model directory (default: vosk-model-small-en-us)")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available voices and exit")
    parser.add_argument("--list-languages", action="store_true",
                        help="List available languages and exit")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech (for testing)")
    parser.add_argument("--no-speech", action="store_true",
                        help="Disable speech recognition (for testing)")
    parser.add_argument("--text-input", type=str,
                        help="Process a single text input and exit (for testing)")
    parser.add_argument("--cooldown", type=float, default=1.0,
                        help="Cooldown period in seconds after speaking (default: 1.0)")
    parser.add_argument("--energy-threshold", type=int, default=300,
                        help="Minimum energy threshold for speech detection (default: 300)")
    parser.add_argument("--max-silence", type=float, default=10.0,
                        help="Maximum silence time in seconds before prompting user again (default: 10.0, 0 to disable)")
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Disable wake word activation (default: enabled)")
    parser.add_argument("--wake-word", type=str, default="hey maxwell",
                        help="Custom wake word to use (default: 'hey maxwell')")
    parser.add_argument("--listen-timeout", type=float, default=10.0,
                        help="Timeout in seconds for speech recognition (default: 10.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set global cooldown period
    global speaking_cooldown, max_silence_before_prompt, wake_word
    speaking_cooldown = args.cooldown
    max_silence_before_prompt = args.max_silence
    
    # Set wake word
    if args.wake_word:
        wake_word = args.wake_word.lower()
    
    logger.info(f"Speaking cooldown set to {speaking_cooldown} seconds")
    logger.info(f"Maximum silence before prompt set to {max_silence_before_prompt} seconds")
    if not args.no_wake_word:
        logger.info(f"Wake word set to '{wake_word}'")
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create TTS instance for listing voices/languages
    if args.list_voices or args.list_languages:
        # Can't list voices or languages if TTS is disabled
        if args.no_tts:
            logger.error("Cannot list voices or languages when TTS is disabled (--no-tts)")
            sys.exit(1)
            
        try:
            tts = KokoroTTS()
            if args.list_voices:
                tts.list_available_voices()
            if args.list_languages:
                tts.list_available_languages()
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    
    # Register signal handler
    signal.signal(signal.SIGINT, handle_interrupt)
    
    # Create and run the assistant
    assistant = ConversationalAssistant(
        ollama_model=args.ollama_model,
        tts_voice=args.voice,
        language=args.language,
        speed=args.speed,
        vad_aggressiveness=args.vad_level,
        use_offline_recognition=args.offline,
        vosk_model_path=args.vosk_model,
        skip_tts=args.no_tts,
        skip_speech=args.no_speech,
        energy_threshold=args.energy_threshold,
        use_wake_word=not args.no_wake_word,
        custom_wake_word=args.wake_word,
        listen_timeout=args.listen_timeout
    )
    
    try:
        assistant.run(args.text_input)
    except Exception as e:
        logger.error(f"Error running assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 