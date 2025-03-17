#!/usr/bin/env python3
import argparse
import time
import os
import signal
import sys
import threading
import logging
import speech_recognition as sr
import traceback
from speech import TextToSpeech
from commands import CommandExecutor
from utils import setup_logger, download_models
from config import Config
import random

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

class Maxwell:
    def __init__(self, config):
        self.config = config
        self.running = True
        self.in_conversation = False
        self.speaking = False
        self.mic_name = None  # Store the microphone name
        
        # Initialize components
        logger.info("Initializing Text-to-Speech...")
        self.tts = TextToSpeech(voice=config.voice, speed=config.speed)
        
        # Initialize speech recognition
        logger.info("Initializing Speech Recognition...")
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = config.energy_threshold
        self.recognizer.dynamic_energy_threshold = False
        
        # Initialize ONE microphone instance
        self.mic = None
        if not config.keyboard_mode:
            self.mic, self.mic_name = self._setup_microphone(config.mic_index, config.sample_rate)
            if not self.mic:
                logger.error("Failed to initialize microphone. Falling back to keyboard mode.")
                config.keyboard_mode = True
            else:
                logger.info(f"Using microphone: {self.mic_name}")
        
        # Speech recognition state
        self.wake_word = config.wake_word.lower()
        self.interrupt_word = config.interrupt_word.lower()
        self.interrupt_signal = threading.Event()
        
        logger.info("Initializing Command Executor...")
        self.command_executor = CommandExecutor(self)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"Maxwell initialized with wake word: '{config.wake_word}'")
        
    def _setup_microphone(self, mic_index, sample_rate=None):
        """Set up a single microphone instance with better error handling"""
        try:
            # First, verify the microphone index with PyAudio directly
            import pyaudio
            p = pyaudio.PyAudio()
            
            # Get microphone name
            mic_name = "Default microphone"
            
            if mic_index is not None:
                try:
                    device_info = p.get_device_info_by_index(mic_index)
                    if device_info['maxInputChannels'] > 0:
                        mic_name = device_info['name']
                        logger.info(f"Using microphone {mic_index}: {mic_name}")
                        
                        # Get the default sample rate if not specified
                        if sample_rate is None:
                            sample_rate = int(device_info['defaultSampleRate'])
                            logger.info(f"Using device default sample rate: {sample_rate}Hz")
                    else:
                        logger.error(f"Device {mic_index} has no input channels")
                        logger.info("Falling back to default microphone")
                        mic_index = None
                        sample_rate = None
                except Exception as e:
                    logger.error(f"Error getting device info for index {mic_index}: {e}")
                    logger.info("Falling back to default microphone")
                    mic_index = None
                    sample_rate = None
            
            p.terminate()
            
            # Create microphone instance
            try:
                logger.info(f"Creating microphone with index {mic_index if mic_index is not None else 'default'}")
                
                # If we have a sample rate, use it
                if sample_rate:
                    logger.info(f"Using sample rate {sample_rate}Hz")
                    mic = sr.Microphone(device_index=mic_index, sample_rate=sample_rate)
                else:
                    mic = sr.Microphone(device_index=mic_index)
                
                # Test if microphone works
                logger.info("Testing microphone...")
                with mic as source:
                    # Just open and close to test
                    pass
                logger.info("Microphone test successful")
                return mic, mic_name
            except Exception as e:
                logger.error(f"Error creating microphone: {e}")
                logger.error(traceback.format_exc())
                
                # Try with a different sample rate if the first attempt failed
                if sample_rate:
                    try:
                        # Try with a standard rate instead
                        standard_rates = [16000, 44100, 48000, 22050, 8000]
                        for rate in standard_rates:
                            if rate != sample_rate:  # Skip the rate we already tried
                                logger.info(f"Trying with sample rate {rate}Hz instead")
                                try:
                                    mic = sr.Microphone(device_index=mic_index, sample_rate=rate)
                                    with mic as source:
                                        # Just open and close to test
                                        pass
                                    logger.info(f"Microphone test successful with sample rate {rate}Hz")
                                    return mic, mic_name
                                except Exception as e2:
                                    logger.error(f"Error with sample rate {rate}Hz: {e2}")
                                    continue
                    except Exception as e3:
                        logger.error(f"Error trying alternative sample rates: {e3}")
                
                return None, None
            
        except Exception as e:
            logger.error(f"Error setting up microphone: {e}")
            logger.error(traceback.format_exc())
            return None, None
        
    def signal_handler(self, sig, frame):
        logger.info("Shutdown signal received, exiting...")
        self.running = False
        self.cleanup()
        sys.exit(0)
        
    def cleanup(self):
        logger.info("Cleaning up resources...")
        if hasattr(self, 'tts'):
            self.tts.cleanup()
            
    def speak(self, text):
        logger.info(f"Speaking: {text}")
        self.speaking = True
        
        try:
            # Split text into sentences for more responsive interruption
            sentences = text.split('. ')
            for i, sentence in enumerate(sentences):
                if i > 0:  # Add period back except for the last sentence
                    sentence = sentence + ('.' if i < len(sentences) - 1 else '')
                
                # Check if we should stop before speaking this sentence
                if self.interrupt_signal.is_set():
                    logger.info("Speech interrupted before sentence")
                    break
                    
                # Speak the sentence
                self.tts.speak(sentence)
                
                # Check for interrupt after each sentence
                if self.interrupt_signal.is_set():
                    logger.info("Speech interrupted after sentence")
                    break
        except Exception as e:
            logger.error(f"Error during speech: {e}")
        finally:
            # Always mark speaking as False when done
            self.speaking = False
            # Clear the interrupt signal
            self.interrupt_signal.clear()
            
    def detect_wake_word(self):
        """Listen for the wake word with clear feedback"""
        if not self.mic:
            logger.error("No microphone available")
            return False
            
        try:
            # Print a clear visual indicator that we're waiting for wake word
            print("\n🔴 Listening for wake word... Say 'hey maxwell'")
            
            with self.mic as source:
                # Adjust for ambient noise
                logger.info("Adjusting for ambient noise in detect_wake_word...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                logger.info(f"Energy threshold after adjustment: {self.recognizer.energy_threshold}")
                
                # Listen for audio with a longer timeout
                logger.info("Listening for wake word...")
                try:
                    # Use a longer timeout (5 seconds instead of 3)
                    audio = self.recognizer.listen(source, timeout=5)
                    
                    # Visual feedback that audio was detected
                    print("🔵 Processing what you said...")
                    
                    # Save the audio for debugging if debug mode is enabled
                    if hasattr(self.config, 'save_audio') and self.config.save_audio:
                        self._save_audio_for_debug(audio)
                    
                    # Recognize speech
                    try:
                        logger.info("Recognizing wake word audio...")
                        text = self.recognizer.recognize_google(audio)
                        logger.info(f"Heard: {text}")
                        
                        # Check if wake word is in the text
                        text_lower = text.lower()
                        if self.wake_word in text_lower:
                            logger.info(f"Wake word detected: {text}")
                            print("✅ Wake word detected!")
                            return True
                        
                        # Check for interrupt word if we're speaking
                        if self.speaking and self.interrupt_word in text_lower:
                            logger.info(f"Interrupt word detected: {text}")
                            self.interrupt_signal.set()
                            self.tts.stop()
                            print("🛑 Speech interrupted!")
                            
                        print("❌ Wake word not detected. Try again.")
                        return False
                    except sr.UnknownValueError:
                        logger.debug("Could not understand audio in wake word detection")
                        print("❓ Couldn't understand what you said. Try again.")
                        return False
                    except sr.RequestError as e:
                        logger.error(f"Google Speech Recognition service error: {e}")
                        print("⚠️ Speech recognition service error. Try again.")
                        return False
                except sr.WaitTimeoutError:
                    logger.debug("No speech detected in wake word timeout period")
                    # Don't print anything here to avoid cluttering the console
                    return False
        except Exception as e:
            logger.error(f"Error detecting wake word: {e}")
            logger.error(traceback.format_exc())
            print("⚠️ Error with microphone. Please try again.")
            return False
        
    def process_command(self, command):
        if command and command.lower().startswith("execute "):
            cmd = command[8:].strip()
            return self.command_executor.execute_command(cmd)
        return False
        
    def handle_query(self, query):
        if not query:
            self.speak("I didn't catch that. Could you please repeat?")
            return
            
        logger.info(f"Processing query: {query}")
        print(f"🤔 Processing: \"{query}\"")
        
        # Check for conversation end
        if "end conversation" in query.lower():
            self.speak("Ending conversation mode.")
            self.in_conversation = False
            print("🔚 Conversation ended. Say the wake word to start again.")
            return
            
        # Check for commands
        if self.process_command(query):
            return
            
        # Otherwise, send to LLM
        print("💭 Thinking...")
        response = self.command_executor.query_llm(query)
        print(f"🗣️ Maxwell: {response}")
        self.speak(response)
        
        # If not in continuous conversation mode, exit after one interaction
        if not self.config.continuous_conversation:
            logger.info("Exiting conversation mode (continuous mode disabled)")
            self.in_conversation = False
            self.speak("Call me if you need me.")
            print("🔚 Conversation ended. Say the wake word to start again.")
        
    def run(self):
        logger.info("Maxwell is running. Say the wake word to begin.")
        
        # Announce which microphone is being used
        if self.mic and self.mic_name:
            logger.info(f"Using microphone: {self.mic_name}")
            self.speak(f"Hello, Maxwell here. I'm using the {self.mic_name} microphone. Say '{self.wake_word}' to get my attention.")
        else:
            self.speak(f"Hello, Maxwell here. Say '{self.wake_word}' to get my attention.")
        
        # Add a test mode option for immediate conversation
        if self.config.test_mode or self.config.always_listen:
            logger.info("Test mode or always-listen mode enabled. Entering conversation mode immediately.")
            self.speak("I'm listening for commands.")
            self.in_conversation = True
        
        # Create a thread to check for keyboard interrupts if enabled
        if self.config.keyboard_interrupt:
            logger.info("Keyboard interrupt enabled. Press 's' to stop speech.")
            
            def check_keyboard():
                while self.running:
                    try:
                        # Windows-specific keyboard handling
                        if os.name == 'nt':  # Windows
                            import msvcrt
                            if msvcrt.kbhit():  # Check if a key was pressed
                                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                                if key == 's':  # 's' for stop
                                    logger.info("Stop key pressed")
                                    if self.speaking:
                                        self.interrupt_signal.set()
                                        self.tts.stop()
                                        logger.info("Speech interrupted by keyboard")
                    except Exception as e:
                        logger.error(f"Error checking keyboard: {e}")
                    time.sleep(0.1)
            
            keyboard_thread = threading.Thread(target=check_keyboard, daemon=True)
            keyboard_thread.start()
        
        # Keyboard mode - use keyboard input instead of microphone
        if self.config.keyboard_mode:
            logger.info("Keyboard mode enabled. Type your queries instead of speaking.")
            self.speak("Keyboard mode enabled. Type your queries instead of speaking.")
            
            while self.running:
                try:
                    # Use a simple input prompt
                    user_input = input("\n💬 You: ")
                    
                    if user_input.lower() == "exit":
                        logger.info("Exiting keyboard mode.")
                        break
                    
                    # Process the input as if it was spoken
                    self.handle_query(user_input)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt detected. Exiting.")
                    break
                except Exception as e:
                    logger.error(f"Error in keyboard mode: {e}")
            
            return  # Exit the run method
        
        try:
            # Print initial instructions
            print("\n" + "="*50)
            print("🎤 Maxwell Voice Assistant")
            print("="*50)
            print(f"• Say '{self.wake_word}' to get Maxwell's attention")
            print(f"• Say '{self.interrupt_word}' to stop Maxwell from talking")
            print(f"• Say 'end conversation' to exit conversation mode")
            print("="*50 + "\n")
            
            while self.running:
                # Listen for wake word or in conversation mode
                if not self.in_conversation:
                    logger.info(f"Waiting for wake word: '{self.config.wake_word}'")
                    wake_word_detected = self.detect_wake_word()
                    if wake_word_detected:
                        logger.info("Wake word detected!")
                        self._play_listening_sound()  # Play sound to indicate listening
                        self.speak("Yes?")
                        self.in_conversation = True
                    else:
                        time.sleep(0.1)  # Prevent CPU hogging
                        continue
                
                # In conversation mode
                logger.info("In conversation mode, listening for query...")
                self._play_listening_sound()  # Play sound to indicate listening
                query = self.listen()
                
                # Handle the query
                self.handle_query(query)
                
        finally:
            # Make sure we clean up
            self.cleanup()

    def _is_audio_static(self, audio_array):
        """Check if the audio is just static noise"""
        try:
            import numpy as np
            
            # Calculate standard deviation - static usually has low variation
            std_dev = np.std(audio_array)
            
            # Calculate the ratio of unique values to total values
            # Static often has fewer unique values relative to length
            unique_ratio = len(np.unique(audio_array)) / len(audio_array)
            
            # Log the statistics
            logger.info(f"Audio statistics: std_dev={std_dev:.2f}, unique_ratio={unique_ratio:.4f}")
            
            # Static detection criteria
            if std_dev < 100 or unique_ratio < 0.01:
                logger.warning("Audio appears to be static noise")
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking for static: {e}")
            return False

    def listen(self):
        """Listen for speech and return the recognized text with better feedback"""
        if not self.mic:
            logger.error("No microphone available")
            return None
        
        try:
            # Clear visual indicator that we're listening for a command
            print("\n🟢 Listening for your command... (speak now)")
            print(f"⏱️ You have {self.config.listen_timeout} seconds to speak")
            
            with self.mic as source:
                # Adjust for ambient noise
                logger.info("Adjusting for ambient noise in listen()...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Reduced to 0.5s for faster response
                logger.info(f"Energy threshold after adjustment: {self.recognizer.energy_threshold}")
                
                # Listen for audio
                logger.info("Listening for speech in Maxwell.listen()...")
                try:
                    # Use the configurable timeout
                    audio = self.recognizer.listen(source, timeout=self.config.listen_timeout)
                    
                    # Visual feedback that audio was detected
                    print("🔵 Processing your command...")
                    
                    # Save the audio for debugging if save_audio is enabled
                    if hasattr(self.config, 'save_audio') and self.config.save_audio:
                        self._save_audio_for_debug(audio)
                    
                    # Recognize speech
                    try:
                        logger.info("Recognizing speech with Google...")
                        text = self.recognizer.recognize_google(audio)
                        logger.info(f"Recognized: {text}")
                        print(f"🎯 I heard: \"{text}\"")
                        return text
                    except sr.UnknownValueError:
                        logger.info("Google Speech Recognition could not understand audio")
                        print("❓ I couldn't understand what you said.")
                        
                        # Try offline recognition if enabled
                        if hasattr(self.config, 'offline_mode') and self.config.offline_mode:
                            try:
                                logger.info("Trying offline recognition with Vosk...")
                                # Make sure you have the Vosk model downloaded
                                text = self.recognizer.recognize_vosk(audio)
                                logger.info(f"Recognized with Vosk: {text}")
                                print(f"🎯 I heard (offline): \"{text}\"")
                                return text
                            except Exception as vosk_error:
                                logger.error(f"Vosk recognition failed: {vosk_error}")
                        
                        return None
                    except sr.RequestError as e:
                        logger.error(f"Could not request results from Google Speech Recognition service: {e}")
                        print("⚠️ Speech recognition service error.")
                        return None
                except sr.WaitTimeoutError:
                    logger.info("No speech detected in timeout period")
                    print("⏱️ I didn't hear anything. Please try again.")
                    return None
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            logger.error(traceback.format_exc())
            print("⚠️ Error with microphone. Please try again.")
            return None
            
    def _save_audio_for_debug(self, audio):
        """Save the audio data to a file for debugging"""
        try:
            # Create debug directory if it doesn't exist
            debug_dir = os.path.join(os.getcwd(), "debug_audio")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Generate a filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(debug_dir, f"audio_{timestamp}.wav")
            
            # Save the audio data
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            
            logger.info(f"Saved debug audio to {filename}")
            
            # Log audio properties
            sample_rate = audio.sample_rate
            sample_width = audio.sample_width
            duration = len(audio.frame_data) / sample_rate
            logger.info(f"Audio properties: sample_rate={sample_rate}Hz, sample_width={sample_width}bytes, duration={duration:.2f}s")
            
            # Log energy level and check for static
            try:
                import numpy as np
                import wave
                
                with wave.open(filename, 'rb') as wf:
                    # Read the audio data
                    n_frames = wf.getnframes()
                    audio_data = wf.readframes(n_frames)
                    
                    # Convert to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Calculate energy
                    energy = np.mean(np.abs(audio_array))
                    logger.info(f"Audio energy level: {energy:.2f}")
                    
                    # Check if audio is too quiet
                    if energy < 100:
                        logger.warning("Audio energy is very low - microphone may not be picking up sound properly")
                    elif energy < 500:
                        logger.warning("Audio energy is low - speech may be too quiet")
                    
                    # Check if audio is static
                    if self._is_audio_static(audio_array):
                        logger.warning("STATIC DETECTED: The microphone appears to be capturing static noise instead of speech")
                        logger.warning("This could be due to a hardware issue, incorrect microphone selection, or driver problem")
                
            except Exception as e:
                logger.error(f"Error analyzing audio energy: {e}")
            
        except Exception as e:
            logger.error(f"Error saving debug audio: {e}")
            logger.error(traceback.format_exc())
        
    def _play_listening_sound(self):
        """Play a short beep to indicate Maxwell is listening"""
        try:
            import winsound
            import platform
            
            if platform.system() == 'Windows':
                # Windows beep (frequency, duration in ms)
                winsound.Beep(1000, 200)
            elif platform.system() == 'Darwin':  # macOS
                # Use system command for macOS
                os.system('afplay /System/Library/Sounds/Tink.aiff')
            else:  # Linux and others
                # Print a bell character which might make a sound on some terminals
                print('\a', end='', flush=True)
            
        except Exception as e:
            logger.debug(f"Could not play listening sound: {e}")
            # Fallback to just printing a symbol
            print("🔊", end='', flush=True)

def test_microphone_directly(mic_index=None, duration=10):
    """Test a specific microphone by recording and recognizing speech directly"""
    logger.info(f"Testing microphone {'index ' + str(mic_index) if mic_index is not None else 'default'} directly...")
    
    try:
        # Create a recognizer
        r = sr.Recognizer()
        r.energy_threshold = 300  # Low threshold for better sensitivity
        r.dynamic_energy_threshold = False
        
        # Create microphone instance
        mic = sr.Microphone(device_index=mic_index) if mic_index is not None else sr.Microphone()
        
        logger.info(f"Starting direct microphone test for {duration} seconds...")
        logger.info("Please speak into the microphone")
        
        end_time = time.time() + duration
        recognized_texts = []
        
        # Use the microphone
        with mic as source:
            # Adjust for ambient noise
            logger.info("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1.0)
            logger.info(f"Energy threshold after adjustment: {r.energy_threshold}")
            
            # Listen in a loop until the duration is up
            while time.time() < end_time:
                remaining = int(end_time - time.time())
                logger.info(f"Listening... ({remaining}s remaining)")
                
                try:
                    audio = r.listen(source, timeout=3)
                    logger.info("Audio captured, recognizing...")
                    
                    # Save the audio for debugging
                    debug_dir = os.path.join(os.getcwd(), "debug_audio")
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = os.path.join(debug_dir, f"test_audio_{timestamp}.wav")
                    with open(filename, "wb") as f:
                        f.write(audio.get_wav_data())
                    logger.info(f"Saved test audio to {filename}")
                    
                    try:
                        text = r.recognize_google(audio)
                        logger.info(f"Recognized: '{text}'")
                        recognized_texts.append(text)
                    except sr.UnknownValueError:
                        logger.info("Could not understand audio")
                    except sr.RequestError as e:
                        logger.error(f"Google Speech Recognition service error: {e}")
                        
                except sr.WaitTimeoutError:
                    logger.info("No speech detected in timeout period in the funky bunch.")
                except Exception as e:
                    logger.error(f"Error in listening: {e}")
                    logger.error(traceback.format_exc())
        
        # Report results
        logger.info("Microphone test completed")
        logger.info(f"Recognized {len(recognized_texts)} speech segments:")
        for i, text in enumerate(recognized_texts):
            logger.info(f"  {i+1}: '{text}'")
        
        if len(recognized_texts) > 0:
            logger.info("Microphone test PASSED - speech was recognized")
            return True
        else:
            logger.info("Microphone test FAILED - no speech was recognized")
            return False
            
    except Exception as e:
        logger.error(f"Error in direct microphone test: {e}")
        logger.error(traceback.format_exc())
        return False

def find_working_microphone(max_mics_to_test=5):
    """Try available microphones to find one that works properly using a simpler approach"""
    logger.info("Testing microphones to find one that works properly...")
    
    try:
        import pyaudio
        
        # Create a PyAudio instance directly
        p = pyaudio.PyAudio()
        
        # Get the number of audio devices
        device_count = p.get_device_count()
        logger.info(f"PyAudio found {device_count} audio devices")
        
        # Create a list of input devices
        input_devices = []
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                logger.info(f"Device {i}: {device_info['name']}")
                logger.info(f"  Input channels: {device_info['maxInputChannels']}")
                logger.info(f"  Default sample rate: {device_info['defaultSampleRate']}")
                
                # Only consider devices with input channels
                if device_info['maxInputChannels'] > 0:
                    input_devices.append((i, device_info['name'], device_info['defaultSampleRate']))
            except Exception as e:
                logger.error(f"Error getting info for device {i}: {e}")
        
        # Terminate PyAudio
        p.terminate()
        
        logger.info(f"Found {len(input_devices)} input devices:")
        for i, name, rate in input_devices:
            logger.info(f"  {i}: {name} (rate: {rate})")
        
        # Prioritize devices with "microphone" in the name
        priority_devices = []
        for i, name, rate in input_devices:
            if "microphone" in name.lower() or "mic" in name.lower():
                priority_devices.append((i, name, rate))
        
        if priority_devices:
            logger.info(f"Found {len(priority_devices)} devices with 'microphone' or 'mic' in the name:")
            for i, name, rate in priority_devices:
                logger.info(f"  {i}: {name} (rate: {rate})")
            
            # Limit to max_mics_to_test
            if len(priority_devices) > max_mics_to_test:
                logger.info(f"Limiting to testing {max_mics_to_test} devices")
                priority_devices = priority_devices[:max_mics_to_test]
            
            devices_to_test = priority_devices
        else:
            # If no priority devices, just test the first few input devices
            logger.info(f"No devices with 'microphone' in the name found, testing first {max_mics_to_test}")
            devices_to_test = input_devices[:min(max_mics_to_test, len(input_devices))]
        
        # Test each device with a simple recording test
        working_devices = []
        
        # Sample rates to try (in order of preference)
        sample_rates = [16000, 44100, 48000, 22050, 8000]
        
        for device_index, device_name, default_rate in devices_to_test:
            logger.info(f"Testing device {device_index}: {device_name}...")
            
            # Add the default rate to the beginning of the list if it's not already there
            if default_rate not in sample_rates:
                sample_rates.insert(0, int(default_rate))
            
            device_works = False
            
            # Try each sample rate
            for rate in sample_rates:
                logger.info(f"Trying sample rate {rate}Hz for device {device_index}...")
                
                try:
                    # Create a new PyAudio instance for each test
                    pa = pyaudio.PyAudio()
                    
                    # Try to open a stream
                    logger.info(f"Opening stream for device {device_index} at {rate}Hz...")
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=1024
                    )
                    
                    # Check if the stream is active
                    if stream.is_active():
                        logger.info(f"Stream is active for device {device_index} at {rate}Hz")
                        
                        # Record a short audio sample
                        logger.info(f"Recording 1 second of audio from device {device_index}...")
                        frames = []
                        for i in range(0, int(rate / 1024)):
                            data = stream.read(1024, exception_on_overflow=False)
                            frames.append(data)
                        
                        # Close the stream
                        stream.stop_stream()
                        stream.close()
                        
                        # Convert the audio data to a numpy array
                        import numpy as np
                        audio_data = b''.join(frames)
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Calculate energy
                        energy = np.mean(np.abs(audio_array))
                        
                        # Calculate standard deviation
                        std_dev = np.std(audio_array)
                        
                        logger.info(f"Device {device_index} stats: energy={energy:.2f}, std_dev={std_dev:.2f}")
                        
                        # Check if it's static
                        is_static = std_dev < 100
                        
                        if not is_static and energy > 100:
                            logger.info(f"Device {device_index} appears to be working properly at {rate}Hz!")
                            working_devices.append((device_index, device_name, energy, std_dev, rate))
                            device_works = True
                            break  # Found a working rate, no need to try others
                        else:
                            if is_static:
                                logger.info(f"Device {device_index} is producing static at {rate}Hz")
                            else:
                                logger.info(f"Device {device_index} has very low energy at {rate}Hz")
                    else:
                        logger.info(f"Stream is not active for device {device_index} at {rate}Hz")
                    
                    # Clean up
                    pa.terminate()
                    
                except Exception as e:
                    logger.info(f"Error testing device {device_index} at {rate}Hz: {e}")
                    try:
                        pa.terminate()
                    except:
                        pass
                    continue
                
                # If we found a working configuration, no need to try other rates
                if device_works:
                    break
        
        # Sort working devices by energy level (higher is better)
        working_devices.sort(key=lambda x: x[2], reverse=True)
        
        if working_devices:
            logger.info("Working devices found (sorted by quality):")
            for i, name, energy, std_dev, rate in working_devices:
                logger.info(f"  {i}: {name} (energy: {energy:.2f}, std_dev: {std_dev:.2f}, rate: {rate}Hz)")
            
            # Return the best device, its name, and its sample rate
            best_device = working_devices[0][0]
            best_name = working_devices[0][1]
            best_rate = working_devices[0][4]
            logger.info(f"Best device appears to be {best_device}: {best_name} at {best_rate}Hz")
            return best_device, best_name, best_rate
        else:
            logger.error("No working microphones found")
            return None, None, None
            
    except Exception as e:
        logger.error(f"Error finding working microphone: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

def test_pyaudio():
    """Test if PyAudio is working correctly"""
    logger.info("Testing PyAudio installation...")
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # Get the number of audio devices
        device_count = p.get_device_count()
        logger.info(f"PyAudio found {device_count} audio devices")
        
        # List all devices
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                logger.info(f"Device {i}: {device_info['name']}")
                logger.info(f"  Input channels: {device_info['maxInputChannels']}")
                logger.info(f"  Output channels: {device_info['maxOutputChannels']}")
                logger.info(f"  Default sample rate: {device_info['defaultSampleRate']}")
            except Exception as e:
                logger.error(f"Error getting info for device {i}: {e}")
        
        # Find default input device
        try:
            default_input = p.get_default_input_device_info()
            logger.info(f"Default input device: {default_input['name']} (index {default_input['index']})")
        except Exception as e:
            logger.error(f"Error getting default input device: {e}")
        
        # Terminate PyAudio
        p.terminate()
        
        logger.info("PyAudio test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing PyAudio: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_pyaudio_issues():
    """Try to fix common PyAudio issues"""
    logger.info("Attempting to fix PyAudio issues...")
    
    try:
        # Check if PyAudio is installed
        try:
            import pyaudio
            logger.info("PyAudio is installed")
        except ImportError:
            logger.error("PyAudio is not installed. Please install it with: pip install pyaudio")
            return False
        
        # Check if PortAudio is installed (Windows only)
        if os.name == 'nt':
            try:
                p = pyaudio.PyAudio()
                p.terminate()
                logger.info("PortAudio is working")
            except Exception as e:
                logger.error(f"PortAudio error: {e}")
                logger.error("On Windows, you may need to install PyAudio using a wheel file.")
                logger.error("Try: pip install pipwin && pipwin install pyaudio")
                return False
        
        # Check for permissions issues (Linux only)
        if os.name == 'posix':
            try:
                import subprocess
                groups = subprocess.check_output(['groups']).decode('utf-8').strip()
                logger.info(f"User groups: {groups}")
                
                if 'audio' not in groups:
                    logger.warning("User is not in the 'audio' group. This may cause permission issues.")
                    logger.warning("Try: sudo usermod -a -G audio $USER")
            except Exception as e:
                logger.error(f"Error checking user groups: {e}")
        
        logger.info("PyAudio check completed")
        return True
    except Exception as e:
        logger.error(f"Error fixing PyAudio issues: {e}")
        logger.error(traceback.format_exc())
        return False

def test_microphone_with_pyaudio(mic_index=None, duration=5):
    """Test a microphone directly with PyAudio"""
    logger.info(f"Testing microphone {mic_index if mic_index is not None else 'default'} with PyAudio...")
    
    try:
        import pyaudio
        import wave
        import numpy as np
        import os
        
        # Create PyAudio instance
        p = pyaudio.PyAudio()
        
        # Sample rate to use
        RATE = 16000
        
        # Get device info if mic_index is specified
        if mic_index is not None:
            try:
                device_info = p.get_device_info_by_index(mic_index)
                logger.info(f"Device {mic_index}: {device_info['name']}")
                logger.info(f"  Input channels: {device_info['maxInputChannels']}")
                logger.info(f"  Default sample rate: {device_info['defaultSampleRate']}")
                
                if device_info['maxInputChannels'] == 0:
                    logger.error(f"Device {mic_index} has no input channels")
                    p.terminate()
                    return False
                
                # Use the device's default sample rate if available
                if device_info['defaultSampleRate'] > 0:
                    RATE = int(device_info['defaultSampleRate'])
                    logger.info(f"Using device's default sample rate: {RATE}Hz")
            except Exception as e:
                logger.error(f"Error getting device info: {e}")
                p.terminate()
                return False
        
        # Set up recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        
        # Create debug directory
        debug_dir = os.path.join(os.getcwd(), "debug_audio")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(debug_dir, f"pyaudio_test_{timestamp}.wav")
        
        logger.info(f"Recording {duration} seconds of audio to {filename} at {RATE}Hz...")
        
        # Open stream
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=CHUNK
            )
            
            # Record audio
            frames = []
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Print progress every second
                if i % int(RATE / CHUNK) == 0:
                    seconds_recorded = i / (RATE / CHUNK)
                    logger.info(f"Recorded {seconds_recorded:.1f} seconds...")
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
            # Save the recorded audio to a WAV file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Analyze the audio
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate energy
            energy = np.mean(np.abs(audio_array))
            
            # Calculate standard deviation
            std_dev = np.std(audio_array)
            
            logger.info(f"Audio stats: energy={energy:.2f}, std_dev={std_dev:.2f}")
            
            # Check if it's static
            is_static = std_dev < 100
            
            if not is_static and energy > 100:
                logger.info("Microphone appears to be working properly!")
                logger.info(f"Audio saved to {filename}")
                p.terminate()
                return True
            else:
                if is_static:
                    logger.info("Microphone is producing static")
                else:
                    logger.info("Microphone has very low energy")
                logger.info(f"Audio saved to {filename}")
                p.terminate()
                return False
                
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            logger.error(traceback.format_exc())
            p.terminate()
            return False
            
    except Exception as e:
        logger.error(f"Error testing microphone with PyAudio: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Maxwell Voice Assistant")
    parser.add_argument("--wake-word", default="hey maxwell", help="Wake word to activate the assistant")
    parser.add_argument("--interrupt-word", default="stop talking", help="Word to interrupt the assistant")
    parser.add_argument("--voice", default="bm_lewis", help="Voice for text-to-speech")
    parser.add_argument("--speed", default=1.25, type=float, help="Speech speed (1.0 is normal)")
    parser.add_argument("--offline", action="store_true", help="Use offline speech recognition")
    parser.add_argument("--continuous", action="store_true", help="Stay in conversation mode until explicitly ended")
    parser.add_argument("--list-voices", action="store_true", help="List available TTS voices and exit")
    parser.add_argument("--model", default="dolphin-llama3:8b-v2.9-q4_0", help="Ollama model to use")
    parser.add_argument("--ollama-host", default="localhost", help="Ollama host address")
    parser.add_argument("--ollama-port", default=11434, type=int, help="Ollama port")
    parser.add_argument("--test", action="store_true", help="Test mode - immediately enter conversation mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--use-mcp", action="store_true", help="Enable MCP tools integration")
    parser.add_argument("--mcp-port", default=8080, type=int, help="Port for MCP server")
    parser.add_argument("--mic-index", type=int, help="Specific microphone index to use")
    parser.add_argument("--keyboard-interrupt", action="store_true", help="Use keyboard 's' key to interrupt speech")
    parser.add_argument("--keyboard-mode", action="store_true", help="Use keyboard input instead of microphone")
    parser.add_argument("--always-listen", action="store_true", help="Always listen for commands without wake word")
    parser.add_argument("--energy-threshold", type=int, default=300, help="Energy threshold for speech recognition (lower = more sensitive)")
    parser.add_argument("--list-mics", action="store_true", help="List available microphones and exit")
    parser.add_argument("--save-audio", action="store_true", help="Save audio files for debugging")
    parser.add_argument("--mic-name", help="Specific microphone name to use (partial match)")
    parser.add_argument("--direct-mic-test", action="store_true", help="Run a direct microphone test and exit")
    parser.add_argument("--test-duration", type=int, default=10, help="Duration of microphone test in seconds")
    parser.add_argument("--force-listen", action="store_true", help="Force always-listen mode with very low energy threshold")
    parser.add_argument("--find-working-mic", action="store_true", help="Test all microphones and find the best one")
    parser.add_argument("--max-mics", type=int, default=5, help="Maximum number of microphones to test when finding a working mic")
    parser.add_argument("--test-pyaudio", action="store_true", help="Test PyAudio installation and exit")
    parser.add_argument("--fix-audio", action="store_true", help="Try to fix audio issues and exit")
    parser.add_argument("--pyaudio-test", action="store_true", help="Test microphone directly with PyAudio")
    parser.add_argument("--sample-rate", type=int, help="Sample rate to use for microphone input")
    parser.add_argument("--auto-find-mic", action="store_true", help="Automatically find and use a working microphone")
    parser.add_argument("--listen-timeout", type=int, default=7, 
                        help="Timeout in seconds for listening to commands (default: 7)")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # List voices if requested
    if args.list_voices:
        from speech import TextToSpeech
        TextToSpeech.list_available_voices()
        return
        
    # List microphones if requested
    if args.list_mics:
        list_microphones()
        return
        
    # Download required models
    download_models(offline_mode=args.offline)
    
    # If mic-name is specified, find the matching microphone index
    if args.mic_name and not args.mic_index:
        mics = list_microphones()
        for i, mic_name in enumerate(mics):
            if args.mic_name.lower() in mic_name.lower():
                logger.info(f"Found microphone matching '{args.mic_name}': {i}: {mic_name}")
                args.mic_index = i
                break
        else:
            logger.warning(f"No microphone found matching '{args.mic_name}'")
    
    # Run direct microphone test if requested
    if args.direct_mic_test:
        logger.info("Running direct microphone test...")
        test_microphone_directly(args.mic_index, args.test_duration)
        return
    
    # Find a working microphone if requested
    if args.find_working_mic:
        logger.info("Finding a working microphone...")
        mic_index, mic_name, sample_rate = find_working_microphone(args.max_mics)
        if mic_index is not None:
            logger.info(f"Found working microphone with index {mic_index}: {mic_name} at {sample_rate}Hz")
            logger.info(f"Use --mic-index {mic_index} --sample-rate {sample_rate} to use this microphone")
        return
    
    # Auto-find a working microphone if requested or if force-listen is enabled
    if (args.auto_find_mic or args.force_listen) and not args.mic_index and not args.keyboard_mode:
        logger.info("Auto-finding a working microphone...")
        mic_index, mic_name, sample_rate = find_working_microphone(args.max_mics)
        if mic_index is not None:
            logger.info(f"Using microphone with index {mic_index}: {mic_name} at {sample_rate}Hz")
            args.mic_index = mic_index
            args.sample_rate = sample_rate
        else:
            logger.warning("No working microphone found, falling back to default")
    
    # Test PyAudio if requested
    if args.test_pyaudio:
        logger.info("Testing PyAudio installation...")
        test_pyaudio()
        return
    
    # Try to fix audio issues if requested
    if args.fix_audio:
        logger.info("Attempting to fix audio issues...")
        fix_pyaudio_issues()
        return
    
    # Test microphone with PyAudio if requested
    if args.pyaudio_test:
        logger.info("Testing microphone with PyAudio...")
        test_microphone_with_pyaudio(args.mic_index, args.test_duration)
        return
    
    # Create config
    config = Config(
        wake_word=args.wake_word,
        interrupt_word=args.interrupt_word,
        voice=args.voice,
        speed=args.speed,
        offline_mode=args.offline,
        continuous_conversation=args.continuous,
        model=args.model,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        test_mode=args.test,
        use_mcp=args.use_mcp,
        mcp_port=args.mcp_port,
        keyboard_interrupt=args.keyboard_interrupt,
        keyboard_mode=args.keyboard_mode,
        mic_index=args.mic_index,
        always_listen=args.always_listen,
        energy_threshold=args.energy_threshold,
        save_audio=args.save_audio,
        sample_rate=args.sample_rate,
        auto_find_mic=args.auto_find_mic,
        listen_timeout=args.listen_timeout
    )
    
    # Log the configuration
    logger.info(f"Starting Maxwell with configuration:")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")
    
    # Create and run assistant
    assistant = Maxwell(config)
    try:
        assistant.run()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        logger.error(traceback.format_exc())
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    main() 