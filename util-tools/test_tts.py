#!/usr/bin/env python3
"""
Simple test script for Kokoro TTS
"""

import os
import logging
import sounddevice as sd
import numpy as np
from kokoro_onnx import Kokoro

# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tts_test")

def test_kokoro_tts():
    """Test Kokoro TTS functionality"""
    
    print("\n==== Testing Kokoro TTS ====\n")
    
    # Get the current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "kokoro-v1.0.onnx")
    voices_path = os.path.join(base_dir, "voices-v1.0.bin")
    
    # Check if files exist
    print(f"Looking for model file: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found!")
        return False
    else:
        print(f"✅ Found model file: {model_path}")
    
    print(f"Looking for voices file: {voices_path}")
    if not os.path.exists(voices_path):
        print(f"❌ ERROR: Voices file not found!")
        return False
    else:
        print(f"✅ Found voices file: {voices_path}")
    
    try:
        # Initialize Kokoro
        print("\nInitializing Kokoro TTS...")
        engine = Kokoro(model_path=model_path, voices_path=voices_path)
        print("✅ Kokoro initialized successfully")
        
        # Get available voices
        print("\nGetting available voices...")
        voices = engine.get_voices()
        print(f"✅ Found {len(voices)} voices")
        print(f"First 5 voices: {', '.join(voices[:5])}")
        
        # Select a voice
        voice = voices[0]  # Use the first available voice
        print(f"\nUsing voice: {voice}")
        
        # Test audio output
        print("\nChecking audio devices...")
        try:
            output_devices = [d for d in sd.query_devices() if d['max_output_channels'] > 0]
            print(f"✅ Found {len(output_devices)} output devices")
            default_device = sd.query_devices(kind='output')
            print(f"Default output device: {default_device['name']}")
        except Exception as e:
            print(f"❌ Error checking audio devices: {e}")
        
        # Generate audio
        text = "This is a test of the Kokoro text-to-speech system. If you can hear this, the system is working correctly."
        print(f"\nGenerating audio for text: '{text}'")
        audio, sample_rate = engine.create(text=text, voice=voice, speed=1.0)
        
        print(f"✅ Audio generated: shape={audio.shape}, sample_rate={sample_rate}, min={audio.min()}, max={audio.max()}")
        
        # Play audio
        print("\nPlaying audio... (you should hear speech)")
        sd.play(audio, sample_rate)
        sd.wait()
        print("✅ Audio playback completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Kokoro TTS: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kokoro_tts()
    
    if success:
        print("\n✅ Kokoro TTS test completed successfully!")
    else:
        print("\n❌ Kokoro TTS test failed!") 