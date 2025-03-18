#!/usr/bin/env python3
"""
Test script for TextToSpeech integration with the assistant
"""

import time
from speech import TextToSpeech
from config_loader import load_config, create_config_object
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_speak():
    """Test the TextToSpeech integration"""
    print("\n==== Testing Assistant TTS Integration ====\n")
    
    # Load config
    print("Loading configuration...")
    config_dict = load_config()
    config = create_config_object(config_dict)
    print(f"Loaded configuration with TTS paths: {config.get('tts_model_path', 'Not set')}, {config.get('tts_voices_path', 'Not set')}")
    
    # Simulate speech in the assistant
    print("\nInitializing TTS...")
    tts = TextToSpeech(config=config)
    
    # Wait for TTS initialization
    time.sleep(1)
    
    if tts.engine is None:
        print("\n❌ ERROR: TTS engine failed to initialize")
        return False
    
    # First message - startup message
    print("\nSimulating assistant startup message...\n")
    startup_message = "Hello, Maxwell here. Say 'hello maxwell' to get my attention."
    print(f"Speaking: \"{startup_message}\"")
    
    # Speak the message
    speaking = True
    tts.speak(startup_message)
    
    # Wait for speech to complete
    while speaking:
        speaking = tts.is_speaking()
        if speaking:
            print(".", end="", flush=True)
            time.sleep(0.5)
    
    print("\n\nDid you hear the startup message? (y/n)")
    heard_startup = input("> ").lower().startswith('y')
    
    if heard_startup:
        print("✅ Startup message test passed!")
    else:
        print("❌ Failed to hear startup message")
    
    # Test a response message
    print("\nSimulating assistant response message...\n")
    response_message = "I found the answer to your question. The sky appears blue because of a phenomenon called Rayleigh scattering."
    print(f"Speaking: \"{response_message}\"")
    
    # Speak the message
    speaking = True
    tts.speak(response_message)
    
    # Wait for speech to complete
    while speaking:
        speaking = tts.is_speaking()
        if speaking:
            print(".", end="", flush=True)
            time.sleep(0.5)
    
    print("\n\nDid you hear the response message? (y/n)")
    heard_response = input("> ").lower().startswith('y')
    
    if heard_response:
        print("✅ Response message test passed!")
    else:
        print("❌ Failed to hear response message")
    
    # Final results
    if heard_startup and heard_response:
        print("\n✅ TTS integration tests PASSED")
        return True
    else:
        print("\n❌ TTS integration tests FAILED")
        return False

if __name__ == "__main__":
    test_speak() 