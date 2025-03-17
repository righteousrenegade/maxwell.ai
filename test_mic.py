#!/usr/bin/env python3
import speech_recognition as sr
import time
import threading

def main():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # Adjust as needed
    recognizer.dynamic_energy_threshold = False
    
    # Function to listen for audio in background
    def listen_in_background():
        print("Starting background listening...")
        should_stop = False
        listen_count = 0
        
        # Create microphone instance
        with sr.Microphone() as source:
            # Adjust for ambient noise
            print("Calibrating microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print(f"Energy threshold after calibration: {recognizer.energy_threshold}")
            
            # Main listening loop
            while not should_stop:
                try:
                    listen_count += 1
                    print(f"Listening for speech... (attempt #{listen_count})")
                    
                    # Listen for audio with a short timeout
                    audio = recognizer.listen(source, timeout=1)
                    
                    print(f"Audio detected! Length: {len(audio.frame_data)/audio.sample_rate:.2f}s")
                    
                    # Try to recognize
                    try:
                        text = recognizer.recognize_google(audio)
                        print(f"Recognized: '{text}'")
                        
                        # Check for stop word
                        if "stop" in text.lower():
                            print("Stop command detected, exiting...")
                            should_stop = True
                            
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                        print(f"Recognition error: {e}")
                        
                except sr.WaitTimeoutError:
                    # This is normal, just continue
                    print("Timeout, no speech detected")
                except Exception as e:
                    print(f"Error in listen loop: {e}")
                    time.sleep(0.1)
                    
                # Small pause to prevent tight loop
                time.sleep(0.1)
                
            print("Listening thread finished")
    
    # Start listening in background thread
    listen_thread = threading.Thread(target=listen_in_background, daemon=True)
    listen_thread.start()
    
    print("Main thread running. Press Ctrl+C to stop.")
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
            print("Main thread still running. Say something or press Ctrl+C to exit.")
    except KeyboardInterrupt:
        print("Exiting program...")

if __name__ == "__main__":
    main() 