#!/usr/bin/env python3
"""
Keyboard input handler for Maxwell
"""

import os
import threading
import time
import logging
import platform

# Get the logger instance
logger = logging.getLogger("maxwell")

class KeyboardHandler:
    """Handles keyboard input for the assistant, specifically to detect
    when the space bar is pressed to interrupt speech."""
    
    def __init__(self, tts):
        """Initialize the keyboard handler.
        
        Args:
            tts: TextToSpeech instance to stop when space is pressed
        """
        self.tts = tts
        self.running = False
        self.thread = None
        self.platform = platform.system()
        
        # Add debounce mechanism to prevent phantom keypresses
        self.last_keypress_time = 0
        self.debounce_interval = 1.0  # 1 second debounce interval
        
        # Import platform-specific modules
        if self.platform == "Windows":
            try:
                import msvcrt
                self.msvcrt = msvcrt
                logger.info("Windows keyboard handler initialized")
            except ImportError:
                logger.error("Failed to import msvcrt for Windows keyboard handling")
                self.msvcrt = None
        else:
            # Unix-based systems
            try:
                import termios
                import sys
                import select
                self.termios = termios
                self.sys = sys
                self.select = select
                logger.info("Unix keyboard handler initialized")
            except ImportError:
                logger.error("Failed to import modules for Unix keyboard handling")
                self.termios = None
    
    def start(self, tts=None):
        """Start the keyboard handler thread.
        
        Args:
            tts: Optional TextToSpeech instance to update
        """
        if tts:
            self.tts = tts
            
        if self.thread and self.thread.is_alive():
            logger.info("Keyboard handler already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._keyboard_thread, daemon=True)
        self.thread.start()
        logger.info("Keyboard handler thread started")
        
    def stop(self):
        """Stop the keyboard handler thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=1.0)
                logger.info("Keyboard handler thread stopped")
            except Exception as e:
                logger.error(f"Error stopping keyboard handler thread: {e}")
    
    def _keyboard_thread(self):
        """Main keyboard handling thread function."""
        logger.info("Keyboard handler thread running")
        
        if self.platform == "Windows" and self.msvcrt:
            self._handle_windows_keyboard()
        elif self.termios:
            self._handle_unix_keyboard()
        else:
            logger.error("No keyboard handler available for this platform")
            self.running = False
    
    def _handle_windows_keyboard(self):
        """Handle keyboard input on Windows."""
        while self.running:
            # Check if a key has been pressed
            if self.msvcrt.kbhit():
                key = self.msvcrt.getch()
                # Space key (32 in ASCII)
                if key == b' ':
                    current_time = time.time()
                    # Check if enough time has passed since the last keypress
                    if current_time - self.last_keypress_time > self.debounce_interval:
                        logger.info("Space key pressed - interrupting speech")
                        if self.tts and hasattr(self.tts, 'stop'):
                            self.tts.stop()
                        self.last_keypress_time = current_time
                        
                        # Flush any pending input to avoid phantom key presses
                        self._flush_input_buffer()
                    else:
                        logger.info(f"Ignoring potential phantom space keypress (within debounce interval)")
                    
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
    
    def _handle_unix_keyboard(self):
        """Handle keyboard input on Unix-based systems."""
        # Save the terminal settings
        fd = self.sys.stdin.fileno()
        old_settings = self.termios.tcgetattr(fd)
        
        try:
            # Configure terminal for non-blocking input
            new_settings = self.termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~self.termios.ICANON
            new_settings[3] = new_settings[3] & ~self.termios.ECHO
            self.termios.tcsetattr(fd, self.termios.TCSANOW, new_settings)
            
            while self.running:
                # Poll for keyboard input
                ready, _, _ = self.select.select([self.sys.stdin], [], [], 0.1)
                
                if ready:
                    key = self.sys.stdin.read(1)
                    # Space key
                    if key == ' ':
                        current_time = time.time()
                        # Check if enough time has passed since the last keypress
                        if current_time - self.last_keypress_time > self.debounce_interval:
                            logger.info("Space key pressed - interrupting speech")
                            if self.tts and hasattr(self.tts, 'stop'):
                                self.tts.stop()
                            self.last_keypress_time = current_time
                            
                            # Flush any pending input to avoid phantom key presses
                            self._flush_input_buffer()
                        else:
                            logger.info(f"Ignoring potential phantom space keypress (within debounce interval)")
                
                # Sleep to avoid high CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in Unix keyboard handler: {e}")
        finally:
            # Restore terminal settings
            self.termios.tcsetattr(fd, self.termios.TCSANOW, old_settings)
    
    def _flush_input_buffer(self):
        """Flush any pending input in the buffer to prevent phantom keypresses."""
        logger.debug("Flushing input buffer")
        try:
            if self.platform == "Windows" and self.msvcrt:
                # Flush the Windows console input buffer
                while self.msvcrt.kbhit():
                    self.msvcrt.getch()
            elif self.termios:
                # Flush the Unix stdin buffer
                self.termios.tcflush(self.sys.stdin.fileno(), self.termios.TCIOFLUSH)
        except Exception as e:
            logger.error(f"Error flushing input buffer: {e}")
            # Continue even if flushing fails 