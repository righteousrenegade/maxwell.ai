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
        
        # Custom callback for space key (if provided)
        self.space_key_callback = None
        
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
        try:
            logger.info("Keyboard handler thread running")
            
            if self.platform == "Windows" and self.msvcrt:
                self._handle_windows_keyboard()
            elif self.termios:
                self._handle_unix_keyboard()
            else:
                logger.error("No keyboard handler available for this platform")
                self.running = False
        except Exception as e:
            logger.error(f"Critical error in keyboard handler thread: {e}")
            logger.error("Keyboard handler thread stopping but keeping program running")
            # Don't re-raise to avoid terminating the main program
    
    def set_space_key_callback(self, callback):
        """Set a custom callback to be called when space key is pressed.
        
        Args:
            callback: A function to call when space key is pressed.
                     If provided, this will be called instead of directly stopping TTS.
        """
        logger.info("=== SPACE KEY DEBUG: Setting custom space key callback ===")
        self.space_key_callback = callback
        logger.info("=== SPACE KEY DEBUG: Custom space key callback set ===")
    
    def _handle_windows_keyboard(self):
        """Handle keyboard input on Windows."""
        try:
            while self.running:
                try:
                    # Check if a key has been pressed
                    if self.msvcrt.kbhit():
                        try:
                            key = self.msvcrt.getch()
                            # Space key (32 in ASCII)
                            if key == b' ':
                                current_time = time.time()
                                # Check if enough time has passed since the last keypress
                                if current_time - self.last_keypress_time > self.debounce_interval:
                                    logger.info("Space key pressed - interrupting speech")
                                    
                                    # Use custom callback if provided
                                    if self.space_key_callback:
                                        logger.info("=== SPACE KEY DEBUG: Using custom space key callback ===")
                                        try:
                                            self.space_key_callback()
                                            logger.info("=== SPACE KEY DEBUG: Custom callback completed successfully ===")
                                        except Exception as callback_error:
                                            logger.error(f"=== SPACE KEY DEBUG: Error in custom callback: {callback_error} ===")
                                            # Fall back to standard approach if callback fails
                                            self._safe_stop_tts()
                                    else:
                                        # Use standard approach
                                        try:
                                            if self.tts and hasattr(self.tts, 'stop'):
                                                # Call stop in a protected way
                                                self._safe_stop_tts()
                                        except Exception as e:
                                            logger.error(f"Error stopping TTS: {e}")
                                            # Continue despite the error
                                            
                                    self.last_keypress_time = current_time
                                    
                                    # Flush any pending input to avoid phantom key presses
                                    self._flush_input_buffer()
                                else:
                                    logger.info(f"Ignoring potential phantom space keypress (within debounce interval)")
                        except Exception as key_error:
                            logger.error(f"Error handling key press: {key_error}")
                            # Continue running even if an error occurs while handling key
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt in keyboard handler - ignoring")
                    # Don't let KeyboardInterrupt propagate
                    continue
                except Exception as loop_error:
                    logger.error(f"Error in keyboard handler loop: {loop_error}")
                    # Don't terminate the loop for any error
                    time.sleep(0.5)  # Sleep a bit longer after an error
        except Exception as outer_error:
            logger.error(f"Fatal error in Windows keyboard handler: {outer_error}")
            # Don't propagate errors from the keyboard handler
            # Just log and continue - don't terminate the application
            self.running = False
    
    def _handle_unix_keyboard(self):
        """Handle keyboard input on Unix-based systems."""
        # Save the terminal settings
        fd = self.sys.stdin.fileno()
        old_settings = None
        try:
            old_settings = self.termios.tcgetattr(fd)
        except Exception as term_error:
            logger.error(f"Error getting terminal settings: {term_error}")
            # Continue even with error
        
        try:
            # Only modify terminal if we got valid settings
            if old_settings:
                try:
                    # Configure terminal for non-blocking input
                    new_settings = self.termios.tcgetattr(fd)
                    new_settings[3] = new_settings[3] & ~self.termios.ICANON
                    new_settings[3] = new_settings[3] & ~self.termios.ECHO
                    self.termios.tcsetattr(fd, self.termios.TCSANOW, new_settings)
                except Exception as setup_error:
                    logger.error(f"Error configuring terminal: {setup_error}")
                
            try:
                while self.running:
                    try:
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
                                    
                                    # Use custom callback if provided
                                    if self.space_key_callback:
                                        logger.info("=== SPACE KEY DEBUG: Using custom space key callback ===")
                                        try:
                                            self.space_key_callback()
                                            logger.info("=== SPACE KEY DEBUG: Custom callback completed successfully ===")
                                        except Exception as callback_error:
                                            logger.error(f"=== SPACE KEY DEBUG: Error in custom callback: {callback_error} ===")
                                            # Fall back to standard approach if callback fails
                                            self._safe_stop_tts()
                                    else:
                                        # Use standard approach
                                        try:
                                            if self.tts and hasattr(self.tts, 'stop'):
                                                # Call stop in a protected way
                                                self._safe_stop_tts()
                                        except Exception as e:
                                            logger.error(f"Error stopping TTS: {e}")
                                            # Continue despite the error
                                            
                                    self.last_keypress_time = current_time
                                    
                                    # Flush any pending input to avoid phantom key presses
                                    self._flush_input_buffer()
                                else:
                                    logger.info(f"Ignoring potential phantom space keypress (within debounce interval)")
                    except KeyboardInterrupt:
                        logger.warning("KeyboardInterrupt in Unix keyboard handler - ignoring")
                        # Don't let KeyboardInterrupt propagate
                        continue
                    except Exception as key_error:
                        logger.error(f"Error handling keyboard input: {key_error}")
                        # Continue running even if an error occurs handling input
                    
                    # Sleep to avoid high CPU usage
                    try:
                        time.sleep(0.1)
                    except Exception:
                        # Handle potential interrupt during sleep
                        pass
            except Exception as loop_error:
                logger.error(f"Error in Unix keyboard handler loop: {loop_error}")
                # Don't terminate the thread on loop error
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in Unix keyboard handler: {e}")
            # No re-raising to avoid terminating the program
        finally:
            # Restore terminal settings
            if old_settings:
                try:
                    self.termios.tcsetattr(fd, self.termios.TCSANOW, old_settings)
                    logger.debug("Unix terminal settings restored")
                except Exception as term_error:
                    logger.error(f"Error restoring terminal settings: {term_error}")
                    # Continue even after failure to restore settings
    
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
    
    def _safe_stop_tts(self):
        """Safely stop TTS with extra protection against exceptions"""
        try:
            logger.info("=== SPACE KEY DEBUG: _safe_stop_tts STARTED ===")
            
            # First check if TTS exists
            if self.tts is None:
                logger.warning("TTS is None in _safe_stop_tts - nothing to stop")
                logger.info("=== SPACE KEY DEBUG: _safe_stop_tts ENDED (tts is None) ===")
                
                # Try direct sounddevice stop as fallback
                try:
                    import sounddevice as sd
                    sd.stop()
                    logger.info("=== SPACE KEY DEBUG: Called sd.stop() directly when TTS is None ===")
                except Exception as sd_err:
                    logger.error(f"=== SPACE KEY DEBUG: Error in direct sd.stop() when TTS is None: {sd_err} ===")
                return
            
            logger.info("=== SPACE KEY DEBUG: TTS object exists ===")
            
            # Check if stop method exists
            if not hasattr(self.tts, 'stop'):
                logger.warning("TTS has no stop method - cannot stop")
                logger.info("=== SPACE KEY DEBUG: _safe_stop_tts ENDED (no stop method) ===")
                
                # Try direct sounddevice stop as fallback
                try:
                    import sounddevice as sd
                    sd.stop()
                    logger.info("=== SPACE KEY DEBUG: Called sd.stop() directly when no stop method ===")
                except Exception as sd_err:
                    logger.error(f"=== SPACE KEY DEBUG: Error in direct sd.stop() when no stop method: {sd_err} ===")
                return
            
            logger.info("=== SPACE KEY DEBUG: TTS has stop method ===")
            
            # Try to get _speaking_lock status if it exists
            if hasattr(self.tts, '_speaking_lock'):
                try:
                    logger.info(f"=== SPACE KEY DEBUG: TTS speaking lock exists, is_speaking: {self.tts.is_speaking()} ===")
                except Exception as e:
                    logger.error(f"=== SPACE KEY DEBUG: Error checking is_speaking: {e} ===")
            
            # Use a timeout approach to avoid blocking indefinitely
            def stop_thread_func():
                try:
                    # Call in a protected way
                    logger.info("=== SPACE KEY DEBUG: About to call tts.stop() in thread ===")
                    self.tts.stop()
                    logger.info("=== SPACE KEY DEBUG: tts.stop() completed successfully in thread ===")
                except Exception as stop_err:
                    logger.error(f"=== SPACE KEY DEBUG: ERROR in threaded tts.stop(): {stop_err} ===")
                    # Try direct sounddevice stop as last resort 
                    try:
                        import sounddevice as sd
                        sd.stop()
                        logger.info("=== SPACE KEY DEBUG: Called sd.stop() after threaded stop error ===")
                    except Exception:
                        pass
            
            # Create and start the stop thread
            stop_thread = threading.Thread(target=stop_thread_func)
            stop_thread.daemon = True
            stop_thread.start()
            
            # Wait with timeout
            stop_thread.join(timeout=1.0)
            if stop_thread.is_alive():
                logger.warning("=== SPACE KEY DEBUG: Stop thread timed out after 1.0s ===")
                # Try direct approach as fallback
                try:
                    import sounddevice as sd
                    sd.stop()
                    logger.info("=== SPACE KEY DEBUG: Called sd.stop() after thread timeout ===")
                except Exception:
                    pass
            
            # Add a delay to ensure the call completes
            try:
                time.sleep(0.1)
                logger.info("=== SPACE KEY DEBUG: Post-stop delay completed ===")
            except Exception as sleep_e:
                logger.error(f"=== SPACE KEY DEBUG: Error in post-stop delay: {sleep_e} ===")
            
            # Check the speaking state after stopping
            try:
                if hasattr(self.tts, 'is_speaking'):
                    is_still_speaking = self.tts.is_speaking()
                    logger.info(f"=== SPACE KEY DEBUG: After stop, is_speaking: {is_still_speaking} ===")
                    
                    # If still speaking after stop attempt, try one more time with direct approach
                    if is_still_speaking:
                        logger.warning("=== SPACE KEY DEBUG: TTS still speaking after stop, trying direct approach ===")
                        try:
                            import sounddevice as sd
                            sd.stop()
                            logger.info("=== SPACE KEY DEBUG: Called sd.stop() when still speaking after stop ===")
                        except Exception:
                            pass
            except Exception as check_e:
                logger.error(f"=== SPACE KEY DEBUG: Error checking speaking state after stop: {check_e} ===")
            
            logger.info("=== SPACE KEY DEBUG: _safe_stop_tts COMPLETED SUCCESSFULLY ===")
        except Exception as e:
            logger.error(f"=== SPACE KEY DEBUG: ERROR in _safe_stop_tts: {e} ===")
            logger.error(f"=== SPACE KEY DEBUG: Error traceback: {e.__class__.__name__} ===")
            
            # Try direct sounddevice stop as ultimate fallback
            try:
                import sounddevice as sd
                sd.stop()
                logger.info("=== SPACE KEY DEBUG: Called sd.stop() after exception in _safe_stop_tts ===")
            except Exception:
                pass
            # Swallow exception completely 