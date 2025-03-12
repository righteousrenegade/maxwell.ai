#!/usr/bin/env python3
"""
Utility functions and global variables for the voice assistant.

This module contains utility functions and global variables used by the voice assistant.
"""

# Standard library imports
import os
import sys
import time
import signal
import logging
from datetime import datetime

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
last_speaking_time = 0  # Track when the assistant last finished speaking
speaking_cooldown = 1.0  # Cooldown period in seconds after speaking
waiting_for_user_input = False  # Flag to indicate when the assistant is waiting for user input
max_silence_before_prompt = 10.0  # Maximum silence time before prompting user again
wake_word = "hey maxwell"  # Wake word to activate the assistant
wake_word_active = False  # Flag to indicate if the wake word has been detected
pending_user_input = None  # Store user input received while speaking
listen_while_speaking = True  # Whether to listen while speaking
interrupt_word = "stop talking"  # Word to interrupt the assistant
command_prefix = "execute"  # Prefix for command execution
is_interrupted = False  # Flag to indicate if the assistant was interrupted

# Available commands
available_commands = {
    "weather": "Get the current weather",
    "time": "Get the current time",
    "date": "Get the current date",
    "news": "Get the latest news headlines",
    "joke": "Tell a joke",
    "reminder": "Set a reminder",
    "timer": "Set a timer",
    "search": "Search the web",
    "details": "Get more details about a search result by number or title",
}

def handle_interrupt(sig, frame):
    """Handle interrupt signal.
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    global stop_listening
    
    logger.info("Interrupt signal received, stopping...")
    print("\nInterrupt signal received, stopping...")
    
    # Set flag to stop listening
    stop_listening = True
    
    # Exit gracefully
    sys.exit(0)

def setup_signal_handlers():
    """Set up signal handlers."""
    # Register signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, handle_interrupt)
    
    # Register signal handler for SIGTERM
    signal.signal(signal.SIGTERM, handle_interrupt)
    
    logger.info("Signal handlers registered")

def format_time():
    """Format the current time.
    
    Returns:
        Formatted time string
    """
    return datetime.now().strftime("%I:%M %p")

def format_date():
    """Format the current date.
    
    Returns:
        Formatted date string
    """
    return datetime.now().strftime("%A, %B %d, %Y")

def print_boxed_message(message, box_char="*"):
    """Print a message in a box.
    
    Args:
        message: Message to print
        box_char: Character to use for the box
    """
    box_width = len(message) + 4
    print("\n" + box_char * box_width)
    print(f"{box_char} {message} {box_char}")
    print(box_char * box_width + "\n")

def print_assistant_response(response):
    """Print an assistant response in a formatted way.
    
    Args:
        response: Response text
    """
    print("\n" + "-"*50)
    print(f"ASSISTANT RESPONSE: \"{response}\"")
    print("-"*50 + "\n")
    logger.info(f"Assistant response: {response}")

def print_user_input(input_text):
    """Print user input in a formatted way.
    
    Args:
        input_text: User input text
    """
    print("\n" + "="*50)
    print(f"USER INPUT: \"{input_text}\"")
    print("="*50 + "\n")
    logger.info(f"User input: {input_text}")

def print_listening_message():
    """Print a message indicating that the assistant is listening."""
    print_boxed_message("LISTENING FOR SPEECH...")
    logger.info("Listening for speech...")

def print_heard_message(text):
    """Print a message indicating what the assistant heard.
    
    Args:
        text: Recognized text
    """
    print_boxed_message(f"HEARD: \"{text}\"")
    logger.info(f"Heard: {text}")

def is_wake_word(text, custom_wake_word=None):
    """Check if the text contains a wake word.
    
    Args:
        text: Text to check
        custom_wake_word: Custom wake word to use
        
    Returns:
        True if the text contains a wake word, False otherwise
    """
    if not text:
        return False
        
    # Use custom wake word if provided
    wake_word_to_check = custom_wake_word.lower() if custom_wake_word else wake_word.lower()
    text_lower = text.lower()
    
    # Define wake word variations
    wake_word_parts = wake_word_to_check.split()
    main_name = wake_word_parts[-1] if len(wake_word_parts) > 0 else wake_word_to_check
    
    # Check for wake word variations
    return (wake_word_to_check in text_lower or 
            main_name in text_lower or 
            any(phrase in text_lower for phrase in [
                f"hi {main_name}", f"hello {main_name}", f"ok {main_name}", 
                f"okay {main_name}", f"hey {main_name}", f"{main_name} please"
            ]))

def is_end_conversation(text):
    """Check if the text indicates the end of a conversation.
    
    Args:
        text: Text to check
        
    Returns:
        True if the text indicates the end of a conversation, False otherwise
    """
    if not text:
        return False
        
    text_lower = text.lower()
    
    # Check for end conversation phrases
    return any(phrase in text_lower for phrase in [
        "end conversation", "end convo", "stop conversation", "stop convo",
        "exit conversation", "exit convo", "quit conversation", "quit convo",
        "goodbye", "bye", "see you later", "talk to you later"
    ])

def is_command(text):
    """Check if the text contains a command.
    
    Args:
        text: Text to check
        
    Returns:
        True if the text contains a command, False otherwise
    """
    if not text:
        return False
        
    text_lower = text.lower()
    
    # Check for command prefix
    return command_prefix in text_lower

def extract_command(text):
    """Extract command and arguments from text.
    
    Args:
        text: Text to extract command from
        
    Returns:
        Tuple of (command_name, command_args)
    """
    if not text:
        return None, None
        
    text_lower = text.lower()
    
    # Check if the text contains the command prefix
    if command_prefix not in text_lower:
        return None, None
        
    # Extract the command and arguments
    command_parts = text_lower.split(command_prefix, 1)[1].strip().split(" ", 1)
    command_name = command_parts[0].strip()
    command_args = command_parts[1].strip() if len(command_parts) > 1 else None
    
    return command_name, command_args 