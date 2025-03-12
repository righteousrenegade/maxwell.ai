#!/usr/bin/env python3
"""
Command execution module for the voice assistant.

This module contains functions for executing commands.
"""

# Standard library imports
import time
import logging
from datetime import datetime

# Import from other modules
from utils import logger, available_commands
from tools import WebBrowser

class CommandExecutor:
    """Command executor for the voice assistant."""
    
    def __init__(self):
        """Initialize the command executor."""
        self.browser = WebBrowser()
        self.available_commands = available_commands
    
    def execute_command(self, command_name, command_args=None):
        """Execute a command.
        
        Args:
            command_name: Name of the command to execute
            command_args: Arguments for the command
            
        Returns:
            Response text from the command execution
        """
        logger.info(f"Executing command: {command_name} with args: {command_args}")
        
        # Check if the command exists
        if command_name not in self.available_commands:
            return f"Sorry, I don't know how to execute the command '{command_name}'. Available commands are: {', '.join(self.available_commands.keys())}."
        
        # Execute the command based on its name
        if command_name == "weather":
            return self._execute_weather(command_args)
        elif command_name == "time":
            return self._execute_time()
        elif command_name == "date":
            return self._execute_date()
        elif command_name == "news":
            return self._execute_news()
        elif command_name == "joke":
            return self._execute_joke()
        elif command_name == "reminder":
            return self._execute_reminder(command_args)
        elif command_name == "timer":
            return self._execute_timer(command_args)
        elif command_name == "search":
            return self._execute_search(command_args)
        
        # Default response for unknown commands (should not reach here)
        return "I'm not sure how to execute that command."
    
    def _execute_weather(self, location=None):
        """Execute the weather command.
        
        Args:
            location: Optional location to get weather for
            
        Returns:
            Weather information
        """
        # This is a placeholder. In a real implementation, you would call a weather API.
        if location:
            return f"It's currently sunny and 72 degrees Fahrenheit in {location}."
        else:
            return "It's currently sunny and 72 degrees Fahrenheit."
    
    def _execute_time(self):
        """Execute the time command.
        
        Returns:
            Current time
        """
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}."
    
    def _execute_date(self):
        """Execute the date command.
        
        Returns:
            Current date
        """
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}."
    
    def _execute_news(self):
        """Execute the news command.
        
        Returns:
            News headlines
        """
        # This is a placeholder. In a real implementation, you would call a news API.
        return "The top headline today is: Scientists discover new renewable energy source."
    
    def _execute_joke(self):
        """Execute the joke command.
        
        Returns:
            A joke
        """
        # This is a placeholder. In a real implementation, you would have a collection of jokes.
        return "Why don't scientists trust atoms? Because they make up everything!"
    
    def _execute_reminder(self, reminder_text):
        """Execute the reminder command.
        
        Args:
            reminder_text: Text for the reminder
            
        Returns:
            Confirmation message
        """
        if not reminder_text:
            return "Please specify what you'd like me to remind you about."
        return f"I've set a reminder for: {reminder_text}"
    
    def _execute_timer(self, duration):
        """Execute the timer command.
        
        Args:
            duration: Duration for the timer
            
        Returns:
            Confirmation message
        """
        if not duration:
            return "Please specify how long you'd like the timer to be."
        return f"I've set a timer for {duration}."
    
    def _execute_search(self, query):
        """Execute the search command.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        if not query:
            return "Please specify what you'd like me to search for."
        
        try:
            # Use the WebBrowser to perform the search
            search_query = query.strip()
            
            # Print the search steps with small delays to simulate the process
            search_steps = [
                f"Navigating to search engine...",
                f"Entering search query: '{search_query}'",
                f"Processing search results...",
                f"Analyzing top results for relevance..."
            ]
            
            for step in search_steps:
                logger.info(step)
                time.sleep(0.3)  # Small delay to simulate processing
            
            # Perform the actual search
            search_result = self.browser.search(search_query)
            
            if search_result['status'] == 'error':
                logger.error(f"Search error: {search_result.get('error')}")
                return f"I encountered an error while searching for '{search_query}'. Please try again later."
            
            # Format the results
            results = search_result.get('results', [])
            
            if not results:
                return f"I couldn't find any relevant results for '{search_query}'."
            
            # Format the response
            response = f"Here are the search results for '{search_query}':\n\n"
            
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Untitled')
                url = result.get('url', '')
                snippet = result.get('snippet', '')
                
                # Truncate snippet if too long
                if len(snippet) > 150:
                    snippet = snippet[:147] + "..."
                
                response += f"{i}. [{title}]({url})\n"
                response += f"   {snippet}\n\n"
            
            # Add a note about visiting links
            response += "To learn more, you can visit any of these links in your web browser."
            
            return response
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return f"I encountered an error while searching for '{query}'. Please try again later." 