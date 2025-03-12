#!/usr/bin/env python3
"""
Command execution module for the voice assistant.

This module contains functions for executing commands.
"""

# Standard library imports
import time
import logging
from datetime import datetime
import re

# Import from other modules
from utils import logger, available_commands
from tools import WebBrowser
from ollama_client import OllamaClient

class CommandExecutor:
    """Command executor for the voice assistant."""
    
    def __init__(self):
        """Initialize the command executor."""
        self.browser = WebBrowser()
        self.available_commands = available_commands
        # Store the last search results for retrieval
        self.last_search_results = []
        self.last_search_query = ""
        # Initialize Ollama client for summarization
        self.ollama = OllamaClient()
    
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
        elif command_name == "details":
            return self._execute_details(command_args)
        
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
            self.last_search_query = search_query
            
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
            
            # Extract results
            results = search_result.get('results', [])
            
            # Store the results for later retrieval
            self.last_search_results = results
            
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
            
            # Add instructions for getting more details
            response += "To get more details about a specific result, say 'execute details [number]' or 'execute details [title]'."
            
            return response
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return f"I encountered an error while searching for '{query}'. Please try again later."
    
    def _clean_html_content(self, html_content):
        """Clean HTML content to extract meaningful text.
        
        Args:
            html_content: HTML content to clean
            
        Returns:
            Cleaned text content
        """
        # Remove script and style elements
        html_content = re.sub(r'<script[^>]*>.*?</script>', ' ', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<style[^>]*>.*?</style>', ' ', html_content, flags=re.DOTALL)
        
        # Remove HTML comments
        html_content = re.sub(r'<!--.*?-->', ' ', html_content, flags=re.DOTALL)
        
        # Remove all HTML tags
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Remove extra whitespace
        html_content = re.sub(r'\s+', ' ', html_content).strip()
        
        # Remove common JavaScript artifacts
        html_content = re.sub(r'function\s*\(.*?\)\s*\{.*?\}', ' ', html_content, flags=re.DOTALL)
        html_content = re.sub(r'var\s+\w+\s*=', ' ', html_content)
        html_content = re.sub(r'if\s*\(.*?\)\s*\{.*?\}', ' ', html_content, flags=re.DOTALL)
        
        return html_content
    
    def _execute_details(self, identifier):
        """Execute the details command to get more information about a search result.
        
        Args:
            identifier: Index number or title of the search result
            
        Returns:
            Detailed information about the search result
        """
        if not identifier:
            return "Please specify which search result you'd like more details about, either by number or title."
        
        if not self.last_search_results:
            return "I don't have any search results to provide details for. Please perform a search first."
        
        try:
            # Check if the identifier is a number
            result = None
            if identifier.isdigit():
                index = int(identifier) - 1
                if 0 <= index < len(self.last_search_results):
                    result = self.last_search_results[index]
                else:
                    return f"I don't have a search result with number {identifier}. Please specify a number between 1 and {len(self.last_search_results)}."
            else:
                # Try to find a result with a matching title
                for r in self.last_search_results:
                    if identifier.lower() in r.get('title', '').lower():
                        result = r
                        break
                
                if not result:
                    return f"I couldn't find a search result with a title containing '{identifier}'. Please try again with a different title or use the result number."
            
            # Get more details about the result
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            snippet = result.get('snippet', '')
            
            # Navigate to the URL to get more content
            logger.info(f"Retrieving more details for: {title}")
            
            # Simulate the process
            detail_steps = [
                f"Navigating to {url}...",
                f"Retrieving content...",
                f"Analyzing content..."
            ]
            
            for step in detail_steps:
                logger.info(step)
                time.sleep(0.3)  # Small delay to simulate processing
            
            # Try to navigate to the URL to get more content
            nav_result = self.browser.navigate(url)
            
            if nav_result['status'] == 'error':
                # If navigation fails, just return the snippet
                return f"Details for '{title}':\n\n{snippet}\n\nI couldn't retrieve additional information from {url}."
            
            # Extract content from the page
            content_result = self.browser.extract_content()
            
            if content_result['status'] == 'error':
                # If content extraction fails, just return the snippet
                return f"Details for '{title}':\n\n{snippet}\n\nI couldn't extract additional information from {url}."
            
            # Get the extracted content
            raw_content = content_result.get('content', '')
            
            # Clean the HTML content to get meaningful text
            cleaned_content = self._clean_html_content(raw_content)
            
            # Truncate if still too long for Ollama
            if len(cleaned_content) > 4000:
                cleaned_content = cleaned_content[:4000]
            
            # Use Ollama to summarize the content
            logger.info("Generating summary with Ollama...")
            
            # Prepare the prompt for summarization
            prompt = f"""<|system|>
You are a helpful assistant that summarizes web content. Extract the main narrative content from the text, ignoring navigation elements, ads, and other non-content elements. Focus on the main information and provide a concise summary.
<|user|>
Please summarize the following web content about "{title}":

{cleaned_content}
<|assistant|>
"""
            
            # Generate summary
            summary = self.ollama.generate(prompt)
            
            # Format the response
            response = f"Details for '{title}':\n\n"
            response += f"URL: {url}\n\n"
            
            if summary:
                response += f"Summary: {summary.strip()}\n\n"
            else:
                response += f"Summary: {snippet}\n\n"
            
            response += f"For more information, visit {url}"
            
            return response
        except Exception as e:
            logger.error(f"Error retrieving details: {e}")
            return f"I encountered an error while retrieving details for '{identifier}'. Please try again later." 