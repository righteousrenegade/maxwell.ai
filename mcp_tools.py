import threading
import logging
from utils import setup_logger
import json

logger = setup_logger()

class Tool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description
        
    def execute(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class MCPToolProvider:
    def __init__(self, assistant):
        self.assistant = assistant
        self.tools = {}
        self.register_tools()
        
    def tool(self, name=None, description=None):
        """Decorator for registering tools"""
        def decorator(func):
            nonlocal name, description
            if name is None:
                name = func.__name__
            if description is None:
                description = func.__doc__ or ""
                
            self.tools[name] = Tool(name, func, description)
            return func
        return decorator
        
    def register_tools(self):
        """Register all tools"""
        
        @self.tool(name="get_time", description="Get the current time")
        def get_time():
            import datetime
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}"
            
        @self.tool(name="get_date", description="Get the current date")
        def get_date():
            import datetime
            current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {current_date}"
            
        @self.tool(name="tell_joke", description="Tell a random joke")
        def tell_joke():
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                "What do you call a fake noodle? An impasta!",
                "How does a penguin build its house? Igloos it together!",
                "Why don't eggs tell jokes? They'd crack each other up!",
                "I'm reading a book about anti-gravity. It's impossible to put down!",
                "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them!"
            ]
            import random
            return random.choice(jokes)
            
        @self.tool(name="search_web", description="Search the web for information")
        def search_web(query):
            return f"Searching for: {query}... (This would connect to a search API in a real implementation)"
            
        @self.tool(name="set_reminder", description="Set a reminder with optional time")
        def set_reminder(text, time_str=None):
            if time_str:
                return f"I've set a reminder for '{text}' at {time_str}"
            else:
                return f"I've set a reminder for: {text}"
                
        @self.tool(name="set_timer", description="Set a timer for the specified duration")
        def set_timer(duration):
            return f"Timer set for {duration}"
            
        @self.tool(name="get_weather", description="Get the weather forecast")
        def get_weather(location="current location"):
            return f"The weather in {location} is sunny with a high of 75°F (This would connect to a weather API in a real implementation)"
            
        @self.tool(name="play_music", description="Play music based on song, artist, or genre")
        def play_music(song=None, artist=None, genre=None):
            if song and artist:
                return f"Playing '{song}' by {artist}"
            elif song:
                return f"Playing '{song}'"
            elif artist:
                return f"Playing music by {artist}"
            elif genre:
                return f"Playing {genre} music"
            else:
                return "Playing some music for you"
                
        @self.tool(name="send_message", description="Send a message to a contact")
        def send_message(recipient, message):
            return f"Message sent to {recipient}: '{message}'"
            
    def execute_tool(self, tool_name, **kwargs):
        """Execute a tool by name with arguments"""
        if tool_name in self.tools:
            try:
                return self.tools[tool_name].execute(**kwargs)
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return f"Error executing {tool_name}: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"
            
    def get_tool_descriptions(self):
        """Get descriptions of all available tools"""
        return {name: tool.description for name, tool in self.tools.items()}
        
    def start_server(self):
        """Start the tool provider (no server needed in this simplified version)"""
        logger.info("Tool provider initialized")
        
    def stop_server(self):
        """Stop the tool provider (no server needed in this simplified version)"""
        logger.info("Tool provider stopped") 