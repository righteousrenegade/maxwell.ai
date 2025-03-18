import threading
import logging
from utils import setup_logger
import json
import requests
from bs4 import BeautifulSoup
import re
import html
import datetime
import hashlib

# Get the logger instance
logger = logging.getLogger("maxwell")
if not logger.handlers:
    logger = setup_logger()

# Global cache for search results
search_results_cache = {}

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
            """Searches the web and saves raw content for later retrieval"""
            if not query:
                return "Need a search term."
            
            try:
                # Direct search
                url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
                headers = {"User-Agent": "Mozilla/5.0"}
                
                response = requests.get(url, headers=headers, timeout=15)
                
                # Save the entire response in the cache
                search_id = hashlib.md5(query.encode()).hexdigest()[:8]
                search_results_cache[search_id] = {
                    "query": query,
                    "html": response.text,
                    "urls": []
                }
                
                # Just extract the first 3 URLs for detail retrieval
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all URLs
                result_urls = []
                for t in soup.find_all('title'):
                    # if not a.has_attr('href'):
                    #     continue
                        
                    # href = a['href']
                    # if href.startswith('/url?q='):
                    #     url = href.split('/url?q=')[1].split('&')[0]
                    #     if url.startswith('http') and 'google.com' not in url:
                    #         result_urls.append(url)
                    result = t.get_text(strip=True)
                
                # Take only first 3 URLs
                result_urls.append(result)
                # search_results_cache[search_id]["urls"] = result_urls
                
                # Return basic results
                return f"Results for '{query}':\n\n1. Result 1\n2. Result 2\n3. Result 3\n\nUse 'execute details search result 1/2/3' to see content."
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                return f"Search failed: {e}"

        @self.tool(name="details_search_result", description="Get details of a previous search result")
        def details_search_result(result_number):
            """Gets the content of a URL from search results"""
            try:
                # Check if we have results
                if not search_results_cache:
                    return "No search results. Run 'execute search [query]' first."
                
                # Parse the result number (1, 2, or 3)
                if not result_number or not result_number.isdigit():
                    return "Specify a result number (1, 2, or 3)."
                
                result_num = int(result_number)
                if result_num < 1 or result_num > 3:
                    return "Specify a result number between 1 and 3."
                
                # Get the most recent search
                latest_search = list(search_results_cache.keys())[-1]
                search_data = search_results_cache[latest_search]
                
                # Check if we have URLs
                urls = search_data.get("urls", [])
                if not urls or len(urls) < result_num:
                    return f"No URL for result {result_num}."
                
                # Get the URL for the requested result (0-indexed list)
                url = urls[result_num - 1]
                
                # Fetch the content
                try:
                    page_response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                    
                    if page_response.status_code != 200:
                        return f"Failed to fetch URL (status: {page_response.status_code})"
                    
                    # Parse the content
                    soup = BeautifulSoup(page_response.text, 'html.parser')
                    
                    # Get the page title
                    title = soup.title.string if soup.title else "No title"
                    
                    # Remove scripts, styles, etc.
                    for tag in soup(['script', 'style']):
                        tag.decompose()
                    
                    # Get the text content
                    content = soup.get_text(separator='\n')
                    
                    # Truncate if too long
                    if len(content) > 1500:
                        content = content[:1500] + "...(truncated)"
                    
                    # Return the details
                    return f"Details for result {result_num}:\n\nURL: {url}\nTitle: {title}\n\nContent:\n{content}"
                    
                except Exception as e:
                    logger.error(f"Error fetching URL: {e}")
                    return f"Error fetching URL content: {e}"
                
            except Exception as e:
                logger.error(f"Error retrieving result: {e}")
                return f"Error retrieving result: {e}"
            
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