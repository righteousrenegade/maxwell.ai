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
import traceback
from googleapiclient.discovery import build

GOOGLE_SEARCH_API_KEY = "AIzaSyD97tA7Lb8YwFvVcqQIZ3xF2eSuRdp_EOE"
GOOGLE_CSE_ID = "a1690cf8f5fd742b9"  # This is a placeholder, replace with your actual CSE ID


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
            """Searches the web using Google Custom Search API and saves results for later retrieval"""
            if not query:
                return "Need a search term."
            
            try:
                logger.info(f"Searching for: '{query}'")
                
                # Create search ID and initialize cache
                search_id = hashlib.md5(query.encode()).hexdigest()[:8]
                search_results_cache[search_id] = {
                    "query": query,
                    "urls": [],
                    "titles": [],
                    "snippets": [],
                    "contents": []  # New field to store cleaned content
                }
                
                # Use Google Custom Search API
                service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
                results = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=3).execute()
                
                # Check if we have results
                if 'items' not in results:
                    logger.warning(f"No search results found for '{query}'")
                    return f"No search results found for '{query}'. Try a different search term."
                
                # Initialize result lists
                result_urls = []
                result_titles = []
                result_snippets = []
                result_contents = []
                
                # Process results
                for item in results['items']:
                    # Extract URL
                    url = item.get('link')
                    if not url:
                        continue
                    
                    # Extract title
                    title = item.get('title', 'No title available')
                    
                    # Extract snippet
                    snippet = item.get('snippet', 'No description available')
                    
                    # Fetch and clean the page content
                    try:
                        page_response = requests.get(url, timeout=15, headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        })
                        
                        if page_response.status_code == 200:
                            # Parse the content
                            soup = BeautifulSoup(page_response.text, 'html.parser')
                            
                            # Remove scripts, styles, and other non-content elements
                            for element in soup(['script', 'style', 'meta', 'noscript', 'header', 'footer', 'nav', 'iframe', 'svg']):
                                element.decompose()
                            
                            # Try to find main content
                            main_content = None
                            for tag in ['main', 'article', 'div[role="main"]', '.main-content', '#content', '.content', '.post-content']:
                                main_element = soup.select_one(tag)
                                if main_element:
                                    main_content = main_element.get_text(separator='\n', strip=True)
                                    break
                            
                            # If no main content found, use the body text
                            if not main_content:
                                main_content = soup.body.get_text(separator='\n', strip=True) if soup.body else soup.get_text(separator='\n', strip=True)
                            
                            # Clean up the content
                            lines = [line.strip() for line in main_content.split('\n') if line.strip()]
                            content = '\n'.join(lines)
                            
                            # Truncate if too long
                            if len(content) > 2000:
                                content = content[:2000] + "...(truncated)"
                        else:
                            content = f"Failed to fetch content (status: {page_response.status_code})"
                    except Exception as e:
                        logger.error(f"Error fetching URL content: {e}")
                        content = f"Error fetching content: {str(e)}"
                    
                    # Add to results
                    result_urls.append(url)
                    result_titles.append(title)
                    result_snippets.append(snippet)
                    result_contents.append(content)
                
                # Log what we found
                logger.info(f"Found {len(result_urls)} search results")
                
                # Update the cache
                search_results_cache[search_id]["urls"] = result_urls
                search_results_cache[search_id]["titles"] = result_titles
                search_results_cache[search_id]["snippets"] = result_snippets
                search_results_cache[search_id]["contents"] = result_contents
                
                # Check if we found any results
                if not result_urls:
                    logger.warning("No search results found")
                    return f"No search results found for '{query}'. Try a different search term."
                
                # Create formatted results with just titles for speaking
                results_text = f"I found these results for '{query}':\n\n"
                for i in range(len(result_urls)):
                    results_text += f"{i+1}. {result_titles[i]}\n"
                
                results_text += "\nYou can say 'details 1', 'details 2', or 'details 3' to learn more."
                
                return results_text
            except Exception as e:
                logger.error(f"Search error: {e}")
                logger.error(traceback.format_exc())
                return f"Search failed: {e}"

        @self.tool(name="details_search_result", description="Get details of a previous search result")
        def details_search_result(result_number):
            """Gets the content of a URL from search results and provides a summary"""
            try:
                # Check if we have results
                if not search_results_cache:
                    return "No search results. Run 'execute search [query]' first."
                
                # Clean the result number input - extract the first digit
                cleaned_number = ""
                for char in str(result_number):
                    if char.isdigit():
                        cleaned_number += char
                        break  # Just take the first digit
                
                # If we couldn't extract a digit, give a helpful message
                if not cleaned_number:
                    return "Please specify a result number (1, 2, or 3)."
                
                result_num = int(cleaned_number)
                if result_num < 1 or result_num > 3:
                    return "Please specify a result number between 1 and 3."
                
                # Get the most recent search
                latest_search = list(search_results_cache.keys())[-1]
                search_data = search_results_cache[latest_search]
                
                # Check if we have URLs
                urls = search_data.get("urls", [])
                titles = search_data.get("titles", [])
                contents = search_data.get("contents", [])
                
                if not urls or len(urls) < result_num:
                    return f"No URL for result {result_num}."
                
                # Get the information for the requested result (0-indexed list)
                url = urls[result_num - 1]
                title = titles[result_num - 1] if result_num <= len(titles) else "No title available"
                
                # Get the content if available
                if contents and len(contents) >= result_num:
                    content = contents[result_num - 1]
                else:
                    content = "Content not available"
                
                # Generate a simple summary instead of using the assistant.summarize method
                summary = ""
                try:
                    # Only attempt to summarize if we have valid content
                    if content and content != "Content not available" and not content.startswith("Failed to fetch") and not content.startswith("Error fetching"):
                        # Extract first few sentences (up to 250 chars) as a simple summary
                        sentences = re.split(r'(?<=[.!?])\s+', content[:500])
                        if sentences and len(sentences) > 0:
                            # Get up to 2 sentences for the summary
                            summary = " ".join(sentences[:min(2, len(sentences))])
                            if len(summary) > 250:
                                summary = summary[:247] + "..."
                    else:
                        summary = "No summary available."
                except Exception as e:
                    logger.error(f"Error generating simple summary: {e}")
                    summary = "No summary available."
                
                # Prepare the response - format optimized for speech
                # Use a more natural response format to avoid triggering more commands
                response = f"Here's result {result_num}: {title}\n\n"
                if summary:
                    response += f"{summary}\n\n"
                response += f"Source: {url}"
                
                return response
            except Exception as e:
                logger.error(f"Error retrieving result: {e}")
                logger.error(traceback.format_exc())
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