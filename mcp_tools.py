import threading
import logging
from utils import setup_logger
import json
import requests
import re
import html
import traceback
import sys
import hashlib
from bs4 import BeautifulSoup
import time

try:
    from googleapiclient.discovery import build
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False
    print("Google Search API not available, will use DuckDuckGo only")

# Replace these with your actual keys if you have them
GOOGLE_SEARCH_API_KEY = "AIzaSyD97tA7Lb8YwFvVcqQIZ3xF2eSuRdp_EOE"
GOOGLE_CSE_ID = "a1690cf8f5fd742b9"  

# Get the logger instance
logger = logging.getLogger("maxwell")
if not logger.handlers:
    logger = setup_logger()

# Global cache for search results
search_results_cache = {}

class Tool:
    def __init__(self, name, func, description, params=None):
        self.name = name
        self.func = func
        self.description = description
        self.params = params or []
        
    def execute(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class MCPToolProvider:
    def __init__(self, assistant=None):
        """Initialize the tool provider
        
        The assistant parameter is kept for backward compatibility but no longer stored
        """
        self.tools = {}
        self.register_tools()
        
    def tool(self, name=None, description=None, params=None):
        """Decorator for registering tools"""
        def decorator(func):
            nonlocal name, description, params
            if name is None:
                name = func.__name__
            if description is None:
                description = func.__doc__ or ""
                
            # Extract parameter information from function signature if not provided
            if params is None:
                import inspect
                sig = inspect.signature(func)
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    param_desc = ""
                    param_required = param.default == inspect.Parameter.empty
                    params.append({
                        "name": param_name,
                        "description": param_desc,
                        "required": param_required,
                    })
            
            self.tools[name] = Tool(name, func, description, params)
            return func
        return decorator
        
    def register_tools(self):
        """Register all tools"""
        
        @self.tool(
            name="get_time", 
            description="Get the current time",
            params=[]
        )
        def get_time():
            """Get the current time"""
            import datetime
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}"
            
        @self.tool(
            name="get_date", 
            description="Get the current date",
            params=[]
        )
        def get_date():
            """Get the current date"""
            import datetime
            current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {current_date}"
            
        @self.tool(
            name="tell_joke", 
            description="Tell a random joke",
            params=[]
        )
        def tell_joke():
            """Tell a random joke to lighten the mood"""
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
            
        @self.tool(
            name="search_web", 
            description="Searches the web for information on any topic or question",
            params=[
                {
                    "name": "query",
                    "description": "The search query to look up information about",
                    "required": True
                }
            ]
        )
        def search_web(query):
            """Searches the web for information on any topic or question"""
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
                
                # Initialize result lists
                result_urls = []
                result_titles = []
                result_snippets = []
                result_contents = []
                
                try:
                    # Try using Google Custom Search API first if available
                    if GOOGLE_SEARCH_AVAILABLE:
                        logger.info("Attempting to use Google Search API...")
                        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
                        results = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=3).execute()
                        
                        # Check if we have results
                        if 'items' in results:
                            # Process results
                            for item in results['items']:
                                # Extract URL
                                url = item.get('link')
                                if not url:
                                    continue
                                    
                                # Extract title and snippet
                                title = item.get('title', 'No title')
                                snippet = item.get('snippet', 'No description available')
                                
                                # Add to result lists
                                result_urls.append(url)
                                result_titles.append(title)
                                result_snippets.append(snippet)
                                result_contents.append("")  # Initially empty
                        else:
                            # No results from Google API, switch to fallback
                            logger.warning("No results from Google API, switching to fallback search...")
                            raise Exception("No results from Google API")
                    else:
                        # Google Search not available, use DuckDuckGo directly
                        logger.info("Google Search API not available, using DuckDuckGo directly...")
                        raise Exception("Google Search API not available")
                        
                except Exception as e:
                    # Google Search API failed or not available, use DuckDuckGo as fallback
                    logger.warning(f"Using DuckDuckGo fallback: {str(e)}")
                    print("🦆 Using DuckDuckGo as fallback search engine...")
                    
                    # Use DuckDuckGo fallback
                    duck_results = duckduckgo_search(query, num_results=3)
                    
                    if duck_results:
                        for result in duck_results:
                            result_urls.append(result['link'])
                            result_titles.append(result['title'])
                            result_snippets.append(result['snippet'])
                            result_contents.append("")  # Initially empty
                    else:
                        return f"Sorry, no search results found for '{query}'. Try a different search term."
                
                # If we have no results after all attempts
                if not result_urls:
                    logger.warning(f"No search results found for '{query}'")
                    return f"No search results found for '{query}'. Try a different search term."
                
                # Now try to fetch content for each URL
                for i, url in enumerate(result_urls):
                    try:
                        # Fetch and clean the page content
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
                    
                    # Update the content in our list
                    result_contents[i] = content
                
                # Log what we found
                logger.info(f"Found {len(result_urls)} search results")
                
                # Store in cache
                search_results_cache[search_id] = {
                    "query": query,
                    "urls": result_urls,
                    "titles": result_titles,
                    "snippets": result_snippets,
                    "contents": result_contents,
                    "timestamp": time.time()
                }
                
                # Format the results for display
                formatted_results = []
                for i, (url, title, snippet) in enumerate(zip(result_urls, result_titles, result_snippets)):
                    formatted_results.append(f"{i+1}. {title}\n   {snippet}\n   URL: {url}\n")
                
                # Build the final response
                response = f"Here are the search results for '{query}':\n\n"
                response += "\n".join(formatted_results)
                response += "\n\nFor more details on a specific result, say 'details 1' or 'tell me more about result 2'."
                
                return response
                
            except Exception as e:
                logger.error(f"Error in search_web: {e}")
                logger.error(traceback.format_exc())
                return f"Sorry, I couldn't search for '{query}' due to an error: {str(e)}"
    
    def get_tool_descriptions(self):
        """Get a dictionary of tool names and descriptions"""
        descriptions = {}
        for name, tool in self.tools.items():
            descriptions[name] = tool.description
        return descriptions
    
    def execute_tool(self, tool_name, **kwargs):
        """Execute a tool by name with the given arguments"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
        
        try:
            return self.tools[tool_name].execute(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            logger.error(traceback.format_exc())
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def start_server(self):
        """Start the local HTTP server (dummy method)"""
        logger.info("MCP tool provider started (no server needed)")
        return True
        
    def stop_server(self):
        """Stop the local HTTP server (dummy method)"""
        logger.info("MCP tool provider stopped")
        return True

def duckduckgo_search(query, num_results=5):
    """Perform a search using DuckDuckGo as a fallback search engine.
    
    Args:
        query: The search query
        num_results: Number of results to return (max 10)
        
    Returns:
        A list of search result dictionaries with 'title', 'link', and 'snippet' keys
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        import urllib.parse
        
        # Ensure we get at most 10 results
        num_results = min(num_results, 10)
        
        # Format the query for the URL
        encoded_query = urllib.parse.quote_plus(query)
        
        # Build the search URL
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        # Set a user agent to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"DuckDuckGo search failed with status code {response.status_code}")
            return []
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all search results
        results = []
        result_elements = soup.select('.result')
        
        # Process each result
        for i, element in enumerate(result_elements):
            if i >= num_results:
                break
                
            # Extract the title
            title_element = element.select_one('.result__a')
            title = title_element.get_text() if title_element else 'No title'
            
            # Extract the URL
            link = title_element['href'] if title_element and 'href' in title_element.attrs else ''
            
            # If the URL is relative or contains '/l/?', extract the actual URL
            if link.startswith('/l/?'):
                # Extract the URL from the redirect
                try:
                    link_params = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                    if 'uddg' in link_params:
                        link = link_params['uddg'][0]
                    elif 'u' in link_params:
                        link = link_params['u'][0]
                except:
                    pass
            
            # Ensure the URL is absolute
            if link and not link.startswith(('http://', 'https://')):
                if link.startswith('/'):
                    link = 'https://duckduckgo.com' + link
                else:
                    link = 'https://' + link
            
            # Extract the snippet
            snippet_element = element.select_one('.result__snippet')
            snippet = snippet_element.get_text() if snippet_element else 'No description available'
            
            # Add to results
            results.append({
                'title': title,
                'link': link,
                'snippet': snippet
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {e}")
        logger.error(traceback.format_exc())
        return [] 