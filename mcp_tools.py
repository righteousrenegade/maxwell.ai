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
import argparse
import sys
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
    def __init__(self, name, func, description, params=None):
        self.name = name
        self.func = func
        self.description = description
        self.params = params or []
        
    def execute(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class MCPToolProvider:
    def __init__(self, assistant):
        self.assistant = assistant
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
            description="Search the web for information on any topic",
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

        @self.tool(
            name="details_search_result", 
            description="Get details of a previous search result",
            params=[
                {
                    "name": "result_number",
                    "description": "The number of the search result to get details for (1, 2, or 3)",
                    "required": True
                }
            ]
        )
        def details_search_result(result_number):
            """Gets detailed information about a search result from a previous web search"""
            try:
                # Check if we have results
                if not search_results_cache:
                    return "No search results. Try searching for something first by saying 'search for [topic]'."
                
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
            
        @self.tool(
            name="set_reminder", 
            description="Set a reminder with optional time",
            params=[
                {
                    "name": "text",
                    "description": "The text of the reminder",
                    "required": True
                },
                {
                    "name": "time_str",
                    "description": "When to be reminded (e.g. '3pm', 'tomorrow at 2pm')",
                    "required": False
                }
            ]
        )
        def set_reminder(text, time_str=None):
            """Set a reminder with optional time specification"""
            if time_str:
                return f"I've set a reminder for '{text}' at {time_str}"
            else:
                return f"I've set a reminder for: {text}"
                
        @self.tool(
            name="set_timer", 
            description="Set a timer for the specified duration",
            params=[
                {
                    "name": "duration",
                    "description": "The duration to set the timer for (e.g. '5 minutes', '30 seconds')",
                    "required": True
                }
            ]
        )
        def set_timer(duration):
            """Set a timer for the specified duration"""
            return f"Timer set for {duration}"
            
        @self.tool(
            name="get_weather", 
            description="Get the weather forecast for a location",
            params=[
                {
                    "name": "location",
                    "description": "The location to get weather for",
                    "required": False
                }
            ]
        )
        def get_weather(location="current location"):
            """Get the weather forecast for a specific location or your current location"""
            return f"The weather in {location} is sunny with a high of 75°F (This would connect to a weather API in a real implementation)"
            
        @self.tool(
            name="play_music", 
            description="Play music based on song, artist, or genre",
            params=[
                {
                    "name": "song",
                    "description": "The name of the song to play",
                    "required": False
                },
                {
                    "name": "artist",
                    "description": "The name of the artist to play music from",
                    "required": False
                },
                {
                    "name": "genre",
                    "description": "The genre of music to play",
                    "required": False
                }
            ]
        )
        def play_music(song=None, artist=None, genre=None):
            """Play music based on song title, artist name, or genre"""
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
                
        @self.tool(
            name="send_message", 
            description="Send a message to a contact",
            params=[
                {
                    "name": "recipient",
                    "description": "The person to send the message to",
                    "required": True
                },
                {
                    "name": "message",
                    "description": "The message content to send",
                    "required": True
                }
            ]
        )
        def send_message(recipient, message):
            """Send a text message to a contact"""
            return f"Message sent to {recipient}: '{message}'"
            
        @self.tool(
            name="list_tools",
            description="List all available tools and their descriptions",
            params=[]
        )
        def list_tools():
            """List all available tools that can be used"""
            tool_list = []
            for name, tool in self.tools.items():
                if name == "list_tools":  # Skip listing itself to avoid recursion
                    continue
                    
                params_info = ""
                if tool.params:
                    required_params = [p["name"] for p in tool.params if p.get("required")]
                    optional_params = [p["name"] for p in tool.params if not p.get("required")]
                    
                    if required_params:
                        params_info += f"\n   Required parameters: {', '.join(required_params)}"
                    if optional_params:
                        params_info += f"\n   Optional parameters: {', '.join(optional_params)}"
                
                tool_list.append(f"• {name}: {tool.description}{params_info}")
            
            return "Available tools:\n\n" + "\n\n".join(tool_list) + "\n\nTo use a tool, you can ask naturally. For example, 'search for climate change' or 'tell me a joke'."
            
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
            
    def detect_tool_listing_request(self, query):
        """Detect if the query is asking to list available tools"""
        listing_patterns = [
            r'(?:list|tell|show)(?:\s+me)?\s+(?:all\s+|available\s+)?tools',
            r'what(?:\s+tools|\s+commands|\s+functions)(?:\s+can\s+|\s+)(you|i)(?:\s+use|\s+do)?',
            r'what\s+can\s+you\s+do',
            r'help(?:\s+me)?(?:\s+with\s+tools)?',
            r'available\s+(?:tools|commands|functions)'
        ]
        
        for pattern in listing_patterns:
            if re.search(pattern, query.lower()):
                return True
        return False
    
    def handle_query(self, query):
        """Handle natural language queries and route to tools if appropriate"""
        # First check if this is a request to list tools
        if self.detect_tool_listing_request(query):
            return self.tools["list_tools"].execute()
            
        # Here you would add logic to detect other tool invocations
        # For example, detecting "search for X" to route to search_web
        # This would typically be handled by the LLM integration
        
        return None  # Return None if no tool match is found
            
    def get_tool_descriptions(self):
        """Get descriptions of all available tools"""
        return {name: tool.description for name, tool in self.tools.items()}
    
    def get_tool_schemas(self):
        """Get JSON schemas for all tools, compatible with MCP protocol"""
        schemas = []
        for name, tool in self.tools.items():
            schema = {
                "name": name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Add parameter definitions
            for param in tool.params:
                param_name = param["name"]
                param_desc = param["description"]
                schema["parameters"]["properties"][param_name] = {
                    "type": "string",
                    "description": param_desc
                }
                
                # Add to required list if needed
                if param.get("required", False):
                    schema["parameters"]["required"].append(param_name)
            
            schemas.append(schema)
        
        return schemas
        
    def start_server(self):
        """Start the tool provider (no server needed in this simplified version)"""
        logger.info("Tool provider initialized")
        
    def stop_server(self):
        """Stop the tool provider (no server needed in this simplified version)"""
        logger.info("Tool provider stopped")
        
    def register_with_mcp(self):
        """Register these tools with the MCP system for natural language invocation"""
        try:
            from mcp.server.fastmcp import FastMCP
            
            mcp_server = FastMCP("Maxwell Tool Provider")
            
            # First, register the list_tools tool explicitly to make it easily discoverable
            list_tool = self.tools.get("list_tools")
            if list_tool:
                @mcp_server.tool(
                    name="list_tools",
                    description="List all available tools and what they can do"
                )
                def mcp_list_tools():
                    return list_tool.execute()
            
            # Register each tool with the MCP server
            for name, tool in self.tools.items():
                # Skip list_tools as we've already registered it above
                if name == "list_tools":
                    continue
                    
                # Create a closure to capture the tool object
                def create_tool_wrapper(tool_obj):
                    # Extract param info for proper registration
                    param_dict = {}
                    for param in tool_obj.params:
                        param_name = param["name"]
                        param_desc = param["description"]
                        param_dict[param_name] = (param_desc, not param.get("required", False))
                    
                    # Define the wrapper function with all parameters
                    def wrapper(**kwargs):
                        return tool_obj.execute(**kwargs)
                    
                    # Add param annotations
                    wrapper.__annotations__ = {p["name"]: str for p in tool_obj.params}
                    return wrapper, param_dict
                
                # Get the wrapper and param info
                wrapper_func, param_info = create_tool_wrapper(tool)
                wrapper_func.__name__ = name
                wrapper_func.__doc__ = tool.description
                
                # Register with appropriate parameters
                mcp_server.tool(
                    name=name,
                    description=tool.description,
                    **param_info
                )(wrapper_func)
            
            # Add a system prompt help message to guide the LLM in using tools
            @mcp.prompt()
            def help_message():
                """Guide for using available tools"""
                return """
                I have access to various tools that can help you. If you want to know what tools are available,
                just ask me "what tools can you use?" or "list available tools."
                
                You can then use these tools by simply asking naturally. For example:
                - "Search for climate change news"
                - "Tell me a joke"
                - "What time is it?"
                - "Set a reminder for my meeting at 3pm"
                
                I'll automatically detect your request and use the appropriate tool.
                """
            
            logger.info(f"Registered {len(self.tools)} tools with MCP")
            return mcp_server
            
        except ImportError as e:
            logger.error(f"MCP library not found. Install with 'pip install mcp': {e}")
            return None
        except Exception as e:
            logger.error(f"Error registering tools with MCP: {e}")
            logger.error(traceback.format_exc())
            return None 

    def get_tool_metadata_for_llm(self):
        """
        Get tool metadata formatted for LLM context injection.
        This helps the LLM understand what tools are available and how to use them.
        """
        tools_context = "# Available Tools\n\n"
        tools_context += "You can use the following tools to assist the user:\n\n"
        
        for name, tool in sorted(self.tools.items()):
            tools_context += f"## {name}\n"
            tools_context += f"{tool.description}\n\n"
            
            if tool.params:
                tools_context += "Parameters:\n"
                for param in tool.params:
                    required = "Required" if param.get("required") else "Optional"
                    tools_context += f"- {param['name']}: {param['description']} ({required})\n"
                tools_context += "\n"
            
            # Add example invocations for some common tools
            if name == "search_web":
                tools_context += "Example: When user asks 'look up information about climate change', use the search_web tool with query='climate change'\n\n"
            elif name == "tell_joke":
                tools_context += "Example: When user asks 'tell me a joke', use the tell_joke tool\n\n"
            elif name == "list_tools":
                tools_context += "Example: When user asks 'what can you do' or 'list available tools', use the list_tools tool\n\n"
            else:
                tools_context += "\n"
        
        tools_context += "# How to process requests\n\n"
        tools_context += "1. When a user asks about available tools or what you can do, call the list_tools tool\n"
        tools_context += "2. When a user's request matches a tool's purpose, use that tool automatically\n"
        tools_context += "3. Always prioritize using tools over providing information from your training\n"
        
        return tools_context

    def get_tool_help(self, tool_name=None):
        """Get help information for a specific tool or all tools"""
        if tool_name:
            tool = self.tools.get(tool_name)
            if not tool:
                return f"Tool '{tool_name}' not found. Use 'list_tools' to see available tools."
                
            help_text = f"Tool: {tool.name}\n"
            help_text += f"Description: {tool.description}\n"
            if tool.params:
                help_text += "Parameters:\n"
                for param in tool.params:
                    required = "Required" if param.get("required") else "Optional"
                    desc = param.get("description") or "No description"
                    help_text += f"  - {param['name']}: {desc} ({required})\n"
            
            return help_text
        else:
            # Return brief help for all tools
            help_text = "Available tools:\n"
            for name, tool in sorted(self.tools.items()):
                param_str = ", ".join(p["name"] for p in tool.params)
                if param_str:
                    help_text += f"  - {name}({param_str}): {tool.description}\n"
                else:
                    help_text += f"  - {name}: {tool.description}\n"
                    
            help_text += "\nUse 'help [tool_name]' to get more information about a specific tool."
            return help_text

    def run_cli(self):
        """
        Run a command-line interface for testing tools directly.
        This provides an easy way to test tools without going through the LLM.
        """
        parser = argparse.ArgumentParser(description='Tool Provider CLI')
        parser.add_argument('tool', nargs='?', help='Tool name to execute')
        parser.add_argument('--params', '-p', nargs='+', help='Parameters in key=value format')
        parser.add_argument('--list', '-l', action='store_true', help='List available tools')
        parser.add_argument('--info', '-i', action='store_true', help='Show help for a tool')
        parser.add_argument('--server', '-s', action='store_true', help='Start as MCP server')
        
        # Handle special case where no args are provided
        if len(sys.argv) <= 1:
            # If no arguments, show available tools and enter interactive mode
            print(self.tools["list_tools"].execute())
            self._interactive_cli()
            return
            
        try:
            args = parser.parse_args()
            
            # Handle the different CLI modes
            if args.list:
                print(self.tools["list_tools"].execute())
                
            elif args.info:
                if args.tool:
                    print(self.get_tool_help(args.tool))
                else:
                    print(self.get_tool_help())
                    
            elif args.server:
                # Start as MCP server
                print("Starting MCP server...")
                server = self.register_with_mcp()
                if server:
                    server.run()
                else:
                    print("Failed to start MCP server. Make sure the MCP package is installed.")
                    
            elif args.tool:
                # Execute the specified tool
                if args.tool not in self.tools:
                    print(f"Unknown tool: {args.tool}")
                    print(self.get_tool_help())
                    return
                    
                # Parse parameters
                params = {}
                if args.params:
                    for param in args.params:
                        if '=' in param:
                            key, value = param.split('=', 1)
                            params[key] = value
                        else:
                            # If no key provided, try to match to the first required parameter
                            tool = self.tools[args.tool]
                            for tool_param in tool.params:
                                if tool_param.get("required") and tool_param["name"] not in params:
                                    params[tool_param["name"]] = param
                                    break
                                    
                # Execute the tool
                try:
                    result = self.execute_tool(args.tool, **params)
                    print(result)
                except Exception as e:
                    print(f"Error executing {args.tool}: {e}")
                    traceback.print_exc()
                    
            else:
                # No arguments provided, show help
                print(self.get_tool_help())
                
        except argparse.ArgumentError as e:
            print(f"Argument error: {e}")
            print("Try using --info instead of --help to get tool information")
            parser.print_help()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

    def _interactive_cli(self):
        """Run an interactive CLI session for testing tools"""
        print("\n=== Interactive Tool Testing Mode ===")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'help' to see available tools")
        print("Type 'help [tool_name]' for specific tool help")
        print("Type '[tool_name] [param1=value1] [param2=value2]...' to execute a tool")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() in ('exit', 'quit'):
                    break
                    
                if not command:
                    continue
                    
                if command.lower() == 'help':
                    print(self.get_tool_help())
                    continue
                    
                if command.lower().startswith('help '):
                    tool_name = command[5:].strip()
                    print(self.get_tool_help(tool_name))
                    continue
                
                # Check if it's asking to list tools
                if self.detect_tool_listing_request(command):
                    print(self.tools["list_tools"].execute())
                    continue
                    
                # Parse the command to extract tool name and parameters
                parts = command.split()
                tool_name = parts[0]
                
                if tool_name not in self.tools:
                    print(f"Unknown tool: {tool_name}")
                    print("Type 'help' to see available tools")
                    continue
                    
                # Extract parameters
                params = {}
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key] = value
                    else:
                        # If no key provided, try to match to the first required parameter
                        tool = self.tools[tool_name]
                        for tool_param in tool.params:
                            if tool_param.get("required") and tool_param["name"] not in params:
                                params[tool_param["name"]] = part
                                break
                                
                # Execute the tool
                try:
                    result = self.execute_tool(tool_name, **params)
                    print(result)
                except Exception as e:
                    print(f"Error executing {tool_name}: {e}")
                    traceback.print_exc()
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()

# Modified main function to work as both module and script
def main():
    """
    Run the tool provider in command-line mode.
    
    This can be used in three ways:
    1. As a command-line tool for testing: python mcp_tools.py [tool_name] [params]
    2. As an interactive CLI: python mcp_tools.py 
    3. As an MCP server: python mcp_tools.py --server
    """
    try:
        # Create a simple assistant mock for testing
        class MockAssistant:
            def __init__(self):
                self.name = "Test Assistant"
                
            def summarize(self, content, instructions):
                return f"Summary: {content[:50]}..." if content else "No content to summarize"
        
        # Create the tool provider
        tool_provider = MCPToolProvider(MockAssistant())
        
        # Run the CLI
        tool_provider.run_cli()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error running tool provider: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 