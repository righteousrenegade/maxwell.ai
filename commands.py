import requests
import datetime
import json
import threading
import time
import logging
import random
import socket
import traceback

from utils import setup_logger
from llm_provider import create_llm_provider, LLMProvider

# Try to import OpenAI client - don't fail if it's not available
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️ OpenAI client not available. Install with 'pip install openai'")

# Get the logger instance
logger = logging.getLogger("maxwell")
if not logger.handlers:
    logger = setup_logger()

# Dictionary mapping natural language patterns to command names
COMMAND_PATTERNS = {
    "what time is it": "get_time",
    "what's the time": "get_time", 
    "tell me the time": "get_time",
    "what day is it": "get_date",
    "what's the date": "get_date",
    "tell me the date": "get_date",
    "tell me a joke": "tell_joke",
    "what's the weather": "get_weather",
    "how's the weather": "get_weather",
    "search for": "search_web",
    "look up": "search_web",
    "what is": "search_web",
    "who is": "search_web",
    "remind me to": "set_reminder",
    "set a timer for": "set_timer",
    "timer for": "set_timer"
}

class CommandExecutor:
    """Command executor that handles direct commands and LLM integration with MCP tools support"""
    
    def __init__(self, assistant=None, config=None):
        """Initialize the command executor with configuration and llm settings"""
        self.assistant = assistant
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage for timers and reminders
        self.timers = {}
        self.reminders = {}
        
        # Initialize tools and mcp_tools dictionaries
        self.available_commands = {}
        self.mcp_tools = {}
        
        # Check for MCP tools integration
        self.use_mcp = getattr(config, 'use_mcp', False) if config else False
        
        # Register basic commands that don't require LLM
        self._register_basic_commands()
        
        # If MCP tools are available, register them
        if self.use_mcp and hasattr(assistant, 'mcp_tool_provider') and assistant.mcp_tool_provider:
            logger.info("MCP tools available, registering with CommandExecutor")
            self._register_mcp_tools()
        
        # Set up the LLM provider 
        self.llm_provider = None
        self.ollama_available = False
        self._setup_llm_provider()
        
        # Log initialization
        logger.info("CommandExecutor initialized")
        
    def _register_mcp_tools(self):
        """Register available MCP tools as commands"""
        try:
            if not hasattr(self.assistant, 'mcp_tool_provider') or not self.assistant.mcp_tool_provider:
                logger.warning("MCP tools requested but not available in assistant")
                return
                
            # Get available tools from the MCP tool provider
            available_tools = self.assistant.mcp_tool_provider.get_tool_descriptions()
            logger.info(f"Found {len(available_tools)} MCP tools")
            
            # Create command wrappers for each MCP tool
            for tool_name, description in available_tools.items():
                logger.info(f"Registering MCP tool as command: {tool_name}")
                
                # Store the tool in our mcp_tools dictionary
                self.mcp_tools[tool_name] = {
                    'name': tool_name,
                    'description': description
                }
                
                # Create a wrapper function for this tool
                def create_tool_wrapper(t_name):
                    def wrapper(args=""):
                        return self._execute_mcp_tool(t_name, args)
                    return wrapper
                
                # Register the wrapper function as a command
                self.available_commands[tool_name] = create_tool_wrapper(tool_name)
                
                # Also register simplified aliases for common tools
                tool_aliases = {
                    'get_time': 'time',
                    'get_date': 'date',
                    'tell_joke': 'joke',
                    'get_weather': 'weather',
                    'search_web': 'search'
                }
                
                if tool_name in tool_aliases:
                    alias = tool_aliases[tool_name]
                    logger.info(f"Adding alias '{alias}' for tool '{tool_name}'")
                    self.available_commands[alias] = create_tool_wrapper(tool_name)
                
        except Exception as e:
            logger.error(f"Error registering MCP tools: {e}")
            logger.error(traceback.format_exc())
    
    def _execute_mcp_tool(self, tool_name, args=""):
        """Execute an MCP tool with the given arguments"""
        logger.info(f"Executing MCP tool: {tool_name} with args: {args}")
        
        try:
            # Check if the tool is available
            if not hasattr(self.assistant, 'mcp_tool_provider') or not self.assistant.mcp_tool_provider:
                logger.warning(f"MCP tool execution requested but MCP provider not available: {tool_name}")
                return f"I'm sorry, tool {tool_name} is not available."
            
            # Parse arguments based on tool type
            kwargs = {}
            
            # Special handling for different tools
            if tool_name == "search_web":
                kwargs["query"] = args
            elif tool_name == "details_search_result":
                kwargs["result_number"] = args
            elif tool_name == "get_weather":
                if args:
                    kwargs["location"] = args
            elif tool_name == "set_reminder":
                kwargs["text"] = args
            elif tool_name == "set_timer":
                kwargs["duration"] = args
            elif tool_name == "play_music":
                if " by " in args:
                    song, artist = args.split(" by ", 1)
                    kwargs["song"] = song.strip()
                    kwargs["artist"] = artist.strip()
                elif args:
                    kwargs["song"] = args
                    
            # Execute the tool through the assistant's mcp_tool_provider
            result = self.assistant.mcp_tool_provider.execute_tool(tool_name, **kwargs)
            logger.info(f"Tool execution result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            logger.error(traceback.format_exc())
            return f"Sorry, I encountered an error executing {tool_name}: {str(e)}"

    def execute(self, command_text):
        """Execute a command, either a direct command or through the LLM"""
        if not command_text:
            return "I didn't receive any command to execute."
            
        logger.info(f"Executing: {command_text}")
        
        # Special case for execute commands
        if command_text.lower().startswith("execute "):
            # Extract the actual command
            command_text = command_text[8:].strip()
            logger.info(f"Execute prefix detected, processing as direct command: {command_text}")
            return self.execute_command(command_text)
        
        # Special case for search commands - prioritize MCP tools
        if (command_text.lower().startswith("search ") or "search for" in command_text.lower()) and self.use_mcp:
            search_term = ""
            if command_text.lower().startswith("search "):
                search_term = command_text[7:].strip()
            elif "search for" in command_text.lower():
                search_term = command_text.lower().split("search for", 1)[1].strip()
                
            if search_term and hasattr(self.assistant, 'mcp_tool_provider'):
                logger.info(f"Search command detected, using MCP search_web tool: {search_term}")
                return self._execute_mcp_tool("search_web", search_term)
        
        # Try to match a direct command first
        # Split into command and arguments
        parts = command_text.split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        # Check for direct command match
        if command in self.available_commands:
            logger.info(f"Direct command match: {command}")
            try:
                return self.available_commands[command](args)
            except Exception as e:
                logger.error(f"Error executing command '{command}': {e}")
                logger.error(traceback.format_exc())
                return f"Sorry, I encountered an error executing that command: {str(e)}"
        
        # Check for command aliases and patterns
        for cmd_pattern, cmd_name in COMMAND_PATTERNS.items():
            if cmd_pattern in command_text.lower():
                if cmd_name in self.available_commands:
                    logger.info(f"Command pattern match: {cmd_pattern} -> {cmd_name}")
                    try:
                        # Extract relevant arguments if needed
                        pattern_args = command_text.lower().replace(cmd_pattern, "").strip()
                        return self.available_commands[cmd_name](pattern_args)
                    except Exception as e:
                        logger.error(f"Error executing pattern command '{cmd_name}': {e}")
                        logger.error(traceback.format_exc())
                        return f"Sorry, I encountered an error processing that request: {str(e)}"
        
        # Check if the command starts with 'search' or contains search-like patterns
        if command == "search" or "search for" in command_text.lower():
            search_term = args
            if "search for" in command_text.lower():
                search_term = command_text.lower().replace("search for", "", 1).strip()
            logger.info(f"Search command detected: {search_term}")
            
            # Use MCP search_web tool if available
            if 'search_web' in self.mcp_tools:
                return self._execute_mcp_tool('search_web', search_term)
        
        # No direct command match, use LLM if available
        # Fixed logic to properly check LLM availability
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                # Check if LLM connection is working
                if not hasattr(self, 'llm_available') or not self.llm_available:
                    # Try to re-check the connection
                    logger.info("LLM connection might be down, checking...")
                    self.llm_available = self.check_llm_connection(force=True)
                
                if self.llm_available:
                    logger.info("Using LLM for query")
                    return self.query_llm(command_text)
                else:
                    logger.warning("LLM connection failed")
                    return "I'm sorry, I can't connect to the language model right now. Please try using direct commands instead."
            except Exception as e:
                logger.error(f"Error querying LLM: {e}")
                logger.error(traceback.format_exc())
                return f"Sorry, I encountered an error processing your request with the language model: {str(e)}"
        else:
            # No LLM available
            logger.warning("No LLM provider initialized")
            return "I'm sorry, I don't have a language model available and I don't recognize that as a direct command."

    def execute_command(self, command_text):
        """Directly execute a command without using the LLM"""
        if not command_text:
            return self.show_help("")
            
        logger.info(f"Direct command execution: {command_text}")
        
        # Split into command and arguments
        parts = command_text.split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        # Special case for help command
        if command == "help":
            return self.show_help(args)
            
        # Special case for search command - direct to MCP search_web tool
        if command == "search" and self.use_mcp and hasattr(self.assistant, 'mcp_tool_provider'):
            logger.info(f"Search command with args: '{args}'")
            return self._execute_mcp_tool("search_web", args)

        # Special case for details search result command
        if command == "details" and "result" in args and self.use_mcp:
            try:
                # Extract the result number
                result_number = ""
                for char in args:
                    if char.isdigit():
                        result_number += char
                
                if result_number:
                    logger.info(f"Details search result command for result number: {result_number}")
                    return self._execute_mcp_tool("details_search_result", result_number)
                else:
                    logger.warning("No result number found in details command")
                    return "Please specify a result number, e.g., 'details search result 1'"
            except Exception as e:
                logger.error(f"Error processing details search result command: {e}")
                return f"Error retrieving search result details: {str(e)}"
            
        # Check if the command exists
        if command in self.available_commands:
            try:
                logger.info(f"Executing direct command: {command} with args: {args}")
                return self.available_commands[command](args)
            except Exception as e:
                logger.error(f"Error executing command: {e}")
                logger.error(traceback.format_exc())
                return f"Error executing command: {str(e)}"
        else:
            # Try executing through MCP tools if available
            if self.use_mcp and hasattr(self.assistant, 'mcp_tool_provider'):
                for tool_name in self.mcp_tools:
                    if command == tool_name or command.replace("_", " ") == tool_name.replace("_", " "):
                        return self._execute_mcp_tool(tool_name, args)
            
            available_cmds = sorted(list(self.available_commands.keys()))
            return f"Unknown command: '{command}'. Available commands: {', '.join(available_cmds[:10])}"

    def _register_basic_commands(self):
        """Register basic commands that don't require LLM or are not covered by MCP tools"""
        # Register help command
        self.available_commands["help"] = self.show_help
        
        # Register diagnostic commands
        self.available_commands["check"] = self.diagnostic_check_llm
        self.available_commands["diagnostics"] = self.diagnostic_check_llm
        
        # Add command shortcut for details search result
        self.available_commands["details"] = lambda args: self._handle_details_command(args)
        
        # If there are other commands not handled by MCP tools, add them here
        
        logger.info(f"Registered {len(self.available_commands)} basic commands")

    def _handle_details_command(self, args):
        """Handle the details command for search results"""
        # Support multiple phrasings: 'search result 1', 'search results 1', or just 'result 1'
        if "result" in args:
            # Extract the result number by finding any digits in the string
            result_number = ""
            for char in args:
                if char.isdigit():
                    result_number += char
                    
            if result_number:
                logger.info(f"Extracted result number: {result_number}")
                return self._execute_mcp_tool("details_search_result", result_number)
            else:
                return "Please specify which search result number you want details for, e.g., 'details search result 1'"
        else:
            return "Please specify which search result you want details for, e.g., 'details search result 1'"

    def _setup_llm_provider(self):
        """Set up the LLM provider based on the configuration"""
        try:
            # Get the provider type from config
            self.llm_provider_type = self.config.get('llm_provider', 'none').lower()
            
            # If no provider configured, we're done
            if self.llm_provider_type == 'none':
                logger.info("No LLM provider configured, running in command-only mode")
                self.llm = None
                self.llm_available = False
                return
                
            # Create the provider
            self.llm = create_llm_provider(self.config)
            logger.info(f"Initialized {self.llm_provider_type} provider")
            
            # Set the llm_provider attribute to be consistent with the check in execute()
            self.llm_provider = self.llm
            
            # Check if connection is available
            self.llm_available = self.llm.check_connection()
            self.model = self.llm.get_model_name()
            
            logger.info(f"LLM Provider initialized: {self.llm_provider_type}, " 
                       f"Model: {self.model}, Available: {self.llm_available}")
                       
        except Exception as e:
            logger.error(f"Error initializing LLM provider: {e}")
            logger.error(traceback.format_exc())
            self.llm_available = False
            self.llm = None
            self.llm_provider = None
            print(f"⚠️ Warning: Failed to initialize {self.llm_provider_type} provider: {e}")
    
    def diagnostic_check_llm(self, args=""):
        """Diagnostic command to check the LLM connection status"""
        # Check if an LLM provider is configured
        if not hasattr(self, 'llm') or not self.llm:
            return "No LLM provider is configured. Using direct commands only."
            
        # Force a connection check to the current provider
        connected = self.check_llm_connection(force=True)
        
        if connected:
            # Get information about the provider
            provider_type = self.llm_provider_type or "unknown"
            model = self.model if hasattr(self, 'model') else "unknown"
            
            return (
                f"✅ Connection to {provider_type} LLM is working properly.\n"
                f"📋 Provider: {provider_type}\n"
                f"📋 Model: {model}\n"
                f"📋 Status: Available"
            )
        else:
            # Return information about what failed
            provider_type = self.llm_provider_type or "unknown"
            
            error_suggestions = {
                "ollama": (
                    "- Check if Ollama server is running with 'ollama serve'\n"
                    "- Verify the base URL is correct (default: http://localhost:11434)\n"
                    "- Ensure the requested model is pulled with 'ollama pull MODEL_NAME'"
                ),
                "openai": (
                    "- Check if your API key is valid\n"
                    "- Verify your internet connection\n"
                    "- Ensure the base URL is correct for your service"
                )
            }
            
            suggestions = error_suggestions.get(provider_type, "- Check your connection and try again")
            
            return (
                f"❌ Connection to {provider_type} LLM failed.\n"
                f"📋 Provider: {provider_type}\n"
                f"📋 Status: Unavailable\n\n"
                f"Troubleshooting steps:\n{suggestions}"
            )
            
    def check_llm_connection(self, force=False):
        """Check if an LLM service is available"""
        # If we have no provider, we can't connect
        if not hasattr(self, 'llm') or not self.llm:
            logger.warning("No LLM provider configured, can't check connection")
            self.llm_available = False
            return False
            
        # If not forcing a check and we already know it's available, return True
        if not force and self.llm_available:
            return True
            
        # Otherwise, check the connection using the provider's method
        try:
            self.llm_available = self.llm.check_connection()
            return self.llm_available
        except Exception as e:
            logger.error(f"Error checking LLM connection: {e}")
            logger.error(traceback.format_exc())
            self.llm_available = False
            return False
    
    def query_llm(self, query, retry=True):
        """
        Query the currently selected LLM provider with a prompt
        
        Args:
            query (str): The query text to send to the LLM
            retry (bool): Whether to retry once if the connection fails
            
        Returns:
            str: The response from the LLM
            
        Raises:
            Exception: If the LLM provider is not available or the query fails
        """
        # Check if LLM is available
        if not hasattr(self, 'llm') or not self.llm:
            logger.warning("No LLM provider available for query")
            raise Exception("No LLM provider is available for processing queries")
            
        if not self.llm_available and retry:
            logger.info("LLM was marked unavailable, trying to reconnect...")
            self.llm_available = self.check_llm_connection(force=True)
            
        if not self.llm_available:
            logger.warning("Cannot query LLM - service is not available")
            raise Exception(f"LLM provider ({self.llm_provider_type}) is not available. Try using specific commands instead.")
            
        try:
            # Send the query to the LLM
            logger.info(f"Sending query to {self.llm_provider_type}: {query}")
            print("💭 Thinking...")
            
            response = self.llm.query(query)
            
            # Cache the successful response
            if response:
                self.last_successful_response = response
                return response
                
            # Handle empty response
            raise Exception("LLM returned an empty response")
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            logger.error(traceback.format_exc())
            
            # If this is the first failure, try to reconnect and retry once
            if retry:
                logger.info("Trying to reconnect to LLM and retry...")
                self.llm_available = self.check_llm_connection(force=True)
                if self.llm_available:
                    logger.info("Reconnected to LLM, retrying query...")
                    return self.query_llm(query, retry=False)  # Retry once
                    
            # If we have a last successful response, we could return that with a warning
            # that this is stale information, but for now just raise an exception
            raise Exception(f"Failed to query language model: {str(e)}")
    
    def show_help(self, args):
        """Show help for available commands and tools"""
        # Get built-in commands
        built_in_commands = sorted(self.available_commands.keys())
        result = f"Available commands: {', '.join(built_in_commands)}.\n"
        result += "You can use these commands directly or by saying 'execute [command]'.\n"
        
        # Get MCP tools if available
        if hasattr(self.assistant, 'get_available_tools'):
            mcp_tools = self.assistant.get_available_tools()
            if mcp_tools:
                result += "\nAdditional tools available:\n"
                for tool_name, description in sorted(mcp_tools.items()):
                    result += f"• {tool_name}: {description}\n"
                result += "\nUse tools with 'execute [tool] param1=value1, param2=value2'"
        
        return result
        
    def use_tool(self, tool_name, **kwargs):
        """Use a tool by name with the given arguments"""
        self.logger.info(f"Using tool: {tool_name} with args: {kwargs}")
        
        # First check if the assistant supports MCP tools via execute_tool
        if hasattr(self.assistant, 'execute_tool'):
            try:
                self.logger.info(f"Executing MCP tool: {tool_name}")
                return self.assistant.execute_tool(tool_name, **kwargs)
            except Exception as e:
                self.logger.error(f"Error executing MCP tool {tool_name}: {e}")
                # Fall back to built-in commands if MCP tool fails
        
        # Check if we have this tool as a built-in command
        if tool_name in self.available_commands:
            try:
                # Convert kwargs to a single string argument if needed
                if kwargs:
                    args = " ".join(f"{k}={v}" for k, v in kwargs.items())
                else:
                    args = ""
                    
                result = self.available_commands[tool_name](args)
                return result
            except Exception as e:
                self.logger.error(f"Error using tool {tool_name}: {e}")
                return f"Error using {tool_name}: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"

    def try_switch_to_ollama(self):
        """Try to switch to Ollama if OpenAI isn't working"""
        if self.llm_provider_type != "openai":
            return False
            
        self.logger.info("Attempting to switch from OpenAI to Ollama as fallback...")
        print("\n" + "="*60)
        print("🚨 OPENAI CONNECTION FAILED - TRYING FALLBACK")
        
        try:
            # Try to import ollama - if not available, can't switch
            import ollama
            
            # Save original settings
            original_provider = self.llm_provider_type
            
            # Change provider
            self.llm_provider_type = "ollama"
            print("🔄 Temporarily switching to Ollama provider...")
            
            # Set Ollama properties based on assistant config
            self.llm = create_llm_provider(self.config)
            self.model = self.llm.get_model_name()
            self.current_llm_provider = self.llm
            
            # Check if Ollama is available
            ollama_available = self.check_llm_connection(force=True)
            
            if ollama_available:
                print("✅ Ollama connection successful! Using as fallback provider.")
                print(f"   Using model: {self.model}")
                print("💡 TIP: To make this change permanent, restart with --llm-provider=ollama")
                print("="*60 + "\n")
                return True
            else:
                # Switch back
                self.llm_provider_type = original_provider
                print("❌ Ollama fallback also failed. Reverting to original provider.")
                print("="*60 + "\n")
                return False
                
        except ImportError:
            print("❌ Ollama is not installed, cannot use as fallback")
            print("   Install it with: pip install ollama")
            print("="*60 + "\n")
            return False
        except Exception as e:
            self.logger.error(f"Error attempting to switch to Ollama: {e}")
            print(f"❌ Error attempting to switch to Ollama: {e}")
            print("="*60 + "\n")
            return False

def create_llm_provider(config):
    """
    Create an LLM provider based on the configuration
    Args:
        config: The configuration object
    Returns:
        LLMProvider: An instance of an LLM provider
    """
    if not config:
        logger.error("No configuration provided for LLM provider")
        raise ValueError("No configuration provided for LLM provider")
        
    # Get the provider type from config
    provider_type = config.get('llm_provider', 'none').lower()
    
    # If no provider or explicitly 'none', raise error
    if provider_type == 'none':
        logger.error("LLM provider set to 'none', cannot create provider")
        raise ValueError("LLM provider set to 'none', cannot create provider")
    
    logger.info(f"Creating LLM provider: {provider_type}")
    
    if provider_type == 'ollama':
        # Create Ollama provider
        return OllamaProvider(config)
    elif provider_type == 'openai':
        # Create OpenAI provider if available
        if not HAS_OPENAI:
            logger.error("OpenAI package not installed but OpenAI provider requested")
            raise ImportError("OpenAI package not installed. Please install with 'pip install openai'")
        logger.info(f"Initializing OpenAI provider with base URL {config.openai_base_url} and model {config.openai_model}")
        return OpenAIProvider(config)
    else:
        logger.error(f"Unknown LLM provider type: {provider_type}")
        raise ValueError(f"Unknown LLM provider type: {provider_type}. Supported: ollama, openai")


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, config):
        """Initialize the LLM provider with configuration"""
        self.config = config
        self.available = False
        self.model_name = "unknown"
        
    def get_model_name(self):
        """Get the current model name"""
        return self.model_name
        
    def check_connection(self):
        """Check if the LLM service is available"""
        raise NotImplementedError("Subclasses must implement check_connection")
        
    def query(self, prompt):
        """Query the LLM with a prompt"""
        raise NotImplementedError("Subclasses must implement query")
        
        
class OllamaProvider(LLMProvider):
    """Ollama LLM provider"""
    
    def __init__(self, config):
        """Initialize the Ollama provider with configuration"""
        super().__init__(config)
        self.provider_type = "ollama"
        
        # Import ollama
        try:
            global ollama
            import ollama
            logger.info("Ollama package imported successfully")
        except ImportError:
            logger.error("Ollama package not installed")
            raise ImportError("Ollama package not installed. Please install with 'pip install ollama'")
            
        # Get the model name and host/port from config
        self.model_name = config.get('ollama_model', 'phi3')
        
        # Get base URL or construct from host/port
        if 'ollama_base_url' in config:
            self.base_url = config.get('ollama_base_url')
        else:
            # Fallback to host/port if available
            host = config.get('ollama_host', 'localhost')
            port = config.get('ollama_port', 11434)
            self.base_url = f"http://{host}:{port}"
            
        logger.info(f"Ollama configured with base_url: {self.base_url}, model: {self.model_name}")
        
        # Set the base URL for Ollama
        ollama.BASE_URL = self.base_url
        
        # Check if Ollama is available
        self.available = self.check_connection()
        
    def check_connection(self):
        """Check if Ollama service is available and the model exists"""
        try:
            # First check if the Ollama server is responding
            import socket
            import urllib.parse
            
            # Parse the base URL to get host and port
            parsed_url = urllib.parse.urlparse(self.base_url)
            host = parsed_url.hostname or 'localhost'
            port = parsed_url.port or 11434
            
            # Try to connect to the Ollama server
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)  # 2 second timeout
            try:
                s.connect((host, port))
                s.close()
                logger.info(f"TCP connection to Ollama server at {host}:{port} successful")
            except Exception as e:
                logger.error(f"Failed to connect to Ollama server at {host}:{port}: {e}")
                return False
                
            # Now try to list the models to see if the server is responding properly
            models = ollama.list()
            if not models or 'models' not in models:
                logger.error("Failed to get models list from Ollama server")
                return False
                
            # Check if our model exists
            available_models = [m['name'] for m in models['models']]
            logger.info(f"Available Ollama models: {available_models}")
            
            if self.model_name not in available_models:
                logger.warning(f"Requested model '{self.model_name}' not available in Ollama")
                
                # Try to find an alternative model
                if available_models:
                    self.model_name = available_models[0]
                    logger.info(f"Using alternative model: {self.model_name}")
                else:
                    logger.error("No models available in Ollama")
                    return False
                    
            logger.info(f"Ollama connection successful, using model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking Ollama connection: {e}")
            logger.error(traceback.format_exc())
            return False
            
    def query(self, prompt):
        """Query the Ollama service with a prompt"""
        if not self.available:
            raise Exception("Ollama service is not available")
            
        try:
            logger.info(f"Querying Ollama with model {self.model_name}")
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            
            if response and 'message' in response and 'content' in response['message']:
                return response['message']['content'].strip()
            else:
                logger.error(f"Unexpected response format from Ollama: {response}")
                raise Exception("Unexpected response format from Ollama")
                
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"Failed to query Ollama: {str(e)}")
            
            
class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(self, config):
        """Initialize the OpenAI provider with configuration"""
        super().__init__(config)
        self.provider_type = "openai"
        
        # Check if OpenAI package is available
        if not HAS_OPENAI:
            logger.error("OpenAI package not installed")
            raise ImportError("OpenAI package not installed. Please install with 'pip install openai'")
            
        # Get configuration from config
        self.api_key = config.get('openai_api_key', 'n/a')
        self.base_url = config.get('openai_base_url', 'http://localhost:1234/v1')
        self.model_name = config.get('openai_model', 'gemma-3-27b-it')
        
        # Initialize the OpenAI client
        try:
            from openai import OpenAI
            client_kwargs = {}
            
            if self.api_key:
                client_kwargs['api_key'] = self.api_key
                
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
                
            self.client = OpenAI(**client_kwargs)
            logger.info(f"OpenAI client initialized with model: {self.model_name}")
            
            # Check if the client is working
            self.available = self.check_connection()
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            logger.error(traceback.format_exc())
            self.available = False
            raise
            
    def check_connection(self):
        """Check if OpenAI service is available and the model exists"""
        try:
            # Try to list models as a connectivity check
            models = self.client.models.list()
            
            # Check if our model exists in the list
            available_models = [model.id for model in models.data]
            logger.info(f"Available OpenAI models: {available_models[:5]}... (showing first 5)")
            
            # For API compatibility with hosted services, we won't fail if the model
            # isn't in the list - just log a warning
            if self.model_name not in available_models:
                logger.warning(f"Requested model '{self.model_name}' not found in available models list")
                
            logger.info("OpenAI connection successful")
            return True
            
        except Exception as e:
            logger.error(f"Error checking OpenAI connection 1: {e}")
            logger.error(traceback.format_exc())
            return False
            
    def query(self, prompt):
        """Query the OpenAI service with a prompt"""
        if not self.available:
            raise Exception("OpenAI service is not available")
            
        try:
            logger.info(f"Querying OpenAI with model {self.model_name}")
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            
            if completion and completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            else:
                logger.error(f"Unexpected response format from OpenAI: {completion}")
                raise Exception("Unexpected response format from OpenAI")
                
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"Failed to query OpenAI: {str(e)}") 