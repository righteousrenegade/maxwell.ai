import requests
import datetime
import json
import threading
import time
import logging
import random
import socket
import traceback
import inspect

from utils import setup_logger

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
    
    def __init__(self, assistant=None, config=None, mcp_tool_provider=None):
        """Initialize the command executor with configuration and llm settings"""
        # Store mcp_tool_provider directly instead of assistant
        logger.info(f"CommandExecutor initialized with mcp_tool_provider: {mcp_tool_provider}")
        self.mcp_tool_provider = mcp_tool_provider
        self.config = config or None
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage for timers and reminders
        self.timers = {}
        self.reminders = {}
        
        # Initialize tools and mcp_tools dictionaries
        self.available_commands = {}
        self.mcp_tools = {}
        if not config:
            logger.error("No config provided, unable to initialize CommandExecutor")
        # Check for MCP tools integration - simple direct approach
        if config:
            self.use_mcp = getattr(config, 'use_mcp', True)
            logger.info(f"CommandExecutor initialized with use_mcp={self.use_mcp}")
        else:
            logger.info("No config provided, defaulting to use_mcp=True")
            self.use_mcp = True
            
        # Register basic commands that don't require LLM
        self._register_basic_commands()
        
        # If MCP tools are available, register them
        if self.use_mcp:
            logger.info("MCP tools available, registering with CommandExecutor")
            self._register_mcp_tools()
        elif self.use_mcp:
            logger.info("MCP tools enabled but not available")
        else:
            logger.info("MCP tools disabled in configuration")
        
        # Log initialization
        logger.info("CommandExecutor initialized")
        
    def _register_mcp_tools(self):
        """Register available MCP tools as commands"""
        try:
            if not self.mcp_tool_provider:
                logger.warning("MCP tools requested but not available")
                return
                
            # Get available tools from the MCP tool provider
            available_tools = self.mcp_tool_provider.get_tool_descriptions()
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
                    
            # Execute the tool through the mcp_tool_provider directly
            logger.info(f"Preceding the execution of MCP tool: {tool_name} with kwargs: {kwargs}")
            result = self.mcp_tool_provider.execute_tool(tool_name, **kwargs)
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
                
            if search_term:
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
        if command == "search":
            logger.info(f"Search command with args: '{args}'")
            print(f"🔍 Searching for: '{args}'")
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
            
        # Check if the command exists in available_commands
        if command in self.available_commands:
            try:
                logger.info(f"Executing direct command: {command} with args: {args}")
                return self.available_commands[command](args)
            except Exception as e:
                logger.error(f"Error executing command: {e}")
                logger.error(traceback.format_exc())
                return f"Error executing command: {str(e)}"
        
        # Try executing through MCP tools if available
        for tool_name in self.mcp_tools:
            if command == tool_name or command.replace("_", " ") == tool_name.replace("_", " "):
                logger.info(f"Executing MCP tool via self._execute_mcp_tool: {tool_name} with args: {args}")
                return self._execute_mcp_tool(tool_name, args)
        
        # Special handling for common commands even if they don't match an exact MCP tool name
        if command in ["time", "date", "weather", "joke"]:
            tool_mapping = {
                "time": "get_time",
                "date": "get_date",
                "weather": "get_weather",
                "joke": "tell_joke"
            }
            tool_name = tool_mapping.get(command)
            if tool_name:
                logger.info(f"Executing mapped MCP tool: {tool_name} with args: {args}")
                return self._execute_mcp_tool(tool_name, args)
        
        # If no direct match, try matching against patterns
        for pattern, cmd in COMMAND_PATTERNS.items():
            if pattern in command_text.lower() and cmd in self.available_commands:
                try:
                    logger.info(f"Pattern match: '{pattern}' -> {cmd}")
                    return self.available_commands[cmd](args)
                except Exception as e:
                    logger.error(f"Error executing pattern-matched command: {e}")
                    return f"Error executing command: {str(e)}"

        # No matches found
        available_cmds = sorted(list(self.available_commands.keys()))
        return f"Unknown command: '{command}'. Available commands: {', '.join(available_cmds[:10])}"

    def _register_basic_commands(self):
        """Register basic commands that don't require LLM or are not covered by MCP tools"""
        # Register help command
        self.available_commands["help"] = self.show_help
        
        # Add command shortcut for details search result
        self.available_commands["details"] = lambda args: self._handle_details_command(args)
        
        # Add basic search command
        self.available_commands["search"] = lambda args: self._execute_search_command(args)
        
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

    def show_help(self, args):
        """Show help text for available commands and tools"""
        logger.info(f"Showing help: {args}")
        
        # Check for specific command help
        if args:
            command = args.lower()
            # First check in available direct commands
            if command in self.available_commands:
                return f"Command: {command}\n{inspect.getdoc(self.available_commands[command])}"
            # Then check in mcp_tools
            elif command in self.mcp_tools:
                return f"Tool: {command}\n{self.mcp_tools[command]['description']}"
            else:
                return f"No help available for '{command}'. Try 'help' for a list of commands."
                
        # List built-in commands
        built_in_commands = sorted([cmd for cmd in self.available_commands.keys() 
                               if cmd not in self.mcp_tools])
                               
        result = f"Available commands: {', '.join(built_in_commands)}.\n"
        result += "You can use these commands directly or by saying 'execute [command]'.\n"
        
        # Get MCP tools if available
        if self.mcp_tools:
            result += "\nAdditional tools available:\n"
            for tool_name, tool_info in sorted(self.mcp_tools.items()):
                result += f"• {tool_name}: {tool_info['description']}\n"
            result += "\nUse tools with 'execute [tool] param1=value1, param2=value2'"
        
        return result
        
    def use_tool(self, tool_name, **kwargs):
        """Use a tool by name with the given arguments"""
        self.logger.info(f"Using tool: {tool_name} with args: {kwargs}")
        
        # First check if we can directly use the MCP tool provider
        try:
            self.logger.info(f"Executing MCP tool: {tool_name}")
            return self.mcp_tool_provider.execute_tool(tool_name, **kwargs)
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
