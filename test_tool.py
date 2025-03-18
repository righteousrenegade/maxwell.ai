#!/usr/bin/env python3
import argparse
import sys
import logging
from mcp_tools import MCPToolProvider
from utils import setup_logger

# Configure logging
logger = logging.getLogger("maxwell")
if not logger.handlers:
    logger = setup_logger()

class MockAssistant:
    """A mock assistant class just for testing tools"""
    def __init__(self):
        self.mcp_tool_provider = None

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Test MCP tools directly from the command line')
    parser.add_argument('tool_name', help='Name of the tool to test')
    parser.add_argument('arguments', nargs='*', help='Arguments for the tool (as positional args or key=value pairs)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create a mock assistant and tool provider
    assistant = MockAssistant()
    tool_provider = MCPToolProvider(assistant)
    assistant.mcp_tool_provider = tool_provider
    
    # If no tool name provided or tool is "list", list available tools
    if not args.tool_name or args.tool_name.lower() == 'list':
        print("Available tools:")
        for name, desc in tool_provider.get_tool_descriptions().items():
            print(f"  {name}: {desc}")
        return
    
    # Parse the arguments
    kwargs = {}
    
    # Special handling for search_web if just given a query string
    if args.tool_name == "search_web" and args.arguments and not any('=' in arg for arg in args.arguments):
        kwargs["query"] = " ".join(args.arguments)
    
    # Special handling for details_search_result
    elif args.tool_name == "details_search_result" and args.arguments:
        # Just take the first argument as the result number
        kwargs["result_number"] = args.arguments[0]
    
    # Special handling for get_weather, set_reminder, set_timer
    elif args.tool_name in ["get_weather", "set_reminder", "set_timer"] and args.arguments and not any('=' in arg for arg in args.arguments):
        if args.tool_name == "get_weather":
            kwargs["location"] = " ".join(args.arguments)
        elif args.tool_name == "set_reminder":
            kwargs["text"] = " ".join(args.arguments)
        elif args.tool_name == "set_timer":
            kwargs["duration"] = " ".join(args.arguments)
    
    # Otherwise, process key=value pairs
    else:
        for arg in args.arguments:
            if '=' in arg:
                key, value = arg.split('=', 1)
                kwargs[key] = value
            else:
                # Just append to kwargs as a positional argument using the right name
                # based on the tool's first parameter
                if not kwargs:
                    if args.tool_name == "search_web":
                        kwargs["query"] = arg
                    elif args.tool_name == "get_weather":
                        kwargs["location"] = arg
                    elif args.tool_name == "set_reminder":
                        kwargs["text"] = arg
                    elif args.tool_name == "set_timer":
                        kwargs["duration"] = arg
                    elif args.tool_name == "details_search_result":
                        kwargs["result_number"] = arg
                    elif args.tool_name == "tell_joke":
                        pass  # No args needed
                    else:
                        print(f"Warning: Argument '{arg}' not in key=value format, ignoring")
                else:
                    print(f"Warning: Argument '{arg}' not in key=value format, ignoring")
    
    # Execute the tool
    print(f"Executing tool: {args.tool_name} with arguments: {kwargs}")
    try:
        result = tool_provider.execute_tool(args.tool_name, **kwargs)
        
        # Extra debugging for search
        if args.tool_name == "search_web":
            from mcp_tools import search_results_cache
            if search_results_cache:
                latest_search = list(search_results_cache.keys())[-1]
                search_data = search_results_cache[latest_search]
                urls = search_data.get("urls", [])
                titles = search_data.get("titles", [])
                snippets = search_data.get("snippets", [])
                print(f"\nDEBUG: Found {len(urls)} URLs, {len(titles)} titles, {len(snippets)} snippets")
                for i, (url, title) in enumerate(zip(urls, titles)):
                    print(f"Result {i+1}: {title[:40]}... - {url[:40]}...")
        
        print("\nResult:")
        print(result)
    except Exception as e:
        print(f"Error executing tool: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 