import requests
import datetime
import json
import threading
import time
import ollama
from utils import setup_logger
import logging
import random
import socket
import traceback

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

class CommandExecutor:
    def __init__(self, assistant):
        self.assistant = assistant
        self.timers = {}
        self.reminders = {}
        self.available_commands = {
            "search": self.search,
            "weather": self.get_weather,
            "time": self.get_time,
            "date": self.get_date,
            "news": self.get_news,
            "joke": self.tell_joke,
            "reminder": self.set_reminder,
            "timer": self.set_timer,
            "help": self.show_help,
            # Add more commonly used commands for direct access
            "what time is it": self.get_time,
            "what's the time": self.get_time,
            "what day is it": self.get_date,
            "what's the date": self.get_date,
            "tell me a joke": self.tell_joke,
            "check ollama": self.diagnostic_check_ollama,  # Add diagnostic command
            "check openai": self.diagnostic_check_openai  # Add OpenAI diagnostic command
        }
        
        # Common properties
        self.llm_provider = self.assistant.config.llm_provider
        self.llm_available = False
        self.model = None
        self.last_check_time = 0
        self.check_interval = 60  # Check connectivity every 60 seconds
        
        # Initialize the selected LLM provider
        if self.llm_provider == "ollama":
            self._initialize_ollama()
        elif self.llm_provider == "openai":
            if not HAS_OPENAI:
                logger.error("OpenAI provider selected but OpenAI library not available.")
                print("⚠️ OpenAI provider selected but OpenAI library not available. Install with 'pip install openai'")
                self.llm_available = False
            else:
                self._initialize_openai()
        else:
            logger.error(f"Unknown LLM provider: {self.llm_provider}")
            print(f"⚠️ Unknown LLM provider: {self.llm_provider}. Supported providers: ollama, openai")
            self.llm_available = False
    
    def _initialize_ollama(self):
        """Initialize the Ollama client"""
        # Initialize Ollama-specific attributes
        self.ollama_host = self.assistant.config.ollama_host
        self.ollama_port = self.assistant.config.ollama_port
        self.ollama_url = f"{self.ollama_host}:{self.ollama_port}"
        logger.info(f"Initializing Ollama client with host {self.ollama_url}")
        
        self.model = self.assistant.config.model
        self.ollama_client = None  # Will be initialized on demand
        
        try:
            # Just check the connection without creating a client object
            self.llm_available = self.check_ollama_connection()
            if not self.llm_available:
                logger.warning("Ollama seems to be unavailable. Will use command mode only.")
                print("⚠️ Warning: Ollama LLM service is not available. Using command mode only.")
                print("💡 Tip: You can use 'execute check ollama' to test connectivity.")
        except Exception as e:
            logger.error(f"Error initializing Ollama client: {e}")
            logger.error(traceback.format_exc())
            self.llm_available = False
            print(f"⚠️ Warning: Failed to initialize Ollama client: {e}")
            print("💡 Tip: You can use 'execute check ollama' to test connectivity.")
    
    def _initialize_openai(self):
        """Initialize the OpenAI client"""
        # Initialize OpenAI-specific attributes
        self.openai_api_key = self.assistant.config.openai_api_key
        self.openai_base_url = self.assistant.config.openai_base_url
        self.openai_model = self.assistant.config.openai_model
        self.openai_system_prompt = self.assistant.config.openai_system_prompt
        self.openai_temperature = self.assistant.config.openai_temperature
        self.openai_max_tokens = self.assistant.config.openai_max_tokens
        self.model = self.openai_model
        
        logger.info(f"Initializing OpenAI client with base URL: {self.openai_base_url}")
        
        # Check for invalid API key (literal "None" string)
        if self.openai_api_key == "None" or not self.openai_api_key:
            logger.error("OpenAI API key is not set or is set to 'None'")
            print("❌ ERROR: OpenAI API key is not set or is set to 'None'")
            print("Please provide a valid API key with --openai-api-key")
            self.llm_available = False
            return
            
        # Check for invalid base URL
        if "://" not in self.openai_base_url:
            logger.error(f"OpenAI base URL is invalid (missing protocol): {self.openai_base_url}")
            print(f"❌ ERROR: OpenAI base URL is invalid (missing protocol): {self.openai_base_url}")
            print("Example: --openai-base-url=https://api.openai.com/v1")
            self.llm_available = False
            return
        
        try:
            # Set up the OpenAI client
            self.openai_client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            
            # Check connectivity
            self.llm_available = self.check_openai_connection()
            if not self.llm_available:
                logger.warning("OpenAI API seems to be unavailable. Will use command mode only.")
                print("⚠️ Warning: OpenAI API is not available. Using command mode only.")
                print("💡 Tip: You can use 'execute check openai' to test connectivity.")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            logger.error(traceback.format_exc())
            self.llm_available = False
            print(f"⚠️ Warning: Failed to initialize OpenAI client: {e}")
            print("💡 Tip: You can use 'execute check openai' to test connectivity.")
    
    def diagnostic_check_ollama(self, args=None):
        """Diagnostic command to check Ollama connectivity"""
        if self.llm_provider != "ollama":
            return "Ollama is not the current LLM provider. Current provider is: " + self.llm_provider
            
        result = self.check_ollama_connection(force=True, detailed=True)
        if result:
            return f"Ollama is available at {self.ollama_url}. Model: {self.model}."
        else:
            return f"Ollama is NOT available at {self.ollama_url}. Please check if Ollama is running on the specified host and port."
    
    def diagnostic_check_openai(self, args=None):
        """Diagnostic command to check OpenAI connectivity"""
        if self.llm_provider != "openai":
            return "OpenAI is not the current LLM provider. Current provider is: " + self.llm_provider
            
        if not HAS_OPENAI:
            return "OpenAI library is not installed. Please install it with 'pip install openai'."
            
        result = self.check_openai_connection(force=True, detailed=True)
        if result:
            return f"OpenAI API is available at {self.openai_base_url}. Model: {self.openai_model}."
        else:
            # Check if we can suggest Ollama as fallback
            try:
                # See if we can check Ollama availability
                print("\n" + "="*60)
                print("🔍 CHECKING FALLBACK OPTIONS:")
                
                fallback_result = False
                try:
                    import ollama
                    # Store original provider
                    original_provider = self.llm_provider
                    
                    # Temporarily set provider to ollama for checking connection
                    self.llm_provider = "ollama"
                    fallback_result = self.check_ollama_connection(force=True, detailed=True)
                    
                    # Reset provider
                    self.llm_provider = original_provider
                    
                    if fallback_result:
                        print("\n💡 FALLBACK OPTION: Ollama is available! You can switch to it with:")
                        print("   --llm-provider=ollama")
                        print("   This should work immediately!")
                except ImportError:
                    print("❌ Ollama is not installed, cannot use as fallback")
                except Exception as e:
                    print(f"❌ Error checking Ollama fallback: {e}")
                
                print("="*60 + "\n")
            except Exception as e:
                logger.error(f"Error checking fallback options: {e}")
            
            return f"OpenAI API is NOT available at {self.openai_base_url}. Please check your API key and connection settings."
    
    def check_ollama_connection(self, force=False, detailed=False):
        """Check if Ollama is available and return connection status"""
        # Skip if we're not using Ollama
        if self.llm_provider != "ollama":
            return False
            
        current_time = time.time()
        
        # Skip check if we checked recently, unless forced
        if not force and current_time - self.last_check_time < self.check_interval:
            return self.llm_available
            
        self.last_check_time = current_time
        
        # Simple HTTP-based check using only the safe list API
        try:
            # Test the Ollama API directly
            logger.info(f"Testing Ollama API at {self.ollama_url}")
            
            if detailed:
                print(f"🔍 Testing connection to Ollama API...")
                
            # Just try to import ollama
            import ollama
            
            # Wrap in try-except to handle potential errors
            try:
                # Use a simple check with small timeout
                # Default to disabled if anything goes wrong
                models_list = None
                
                # Just try to list models with a fast timeout
                models_list = ollama.list()
                
                # If we got here, we know the service is running
                if detailed:
                    # Try to extract model names for detailed info
                    try:
                        model_names = []
                        if isinstance(models_list, dict) and "models" in models_list:
                            model_names = [m.get("name") for m in models_list["models"] 
                                         if isinstance(m, dict) and "name" in m]
                        elif isinstance(models_list, list):
                            model_names = [m.get("name") for m in models_list 
                                         if isinstance(m, dict) and "name" in m]
                                         
                        if model_names:
                            print(f"✅ Ollama is running! Available models: {', '.join(model_names)}")
                            
                            # Check if our model is available
                            if self.model not in model_names:
                                print(f"⚠️ Model {self.model} is not available")
                                
                                # Try to use a default model if available
                                if "llama3" in model_names or any("llama3" in m for m in model_names):
                                    m = next((m for m in model_names if "llama3" in m), "llama3")
                                    self.model = m
                                    print(f"🔄 Using fallback model: {self.model}")
                                elif "llama2" in model_names:
                                    self.model = "llama2"
                                    print(f"🔄 Using fallback model: llama2")
                                elif model_names:
                                    self.model = model_names[0]
                                    print(f"🔄 Using fallback model: {self.model}")
                        else:
                            print("✅ Ollama is running, but couldn't determine available models")
                    except Exception as model_ex:
                        print("✅ Ollama is running, but couldn't determine available models")
                        logger.warning(f"Error parsing model list: {model_ex}")
                
                # If we got here without exception, Ollama is available
                self.llm_available = True
                logger.info("Ollama connection check: SUCCESSFUL")
                return True
                
            except Exception as e:
                # Handle connection errors
                logger.error(f"Error listing Ollama models: {e}")
                if detailed:
                    print(f"❌ Error checking Ollama: {str(e)}")
                self.llm_available = False
                logger.warning("Ollama connection check: FAILED")
                return False
                
        except ImportError as ie:
            # Handle import errors
            logger.error(f"Error importing Ollama: {ie}")
            if detailed:
                print(f"❌ Ollama module not available: {str(ie)}")
            self.llm_available = False
            return False
            
        except Exception as e:
            # Handle any other errors
            logger.error(f"Error checking Ollama: {e}")
            if detailed:
                print(f"❌ Error connecting to Ollama: {str(e)}")
            self.llm_available = False
            return False
    
    def check_openai_connection(self, force=False, detailed=False):
        """Check if OpenAI API is available and return connection status"""
        # Skip if we're not using OpenAI
        if self.llm_provider != "openai":
            return False
            
        # Skip if OpenAI is not installed
        if not HAS_OPENAI:
            logger.error("OpenAI library is not installed")
            if detailed:
                print("❌ OpenAI library is not installed")
            return False
        
        current_time = time.time()
        
        # Skip check if we checked recently, unless forced
        if not force and current_time - self.last_check_time < self.check_interval:
            return self.llm_available
            
        self.last_check_time = current_time
        
        # Test the OpenAI API with a simple models listing request
        try:
            logger.info(f"Testing OpenAI API at {self.openai_base_url}")
            
            if detailed:
                print(f"🔍 Testing connection to OpenAI API...")
                print(f"   URL: {self.openai_base_url}")
                print(f"   Model: {self.openai_model}")
                # Don't print the actual API key value
                print(f"   API Key: {'Set' if self.openai_api_key and self.openai_api_key != 'None' else 'NOT SET'}")
                
                # Check API key specifically
                if self.openai_api_key == "None" or not self.openai_api_key:
                    print("❌ ERROR: API key is not set or is 'None'")
                    print("   You must provide a valid API key with --openai-api-key")
                    self.llm_available = False
                    return False
            
            # Set a shorter timeout for the API call
            from openai import OpenAIError
            
            # Use a separate client instance with a short timeout for testing
            test_client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                timeout=5.0  # 5 second timeout for faster diagnostics
            )
            
            # Use list models to test connectivity - this is a lightweight call
            try:
                models_response = test_client.models.list()
                
                # Check if we got a valid response
                if detailed:
                    try:
                        # Extract model IDs for information
                        if hasattr(models_response, 'data') and models_response.data:
                            model_ids = [model.id for model in models_response.data if hasattr(model, 'id')]
                            print(f"✅ OpenAI API is available! Available models: {', '.join(model_ids[:5])}...")
                            
                            # Check if our model is in the list
                            if self.openai_model not in model_ids:
                                print(f"⚠️ Warning: Model {self.openai_model} may not be available")
                                print("💡 The API might still work if the model is available but not listed")
                                
                                # Try to suggest an alternative model
                                if model_ids:
                                    print(f"💡 Available models include: {', '.join(model_ids[:5])}")
                        else:
                            print("✅ OpenAI API is available, but couldn't determine available models")
                    except Exception as model_ex:
                        logger.warning(f"Error parsing model list: {model_ex}")
                        print("✅ OpenAI API is available, but couldn't parse model list")
                
                # If we get here, the API is working
                self.llm_available = True
                logger.info("OpenAI API connection check: SUCCESSFUL")
                return True
                
            except OpenAIError as oe:
                # Handle specific OpenAI errors
                logger.error(f"OpenAI API error: {oe}")
                if detailed:
                    print(f"❌ OpenAI API error: {str(oe)}")
                    
                    # Check for common error types and provide more specific guidance
                    err_msg = str(oe).lower()
                    if "authentication" in err_msg or "api key" in err_msg:
                        print("💡 This appears to be an authentication error.")
                        print("   Check that your API key is correct and properly formatted.")
                    elif "not found" in err_msg:
                        print("💡 This appears to be a 'not found' error.")
                        print("   Check that your base URL is correct.")
                    elif "timeout" in err_msg or "timed out" in err_msg:
                        print("💡 The API request timed out.")
                        print("   The server might be slow or unreachable.")
                    elif "connect" in err_msg or "connection" in err_msg:
                        print("💡 This appears to be a connection error.")
                        print("   Check that the API server is running and reachable.")
                        
                        # If local server, provide more specific help
                        if "localhost" in self.openai_base_url or "127.0.0.1" in self.openai_base_url:
                            try:
                                import socket
                                host = "localhost"
                                # Try to extract port from URL
                                port_str = self.openai_base_url.split("://")[1].split(":")[1].split("/")[0]
                                port = int(port_str)
                                
                                print(f"\n🔍 Checking if port {port} is open on {host}...")
                                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                s.settimeout(1)
                                result = s.connect_ex((host, port))
                                if result == 0:
                                    print(f"✅ Port {port} is open on {host}")
                                    print("   The server is running but responding with errors.")
                                else:
                                    print(f"❌ Port {port} is CLOSED on {host}")
                                    print(f"   Make sure your local OpenAI API server is running on port {port}")
                                s.close()
                            except Exception as port_e:
                                # Silently handle port check errors
                                pass
                    
                    print("💡 Check your API key and base URL settings")
                
                self.llm_available = False
                logger.warning("OpenAI API connection check: FAILED")
                return False
                
            except Exception as e:
                # Handle general API errors
                logger.error(f"Error checking OpenAI API: {e}")
                if detailed:
                    print(f"❌ Error checking OpenAI API: {str(e)}")
                    print("💡 Check your API key and base URL settings")
                self.llm_available = False
                logger.warning("OpenAI API connection check: FAILED")
                return False
                
        except Exception as e:
            # Handle any other errors
            logger.error(f"Error checking OpenAI API: {e}")
            if detailed:
                print(f"❌ Error connecting to OpenAI API: {str(e)}")
            self.llm_available = False
            return False
    
    def query_llm(self, query, retry=True):
        """Send a query to the selected LLM provider"""
        # Check if LLM is available
        if not self.llm_available and retry:
            logger.info("LLM was marked unavailable, trying to reconnect...")
            
            if self.llm_provider == "ollama":
                self.llm_available = self.check_ollama_connection(force=True)
            elif self.llm_provider == "openai":
                self.llm_available = self.check_openai_connection(force=True)
            
        if not self.llm_available:
            logger.warning("Cannot query LLM - service is not available")
            
            # Try to switch to Ollama as fallback if using OpenAI
            if self.llm_provider == "openai" and retry:
                logger.info("Trying to switch to Ollama as fallback...")
                
                if self.try_switch_to_ollama():
                    # If switch successful, try the query again with the new provider
                    logger.info("Successfully switched to Ollama, retrying query")
                    return self.query_llm(query, retry=False)  # Prevent infinite loops
            
            # If we get here, we can't use any LLM
            raise Exception(f"{self.llm_provider.capitalize()} language model service is not available. Try using specific commands instead.")
            
        # Dispatch to the appropriate provider
        if self.llm_provider == "ollama":
            return self._query_ollama(query, retry)
        elif self.llm_provider == "openai":
            return self._query_openai(query, retry)
        else:
            logger.error(f"Unknown LLM provider: {self.llm_provider}")
            raise Exception(f"Unknown LLM provider: {self.llm_provider}")
    
    def _query_ollama(self, query, retry=True):
        """Send a query to the Ollama LLM"""
        try:
            logger.info(f"Sending query to Ollama ({self.model}): {query}")
            
            # Use the chat API with direct approach for Ollama
            from ollama import chat
            
            # Simple direct call, no client object needed
            response = chat(
                model=self.model,
                messages=[{"role": "user", "content": query}]
            )
            
            # Extract the response text using the structure from example
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                # Access directly as object attributes (new API style)
                response_text = response.message.content
            elif isinstance(response, dict):
                # Try dictionary access (older style)
                response_text = response.get("message", {}).get("content", "")
            else:
                # Last resort fallback
                response_text = str(response)
                logger.warning(f"Unexpected response format: {type(response)}")
            
            logger.info(f"Ollama response: {response_text[:100]}...")  # Log first 100 chars
            return response_text
            
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            logger.error(traceback.format_exc())
            
            # If this is the first failure, try to reconnect and retry once
            if retry:
                logger.info("Trying to reconnect to Ollama and retry...")
                self.llm_available = self.check_ollama_connection(force=True)
                if self.llm_available:
                    logger.info("Reconnected to Ollama, retrying query...")
                    return self._query_ollama(query, retry=False)  # Retry once
                    
            # If still failing or this is already a retry, raise the exception
            raise Exception(f"Failed to query Ollama language model: {str(e)}")
    
    def _query_openai(self, query, retry=True):
        """Send a query to the OpenAI API"""
        try:
            logger.info(f"Sending query to OpenAI API ({self.openai_model}): {query}")
            
            # Quick validation of key config
            if self.openai_api_key == "None" or not self.openai_api_key:
                logger.error("Cannot query OpenAI: API key is not set or is set to 'None'")
                raise Exception("OpenAI API key is not properly configured. Please set a valid API key with --openai-api-key.")
            
            # Prepare messages
            messages = []
            
            # Add system prompt if provided
            if self.openai_system_prompt:
                messages.append({"role": "system", "content": self.openai_system_prompt})
                
            # Add user message
            messages.append({"role": "user", "content": query})
            
            # Prepare parameters for the API call
            params = {
                "model": self.openai_model,
                "messages": messages,
                "temperature": self.openai_temperature,
            }
            
            # Add max_tokens if specified
            if self.openai_max_tokens:
                params["max_tokens"] = self.openai_max_tokens
            
            # Import OpenAIError for more specific error handling
            from openai import OpenAIError
            
            try:
                # Make the API call with a timeout
                response = self.openai_client.chat.completions.create(**params)
                
                # Extract the response text
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    # Get the first choice's content
                    first_choice = response.choices[0]
                    if hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
                        response_text = first_choice.message.content
                    else:
                        response_text = str(first_choice)
                        logger.warning(f"Unexpected response structure: {first_choice}")
                else:
                    response_text = str(response)
                    logger.warning(f"Unexpected response format: {response}")
                
                logger.info(f"OpenAI response: {response_text[:100]}...")  # Log first 100 chars
                return response_text
                
            except OpenAIError as oe:
                # Handle specific OpenAI errors
                logger.error(f"OpenAI API error: {oe}")
                
                # Extract useful info from error message
                err_msg = str(oe).lower()
                
                # Provide more helpful error messages based on error type
                if "authentication" in err_msg or "api key" in err_msg:
                    raise Exception("Authentication error with OpenAI API. Check that your API key is correct.")
                elif "not found" in err_msg:
                    raise Exception("Resource not found error with OpenAI API. Check that your base URL and model are correct.")
                elif "timeout" in err_msg or "timed out" in err_msg:
                    raise Exception("OpenAI API request timed out. The server might be slow or unreachable.")
                elif "rate limit" in err_msg:
                    raise Exception("OpenAI API rate limit exceeded. Please try again later or reduce request frequency.")
                else:
                    # Pass through the original error with some context
                    raise Exception(f"OpenAI API error: {str(oe)}")
                    
            except Exception as e:
                # Handle other exceptions during API call
                logger.error(f"Unexpected error during OpenAI API call: {e}")
                logger.error(traceback.format_exc())
                raise Exception(f"Unexpected error during OpenAI API call: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error querying OpenAI API: {e}")
            logger.error(traceback.format_exc())
            
            # If this is the first failure, try to reconnect and retry once
            if retry:
                logger.info("Trying to reconnect to OpenAI API and retry...")
                self.llm_available = self.check_openai_connection(force=True)
                if self.llm_available:
                    logger.info("Reconnected to OpenAI API, retrying query...")
                    return self._query_openai(query, retry=False)  # Retry once
                
                # If we still can't connect, try switching to Ollama
                logger.info("OpenAI reconnection failed, trying to switch to Ollama...")
                if self.try_switch_to_ollama():
                    # If switch successful, use Ollama provider
                    logger.info("Successfully switched to Ollama, using it for this query")
                    return self._query_ollama(query, retry=False)  # Prevent infinite loops
                    
            # If still failing or this is already a retry, raise the exception
            raise Exception(f"Failed to query OpenAI language model: {str(e)}")
            
    def execute(self, query):
        """Process a query - either execute a command or send to the LLM"""
        logger.info(f"Processing: {query}")
        
        # Check if this query starts with the execute command
        if query.lower().startswith("execute "):
            command_text = query[8:].strip()  # Remove "execute " prefix
            logger.info(f"Explicit command execution: {command_text}")
            
            # First check for built-in commands (without trying LLM as a fallback)
            result = self.execute_command(command_text)
            
            # If the command is recognized and processed successfully, return the result
            # Otherwise, we still return the result which will be an error message
            return result
            
        # Check if this matches a known command pattern
        parts = query.split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        
        if command in self.available_commands:
            args = parts[1] if len(parts) > 1 else ""
            logger.info(f"Executing command: {command} with args: {args}")
            result = self.available_commands[command](args)
            return result
            
        # If LLM is not available, don't try to process with LLM
        if not self.llm_available:
            logger.warning("Cannot process query with LLM - service is not available")
            return f"I'm sorry, I can't process that query because the {self.llm_provider} language model service is not available. You can use specific commands like 'execute time' or 'execute joke', or check the connection with 'execute check {self.llm_provider}'."
            
        # If not a command and LLM is available, send to LLM
        try:
            # Double-check LLM is available before sending query
            if self.llm_provider == "ollama" and not self.check_ollama_connection(force=True):
                logger.warning("Ollama became unavailable, cannot process query")
                return "I'm sorry, I can't process that query because the Ollama language model service is not available. You can use specific commands like 'execute time' or 'execute joke', or check the connection with 'execute check ollama'."
            elif self.llm_provider == "openai" and not self.check_openai_connection(force=True):
                logger.warning("OpenAI API became unavailable, cannot process query")
                return "I'm sorry, I can't process that query because the OpenAI language model service is not available. You can use specific commands like 'execute time' or 'execute joke', or check the connection with 'execute check openai'."
                
            logger.info(f"Sending to {self.llm_provider}: {query}")
            print("💭 Thinking...")
            response = self.query_llm(query)
            
            # If not in continuous conversation mode, exit after one interaction
            if hasattr(self.assistant.config, 'continuous_conversation') and not self.assistant.config.continuous_conversation:
                logger.info("Exiting conversation mode (continuous mode disabled)")
                self.assistant.in_conversation = False
                self.assistant.audio_manager.set_conversation_mode(False)
                response += " Call me if you need me."
                print("🔚 Conversation ended. Say the wake word to start again.")
                
            return response
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            logger.error(traceback.format_exc())
            
            # Mark LLM as unavailable if there was an error
            self.llm_available = False
            
            # If LLM fails, provide a more helpful response
            return f"I couldn't connect to the {self.llm_provider} language model service. You can use specific commands like 'execute time', 'execute joke', or check the connection with 'execute check {self.llm_provider}'."
        
    def execute_command(self, command_text):
        logger.info(f"Executing command: {command_text}")
        
        # Split command into parts
        parts = command_text.split(maxsplit=1)
        if not parts:
            return "I couldn't understand that command."
            
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # First check if this is a command we have directly available
        if command in self.available_commands:
            try:
                logger.info(f"Executing built-in command: {command} with args: {args}")
                result = self.available_commands[command](args)
                return result
            except Exception as e:
                logger.error(f"Error executing command {command}: {e}")
                return f"I encountered an error executing that command: {str(e)}"
        
        # Then check if this might be an MCP tool (if available)
        if hasattr(self.assistant, 'execute_tool') and hasattr(self.assistant, 'get_available_tools'):
            available_tools = self.assistant.get_available_tools()
            if command in available_tools:
                try:
                    logger.info(f"Executing MCP tool: {command} with args: {args}")
                    # Parse the args into a dictionary
                    kwargs = {}
                    if args:
                        # Simple parsing for key=value pairs
                        for arg_pair in args.split(','):
                            if '=' in arg_pair:
                                key, value = arg_pair.split('=', 1)
                                kwargs[key.strip()] = value.strip().strip('"\'')
                    
                    result = self.assistant.execute_tool(command, **kwargs)
                    return result
                except Exception as e:
                    logger.error(f"Error executing MCP tool {command}: {e}")
                    return f"I encountered an error executing that tool: {str(e)}"
        
        # If we get here, it's an unknown command
        return f"Unknown command: {command}. Try 'execute help' to see available commands."
            
    def search(self, query):
        if not query:
            return "Please specify what you'd like to search for."
            
        print(f"🔍 Searching for {query}...")
        try:
            # Simple search using DuckDuckGo
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url)
            data = response.json()
            
            if data.get("Abstract"):
                return data["Abstract"]
            else:
                print("No clear answer found. Asking the language model...")
                llm_response = self.query_llm(f"Please provide information about: {query}")
                return llm_response
        except Exception as e:
            logger.error(f"Search error: {e}")
            print("Search error. Falling back to language model...")
            llm_response = self.query_llm(f"Please provide information about: {query}")
            return llm_response
            
    def get_weather(self, args):
        # In a real implementation, you would use a weather API
        return "I'm sorry, I don't have access to weather data yet. This would require an API key for a weather service."
        
    def get_time(self, args):
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}."
        
    def get_date(self, args):
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}."
        
    def get_news(self, args):
        return "I'm sorry, I don't have access to news data yet. This would require an API key for a news service."
        
    def tell_joke(self, args):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them!",
            "Why was the computer cold? It left its Windows open!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "What's the best thing about Switzerland? I don't know, but the flag is a big plus!",
            "How do you organize a space party? You planet!",
            "Why don't skeletons fight each other? They don't have the guts!",
            "What did one ocean say to the other ocean? Nothing, they just waved!"
        ]
        return random.choice(jokes)
        
    def set_reminder(self, text):
        if not text:
            return "Please specify what you want to be reminded about."
            
        # Simple reminder implementation
        parts = text.split(" in ")
        if len(parts) != 2:
            return "Please use the format: 'reminder [message] in [time]', for example 'reminder take out trash in 5 minutes'."
            
        message = parts[0].strip()
        time_str = parts[1].strip()
        
        # Parse time
        try:
            duration = self._parse_duration(time_str)
            if duration <= 0:
                return "Please specify a positive duration for the reminder."
                
            # Create a unique ID for this reminder
            reminder_id = f"reminder_{int(time.time())}"
            
            # Schedule the reminder
            self.reminders[reminder_id] = {
                "message": message,
                "time": time.time() + duration,
                "duration_text": time_str
            }
            
            # Start a timer thread
            threading.Thread(
                target=self._run_reminder,
                args=(reminder_id, duration, message, time_str),
                daemon=True
            ).start()
            
            return f"I'll remind you to {message} in {time_str}."
        except Exception as e:
            logger.error(f"Error setting reminder: {e}")
            return f"I couldn't set that reminder: {str(e)}"
            
    def set_timer(self, duration_text):
        if not duration_text:
            return "Please specify a duration for the timer."
            
        try:
            # Parse the duration
            duration = self._parse_duration(duration_text)
            if duration <= 0:
                return "Please specify a positive duration for the timer."
                
            # Create a unique ID for this timer
            timer_id = f"timer_{int(time.time())}"
            
            # Store the timer
            self.timers[timer_id] = {
                "duration": duration,
                "start_time": time.time(),
                "duration_text": duration_text
            }
            
            # Start a timer thread
            threading.Thread(
                target=self._run_timer,
                args=(timer_id, duration, duration_text),
                daemon=True
            ).start()
            
            return f"Timer set for {duration_text}."
        except Exception as e:
            logger.error(f"Error setting timer: {e}")
            return f"I couldn't set that timer: {str(e)}"
            
    def _parse_duration(self, duration_text):
        """Parse a duration string like '5 minutes' into seconds"""
        parts = duration_text.lower().split()
        if len(parts) < 2:
            raise ValueError("Please specify both a number and a unit (seconds, minutes, hours)")
            
        try:
            value = float(parts[0])
            unit = parts[1]
            
            if unit.startswith("second"):
                return value
            elif unit.startswith("minute"):
                return value * 60
            elif unit.startswith("hour"):
                return value * 3600
            else:
                raise ValueError(f"Unknown time unit: {unit}")
        except ValueError as e:
            raise ValueError(f"I couldn't understand that duration: {str(e)}")
            
    def _run_reminder(self, reminder_id, duration, message, duration_text):
        """Run a reminder in the background"""
        logger.info(f"Reminder set: {message} in {duration_text} ({duration} seconds)")
        time.sleep(duration)
        
        # Check if the reminder still exists (it might have been cancelled)
        if reminder_id in self.reminders:
            logger.info(f"Reminder triggered: {message}")
            self.assistant.speak(f"Reminder: {message}")
            # Remove the reminder
            del self.reminders[reminder_id]
            
    def _run_timer(self, timer_id, duration, duration_text):
        """Run a timer in the background"""
        logger.info(f"Timer set for {duration_text} ({duration} seconds)")
        time.sleep(duration)
        
        # Check if the timer still exists (it might have been cancelled)
        if timer_id in self.timers:
            logger.info(f"Timer finished: {duration_text}")
            self.assistant.speak(f"Timer for {duration_text} is done!")
            # Remove the timer
            del self.timers[timer_id]
            
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
        logger.info(f"Using tool: {tool_name} with args: {kwargs}")
        
        # First check if the assistant supports MCP tools via execute_tool
        if hasattr(self.assistant, 'execute_tool'):
            try:
                logger.info(f"Executing MCP tool: {tool_name}")
                return self.assistant.execute_tool(tool_name, **kwargs)
            except Exception as e:
                logger.error(f"Error executing MCP tool {tool_name}: {e}")
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
                logger.error(f"Error using tool {tool_name}: {e}")
                return f"Error using {tool_name}: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"

    def try_switch_to_ollama(self):
        """Try to switch to Ollama if OpenAI isn't working"""
        if self.llm_provider != "openai":
            return False
            
        logger.info("Attempting to switch from OpenAI to Ollama as fallback...")
        print("\n" + "="*60)
        print("🚨 OPENAI CONNECTION FAILED - TRYING FALLBACK")
        
        try:
            # Try to import ollama - if not available, can't switch
            import ollama
            
            # Save original settings
            original_provider = self.llm_provider
            
            # Change provider
            self.llm_provider = "ollama"
            print("🔄 Temporarily switching to Ollama provider...")
            
            # Set Ollama properties based on assistant config
            self.ollama_host = self.assistant.config.ollama_host
            self.ollama_port = self.assistant.config.ollama_port
            self.ollama_url = f"{self.ollama_host}:{self.ollama_port}"
            self.model = self.assistant.config.model
            
            # Check if Ollama is available
            ollama_available = self.check_ollama_connection(force=True, detailed=True)
            
            if ollama_available:
                print("✅ Ollama connection successful! Using as fallback provider.")
                print(f"   Using model: {self.model}")
                print("💡 TIP: To make this change permanent, restart with --llm-provider=ollama")
                print("="*60 + "\n")
                return True
            else:
                # Switch back
                self.llm_provider = original_provider
                print("❌ Ollama fallback also failed. Reverting to original provider.")
                print("="*60 + "\n")
                return False
                
        except ImportError:
            print("❌ Ollama is not installed, cannot use as fallback")
            print("   Install it with: pip install ollama")
            print("="*60 + "\n")
            return False
        except Exception as e:
            logger.error(f"Error attempting to switch to Ollama: {e}")
            print(f"❌ Error attempting to switch to Ollama: {e}")
            print("="*60 + "\n")
            return False 