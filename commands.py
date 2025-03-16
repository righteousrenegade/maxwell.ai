import requests
import datetime
import json
import threading
import time
import ollama
from utils import setup_logger
import logging

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
            "help": self.show_help
        }
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(
            host=f"http://{self.assistant.config.ollama_host}:{self.assistant.config.ollama_port}"
        )
        self.model = self.assistant.config.model
        
        # MCP tools will be registered separately
        
    def execute_command(self, command_text):
        logger.info(f"Executing command: {command_text}")
        
        # Split command into parts
        parts = command_text.split(maxsplit=1)
        if not parts:
            self.assistant.speak("I couldn't understand that command.")
            return True
            
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Check if command exists
        if command in self.available_commands:
            try:
                self.available_commands[command](args)
            except Exception as e:
                logger.error(f"Error executing command {command}: {e}")
                self.assistant.speak(f"I encountered an error executing that command: {str(e)}")
            return True
        else:
            self.assistant.speak(f"I don't know the command '{command}'. Say 'execute help' for a list of commands.")
            return True
            
    def query_llm(self, query):
        logger.info(f"Querying LLM with: {query}")
        try:
            # Add system prompt to encourage tool usage
            system_prompt = """You are Maxwell, an intelligent voice assistant. You have access to various tools.
            When appropriate, use these tools to provide better responses.
            
            Available tools:
            """
            
            # Add tool descriptions if MCP is enabled
            if hasattr(self.assistant, 'mcp_provider'):
                tool_descriptions = self.assistant.mcp_provider.get_tool_descriptions()
                for name, description in tool_descriptions.items():
                    system_prompt += f"- {name}: {description}\n"
                    
            system_prompt += "\nTo use a tool, include [TOOL:tool_name(param1=value1, param2=value2)] in your response."
            system_prompt += "\nAlways respond in a helpful, concise, and conversational manner."
            
            response = self.ollama_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            content = response['message']['content']
            
            # Process tool calls in the response
            import re
            tool_pattern = r'\[TOOL:(\w+)\((.*?)\)\]'
            
            def replace_tool_call(match):
                tool_name = match.group(1)
                args_str = match.group(2)
                
                # Parse arguments
                kwargs = {}
                if args_str:
                    for arg in args_str.split(','):
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            kwargs[key.strip()] = value.strip().strip('"\'')
                
                # Execute the tool
                result = self.use_tool(tool_name, **kwargs)
                return result
                
            # Replace all tool calls with their results
            processed_content = re.sub(tool_pattern, replace_tool_call, content)
            return processed_content
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
            
    def search(self, query):
        if not query:
            self.assistant.speak("Please specify what you'd like to search for.")
            return
            
        self.assistant.speak(f"Searching for {query}...")
        try:
            # Simple search using DuckDuckGo
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url)
            data = response.json()
            
            if data.get("Abstract"):
                self.assistant.speak(data["Abstract"])
            else:
                self.assistant.speak("I couldn't find a clear answer. Let me ask the language model.")
                llm_response = self.query_llm(f"Please provide information about: {query}")
                self.assistant.speak(llm_response)
        except Exception as e:
            logger.error(f"Search error: {e}")
            self.assistant.speak("I encountered an error while searching. Let me try the language model instead.")
            llm_response = self.query_llm(f"Please provide information about: {query}")
            self.assistant.speak(llm_response)
            
    def get_weather(self, args):
        # In a real implementation, you would use a weather API
        self.assistant.speak("I'm sorry, I don't have access to weather data yet. This would require an API key for a weather service.")
        
    def get_time(self, args):
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.assistant.speak(f"The current time is {current_time}")
        
    def get_date(self, args):
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        self.assistant.speak(f"Today is {current_date}")
        
    def get_news(self, args):
        self.assistant.speak("I'm sorry, I don't have access to news data yet. This would require an API key for a news service.")
        
    def tell_joke(self, args):
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
        joke = random.choice(jokes)
        self.assistant.speak(joke)
        
    def set_reminder(self, text):
        if not text:
            self.assistant.speak("Please specify what you'd like me to remind you about.")
            return
            
        # Simple reminder implementation
        now = datetime.datetime.now()
        reminder_id = len(self.reminders) + 1
        self.reminders[reminder_id] = {
            "text": text,
            "created": now
        }
        self.assistant.speak(f"I've set a reminder for: {text}")
        
    def set_timer(self, duration_text):
        if not duration_text:
            self.assistant.speak("Please specify a duration for the timer.")
            return
            
        try:
            # Parse duration (simple implementation)
            duration = 0
            if "minute" in duration_text or "minutes" in duration_text:
                parts = duration_text.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i+1 < len(parts) and "minute" in parts[i+1]:
                        duration = int(part) * 60
                        break
            elif "second" in duration_text or "seconds" in duration_text:
                parts = duration_text.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i+1 < len(parts) and "second" in parts[i+1]:
                        duration = int(part)
                        break
            
            if duration <= 0:
                self.assistant.speak("I couldn't understand the duration. Please specify like '5 minutes' or '30 seconds'.")
                return
                
            timer_id = len(self.timers) + 1
            self.assistant.speak(f"Timer set for {duration_text}.")
            
            # Start timer in a separate thread
            timer_thread = threading.Thread(
                target=self._run_timer,
                args=(timer_id, duration, duration_text),
                daemon=True
            )
            timer_thread.start()
            
            self.timers[timer_id] = {
                "thread": timer_thread,
                "duration": duration,
                "text": duration_text,
                "start_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Timer error: {e}")
            self.assistant.speak("I had trouble setting that timer.")
            
    def _run_timer(self, timer_id, duration, duration_text):
        time.sleep(duration)
        if timer_id in self.timers:
            self.assistant.speak(f"Your timer for {duration_text} has finished!")
            del self.timers[timer_id]
            
    def show_help(self, args):
        help_text = "Here are the commands I understand: "
        for cmd in self.available_commands.keys():
            help_text += f"{cmd}, "
        help_text = help_text[:-2]  # Remove trailing comma and space
        self.assistant.speak(help_text)
        
    def use_tool(self, tool_name, **kwargs):
        """Use a tool from the MCP tool provider"""
        if hasattr(self.assistant, 'mcp_provider'):
            result = self.assistant.mcp_provider.execute_tool(tool_name, **kwargs)
            return result
        else:
            return f"Tool {tool_name} not available (MCP not enabled)" 