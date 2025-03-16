import requests
import datetime
import json
import threading
import time
import ollama
from utils import setup_logger

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
            response = self.ollama_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": query}]
            )
            return response['message']['content']
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