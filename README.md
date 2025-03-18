# Maxwell Voice Assistant

A modular voice assistant powered by Ollama LLM and Kokoro TTS.

## DEMO

[![Watch the demo!](media/maxwelldemo.mp4)](media/maxwelldemo.mp4)
![Maxwell.AI](media/maxwellai1.png)

## Features

- **Voice Recognition**: Recognizes speech with wake word detection
- **Text-to-Speech**: Uses Kokoro TTS for natural-sounding responses
- **LLM Integration**: Connects to Ollama for intelligent responses
- **Conversation Mode**: Stays active until explicitly ended
- **Command Execution**: Built-in commands including web search
- **Interrupt Capability**: Can be interrupted while speaking
- **Offline Mode**: Optional offline speech recognition with Vosk

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mini-bot.git
   cd mini-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Ollama:
   - Follow instructions at [Ollama's website](https://ollama.ai/)
   - Start the Ollama server

## Usage

### Basic Usage

Run the assistant with default settings:

```
python assistant.py
```

### Command-line Options

The assistant supports many command-line options:

```
python assistant.py --help
```

### Voice Commands

- Say **"Hey Maxwell"** (or your custom wake word) to activate
- Say **"execute [command]"** to run a command:
  - `execute search [query]`: Search the web
  - `execute weather`: Get the weather
  - `execute time`: Get the current time
  - `execute date`: Get the current date
  - `execute news`: Get news headlines
  - `execute joke`: Tell a joke
  - `execute reminder [text]`: Set a reminder
  - `execute timer [duration]`: Set a timer
- Say **"stop talking"** (or your custom interrupt word) to interrupt
- Say **"end conversation"** to exit conversation mode

## Project Structure

The project has been refactored into a modular structure:

- `assistant.py`: Main assistant class and entry point

## Customization

### Adding New Commands

To add new commands, modify the `commands.py` file:

1. Add your command to the `available_commands` dictionary in `utils.py`
2. Add a method to handle the command in the `CommandExecutor` class in `commands.py`
3. Add a condition in the `execute_command` method to call your new method

### Changing TTS Voice

Use the `--voice` option to change the TTS voice:

```
python assistant.py --voice "bm_lewis"
```

List available voices:

```
python assistant.py --list-voices
```

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running (`ollama serve`)
- **TTS Issues**: The TTS models will be downloaded on first run
- **Offline Recognition**: Vosk models will be downloaded if not present

## License

This project is licensed under the MIT License - see the LICENSE file for details.
