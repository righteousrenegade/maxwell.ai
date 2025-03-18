# Maxwell Configuration System

This document explains how to configure Maxwell using either YAML or .env files.

## Quick Start

1. Copy `config.yaml.example` to `config.yaml` and edit as needed
2. Or copy `.env.example` to `.env` and edit as needed
3. Run Maxwell normally - it will automatically load your configuration

## Configuration Methods

Maxwell supports three ways to configure settings (in order of precedence):

1. **Environment Variables** (highest priority)
2. **.env File** (middle priority)
3. **config.yaml File** (lowest priority)

This means that if a setting is defined in multiple places, the environment variable takes precedence over the .env file, which takes precedence over the YAML file.

## Configuration Files

### YAML Configuration (config.yaml)

The YAML configuration file uses a structured format with sections. Example:

```yaml
# General settings
general:
  wake_word: "hello maxwell"
  voice: "bm_lewis"
  speed: 1.15

# LLM Provider Configuration
llm:
  provider: "openai"  # "ollama" or "openai"
  
# OpenAI Configuration
openai:
  api_key: "YOUR_API_KEY_HERE"
  base_url: "https://api.openai.com/v1"
  model: "gpt-3.5-turbo"

# Kokoro TTS Configuration
tts:
  model_path: "kokoro-v1.0.onnx"
  voices_path: "voices-v1.0.bin"
```

### Environment Variable File (.env)

The .env file uses simple KEY=VALUE pairs:

```
# OpenAI Settings
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=http://localhost:1234/v1

# LLM Settings
LLM_PROVIDER=openai

# TTS Settings
TTS_MODEL_PATH=kokoro-v1.0.onnx
TTS_VOICES_PATH=voices-v1.0.bin
```

## Important Settings

### TTS Configuration

The Kokoro TTS engine requires these settings:

```yaml
# In config.yaml
tts:
  model_path: "kokoro-v1.0.onnx"
  voices_path: "voices-v1.0.bin"
```

Or in .env:
```
TTS_MODEL_PATH=kokoro-v1.0.onnx
TTS_VOICES_PATH=voices-v1.0.bin
```

### OpenAI Configuration

To use OpenAI, you need these settings:

```yaml
# In config.yaml
llm:
  provider: "openai"
  
openai:
  api_key: "YOUR_API_KEY_HERE"
  base_url: "https://api.openai.com/v1"  # Or your local endpoint
  model: "gpt-3.5-turbo"
```

Or in .env:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
```

## Testing Your Configuration

Run the configuration example to see what settings are being loaded:

```
python config_example.py
```

This will show which settings are loaded and from which source.

## Debugging Configuration Issues

The system logs detailed information about where each configuration value comes from. Enable debug logging to see all configuration details:

```python
logging.basicConfig(level=logging.DEBUG)
```

Then check the log output for lines starting with `[CONFIG]` to see where each setting is coming from. 