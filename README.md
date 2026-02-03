# langchain-camb

LangChain integration for [CAMB AI](https://camb.ai) - multilingual audio and localization services supporting 140+ languages.

## Installation

```bash
pip install langchain-camb
```

## Quick Start

```python
from langchain_camb import CambToolkit, CambTTSTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Set your API key
import os
os.environ["CAMB_API_KEY"] = "your-api-key"

# Use individual tools
tts = CambTTSTool()
audio_path = tts.invoke({
    "text": "Hello, world!",
    "language": "en-us",
    "voice_id": 147320
})
print(f"Audio saved to: {audio_path}")

# Or use the toolkit with a LangChain agent
toolkit = CambToolkit()
agent = create_react_agent(ChatOpenAI(), toolkit.get_tools())

result = agent.invoke({
    "messages": [{"role": "user", "content": "Say 'Good morning' in Spanish"}]
})
```

## Available Tools

| Tool | Description |
|------|-------------|
| `CambTTSTool` | Text-to-Speech - Convert text to natural speech |
| `CambTranslatedTTSTool` | Translate text and convert to speech in one step |
| `CambTranslationTool` | Text translation between 140+ languages |
| `CambTranscriptionTool` | Speech-to-text with speaker identification |
| `CambVoiceListTool` | List available voices for TTS |
| `CambVoiceCloneTool` | Clone voices from 2+ second audio samples |
| `CambTextToSoundTool` | Generate music and sound effects from text |
| `CambAudioSeparationTool` | Separate vocals from background audio |

## Tool Examples

### Text-to-Speech

```python
from langchain_camb import CambTTSTool

tts = CambTTSTool()
result = tts.invoke({
    "text": "Hello, how are you today?",
    "language": "en-us",
    "voice_id": 147320,
    "speech_model": "mars-flash",  # or "mars-pro", "mars-instruct"
    "speed": 1.0,  # 0.5 to 2.0
    "output_format": "file_path"  # or "base64", "bytes"
})
```

### Translation

```python
from langchain_camb import CambTranslationTool

translator = CambTranslationTool()
result = translator.invoke({
    "text": "Hello, how are you?",
    "source_language": 1,   # English (en-us)
    "target_language": 54,  # Spanish (es-es)
    "formality": 1  # 1=formal, 2=informal
})
print(result)  # "Hola, ¿cómo está usted?"
```

### Transcription

```python
from langchain_camb import CambTranscriptionTool

transcriber = CambTranscriptionTool()
result = transcriber.invoke({
    "audio_url": "https://example.com/audio.mp3",
    "language": 1  # English
})
# Returns JSON with text, segments, and speaker identification
```

### Voice Cloning

```python
from langchain_camb import CambVoiceCloneTool

cloner = CambVoiceCloneTool()
result = cloner.invoke({
    "voice_name": "My Custom Voice",
    "audio_file_path": "/path/to/sample.wav",  # 2+ seconds
    "gender": 2,  # 1=Male, 2=Female
    "description": "A warm, friendly voice"
})
# Returns new voice_id to use with TTS
```

### Text-to-Sound

```python
from langchain_camb import CambTextToSoundTool

sound_gen = CambTextToSoundTool()
result = sound_gen.invoke({
    "prompt": "Upbeat electronic music with a driving beat",
    "duration": 30,
    "audio_type": "music"  # or "sound"
})
```

### Audio Separation

```python
from langchain_camb import CambAudioSeparationTool

separator = CambAudioSeparationTool()
result = separator.invoke({
    "audio_file_path": "/path/to/mixed_audio.mp3"
})
# Returns JSON with paths to vocals and background audio
```

## Using the Toolkit

The `CambToolkit` bundles all tools and allows filtering:

```python
from langchain_camb import CambToolkit

# All tools
toolkit = CambToolkit()
tools = toolkit.get_tools()

# TTS-focused toolkit
tts_toolkit = CambToolkit(
    include_tts=True,
    include_voice_list=True,
    include_translation=False,
    include_transcription=False,
    include_translated_tts=False,
    include_voice_clone=False,
    include_text_to_sound=False,
    include_audio_separation=False,
)

# Translation-focused toolkit
translation_toolkit = CambToolkit(
    include_tts=False,
    include_translated_tts=True,
    include_translation=True,
    include_transcription=True,
    include_voice_list=False,
    include_voice_clone=False,
    include_text_to_sound=False,
    include_audio_separation=False,
)
```

## Language Codes

CAMB AI uses integer language codes. Common codes:

| Code | Language | BCP-47 |
|------|----------|--------|
| 1 | English (US) | en-us |
| 31 | German (Germany) | de-de |
| 54 | Spanish (Spain) | es-es |
| 76 | French (France) | fr-fr |
| 87 | Italian | it-it |
| 88 | Japanese | ja-jp |
| 94 | Korean | ko-kr |
| 108 | Dutch | nl-nl |
| 111 | Portuguese (Brazil) | pt-br |
| 114 | Russian | ru-ru |
| 139 | Chinese (Simplified) | zh-cn |

Use `client.languages.get_source_languages()` for the full list of 140+ languages.

For TTS, use BCP-47 codes like `"en-us"`, `"es-es"`, `"fr-fr"`.

## Configuration

### API Key

Set your API key via environment variable or parameter:

```python
# Environment variable
import os
os.environ["CAMB_API_KEY"] = "your-api-key"

# Or pass directly
tool = CambTTSTool(api_key="your-api-key")
toolkit = CambToolkit(api_key="your-api-key")
```

### Timeouts and Polling

For async operations (transcription, text-to-sound, etc.):

```python
tool = CambTranscriptionTool(
    timeout=60.0,         # HTTP request timeout
    max_poll_attempts=60, # Max polling attempts
    poll_interval=2.0,    # Seconds between polls
)
```

## Agent Integration

### LangGraph ReAct Agent

```python
from langchain_camb import CambToolkit
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

toolkit = CambToolkit()
agent = create_react_agent(ChatOpenAI(model="gpt-4"), toolkit.get_tools())

# TTS
result = agent.invoke({
    "messages": [{"role": "user", "content": "Say 'Hello world' in a friendly tone"}]
})

# Translation
result = agent.invoke({
    "messages": [{"role": "user", "content": "Translate 'Good morning' to French and German"}]
})

# Transcription
result = agent.invoke({
    "messages": [{"role": "user", "content": "Transcribe https://example.com/audio.mp3"}]
})
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/unit/

# Run integration tests (requires CAMB_API_KEY)
CAMB_API_KEY=your-key pytest tests/integration/ -m integration

# Lint
ruff check .

# Type check
mypy langchain_camb/
```

## License

MIT
