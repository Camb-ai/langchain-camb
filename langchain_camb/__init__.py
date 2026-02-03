"""LangChain integration for CAMB AI.

CAMB AI provides multilingual audio and localization services including:
- Text-to-Speech (140+ languages)
- Translation
- Transcription with speaker identification
- Voice cloning
- Text-to-Sound generation
- Audio separation

Example:
    ```python
    from langchain_camb import CambToolkit, CambTTSTool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    # Use individual tools
    tts = CambTTSTool()
    audio_path = tts.invoke({
        "text": "Hello, world!",
        "language": "en-us",
        "voice_id": 147320
    })

    # Or use the toolkit with an agent
    toolkit = CambToolkit()
    agent = create_react_agent(ChatOpenAI(), toolkit.get_tools())
    agent.invoke({
        "messages": [{"role": "user", "content": "Say hello in Spanish"}]
    })
    ```
"""

from langchain_camb.tools import (
    AudioSeparationInput,
    CambAudioSeparationTool,
    CambBaseTool,
    CambTextToSoundTool,
    CambTranscriptionTool,
    CambTranslatedTTSTool,
    CambTranslationTool,
    CambTTSTool,
    CambVoiceCloneTool,
    CambVoiceListTool,
    TextToSoundInput,
    TranscriptionInput,
    TranslatedTTSInput,
    TranslationInput,
    TTSInput,
    VoiceCloneInput,
    VoiceListInput,
)
from langchain_camb.toolkits import CambToolkit
from langchain_camb.version import __version__

__all__ = [
    # Version
    "__version__",
    # Toolkit
    "CambToolkit",
    # Base
    "CambBaseTool",
    # Tools
    "CambTTSTool",
    "CambTranslatedTTSTool",
    "CambTranslationTool",
    "CambTranscriptionTool",
    "CambVoiceListTool",
    "CambVoiceCloneTool",
    "CambTextToSoundTool",
    "CambAudioSeparationTool",
    # Input schemas
    "TTSInput",
    "TranslatedTTSInput",
    "TranslationInput",
    "TranscriptionInput",
    "VoiceListInput",
    "VoiceCloneInput",
    "TextToSoundInput",
    "AudioSeparationInput",
]
