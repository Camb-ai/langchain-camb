"""CAMB AI LangChain tools."""

from langchain_camb.tools.audio_separation import (
    AudioSeparationInput,
    CambAudioSeparationTool,
)
from langchain_camb.tools.base import CambBaseTool
from langchain_camb.tools.text_to_sound import CambTextToSoundTool, TextToSoundInput
from langchain_camb.tools.transcription import CambTranscriptionTool, TranscriptionInput
from langchain_camb.tools.translated_tts import (
    CambTranslatedTTSTool,
    TranslatedTTSInput,
)
from langchain_camb.tools.translation import CambTranslationTool, TranslationInput
from langchain_camb.tools.tts import CambTTSTool, TTSInput
from langchain_camb.tools.voice_clone import CambVoiceCloneTool, VoiceCloneInput
from langchain_camb.tools.voice_list import CambVoiceListTool, VoiceListInput

__all__ = [
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
