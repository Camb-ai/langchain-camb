"""Standard unit tests for CAMB AI tools using LangChain test framework.

These tests verify that the tools conform to LangChain's standard tool interface.
They run without making actual API calls.
"""

from typing import Type

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_camb import (
    CambAudioSeparationTool,
    CambTextToSoundTool,
    CambTranscriptionTool,
    CambTranslatedTTSTool,
    CambTranslationTool,
    CambTTSTool,
    CambVoiceCloneTool,
    CambVoiceListTool,
)


class TestCambTTSToolUnit(ToolsUnitTests):
    """Standard unit tests for CambTTSTool."""

    @property
    def tool_constructor(self) -> Type[CambTTSTool]:
        return CambTTSTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "text": "Hello, world!",
            "language": "en-us",
            "voice_id": 147320,
        }


class TestCambVoiceListToolUnit(ToolsUnitTests):
    """Standard unit tests for CambVoiceListTool."""

    @property
    def tool_constructor(self) -> Type[CambVoiceListTool]:
        return CambVoiceListTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {}


class TestCambTranslationToolUnit(ToolsUnitTests):
    """Standard unit tests for CambTranslationTool."""

    @property
    def tool_constructor(self) -> Type[CambTranslationTool]:
        return CambTranslationTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "text": "Hello",
            "source_language": 1,
            "target_language": 2,
        }


class TestCambTranscriptionToolUnit(ToolsUnitTests):
    """Standard unit tests for CambTranscriptionTool."""

    @property
    def tool_constructor(self) -> Type[CambTranscriptionTool]:
        return CambTranscriptionTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "language": 1,
            "audio_url": "https://example.com/audio.mp3",
        }


class TestCambTranslatedTTSToolUnit(ToolsUnitTests):
    """Standard unit tests for CambTranslatedTTSTool."""

    @property
    def tool_constructor(self) -> Type[CambTranslatedTTSTool]:
        return CambTranslatedTTSTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "text": "Hello",
            "source_language": 1,
            "target_language": 2,
            "voice_id": 147320,
        }


class TestCambVoiceCloneToolUnit(ToolsUnitTests):
    """Standard unit tests for CambVoiceCloneTool."""

    @property
    def tool_constructor(self) -> Type[CambVoiceCloneTool]:
        return CambVoiceCloneTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "voice_name": "test_voice",
            "audio_file_path": "/path/to/voice.mp3",
            "gender": 1,
        }


class TestCambTextToSoundToolUnit(ToolsUnitTests):
    """Standard unit tests for CambTextToSoundTool."""

    @property
    def tool_constructor(self) -> Type[CambTextToSoundTool]:
        return CambTextToSoundTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "prompt": "A thunderstorm with heavy rain",
        }


class TestCambAudioSeparationToolUnit(ToolsUnitTests):
    """Standard unit tests for CambAudioSeparationTool."""

    @property
    def tool_constructor(self) -> Type[CambAudioSeparationTool]:
        return CambAudioSeparationTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "audio_url": "https://example.com/audio.mp3",
        }
