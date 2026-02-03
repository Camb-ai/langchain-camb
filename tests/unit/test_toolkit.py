"""Unit tests for CambToolkit."""

import os
from unittest.mock import patch

import pytest

from langchain_camb import (
    CambAudioSeparationTool,
    CambTextToSoundTool,
    CambToolkit,
    CambTranscriptionTool,
    CambTranslatedTTSTool,
    CambTranslationTool,
    CambTTSTool,
    CambVoiceCloneTool,
    CambVoiceListTool,
)


@pytest.fixture(autouse=True)
def set_api_key():
    """Set API key for all tests."""
    with patch.dict(os.environ, {"CAMB_API_KEY": "test-api-key"}):
        yield


class TestCambToolkit:
    """Tests for CambToolkit."""

    def test_get_all_tools(self):
        """Test getting all tools."""
        toolkit = CambToolkit()
        tools = toolkit.get_tools()

        assert len(tools) == 8

        tool_names = {tool.name for tool in tools}
        expected_names = {
            "camb_tts",
            "camb_translated_tts",
            "camb_translation",
            "camb_transcription",
            "camb_voice_list",
            "camb_voice_clone",
            "camb_text_to_sound",
            "camb_audio_separation",
        }
        assert tool_names == expected_names

    def test_filter_tools(self):
        """Test filtering tools with include flags."""
        toolkit = CambToolkit(
            include_tts=True,
            include_translated_tts=False,
            include_translation=True,
            include_transcription=False,
            include_voice_list=True,
            include_voice_clone=False,
            include_text_to_sound=False,
            include_audio_separation=False,
        )
        tools = toolkit.get_tools()

        assert len(tools) == 3

        tool_types = {type(tool) for tool in tools}
        assert CambTTSTool in tool_types
        assert CambTranslationTool in tool_types
        assert CambVoiceListTool in tool_types

    def test_api_key_passed_to_tools(self):
        """Test that API key is passed to all tools."""
        toolkit = CambToolkit(api_key="custom-key")
        tools = toolkit.get_tools()

        for tool in tools:
            assert tool.api_key == "custom-key"

    def test_base_url_passed_to_tools(self):
        """Test that base_url is passed to all tools."""
        toolkit = CambToolkit(base_url="https://custom.api.com")
        tools = toolkit.get_tools()

        for tool in tools:
            assert tool.base_url == "https://custom.api.com"

    def test_timeout_passed_to_tools(self):
        """Test that timeout is passed to all tools."""
        toolkit = CambToolkit(timeout=120.0)
        tools = toolkit.get_tools()

        for tool in tools:
            assert tool.timeout == 120.0

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CAMB_API_KEY", None)
            toolkit = CambToolkit()
            with pytest.raises(ValueError, match="API key is required"):
                toolkit.get_tools()

    def test_only_tts_toolkit(self):
        """Test creating a TTS-only toolkit."""
        toolkit = CambToolkit(
            include_tts=True,
            include_translated_tts=False,
            include_translation=False,
            include_transcription=False,
            include_voice_list=True,  # Useful for finding voices
            include_voice_clone=False,
            include_text_to_sound=False,
            include_audio_separation=False,
        )
        tools = toolkit.get_tools()

        assert len(tools) == 2
        tool_types = {type(tool) for tool in tools}
        assert CambTTSTool in tool_types
        assert CambVoiceListTool in tool_types

    def test_translation_toolkit(self):
        """Test creating a translation-focused toolkit."""
        toolkit = CambToolkit(
            include_tts=False,
            include_translated_tts=True,
            include_translation=True,
            include_transcription=True,
            include_voice_list=False,
            include_voice_clone=False,
            include_text_to_sound=False,
            include_audio_separation=False,
        )
        tools = toolkit.get_tools()

        assert len(tools) == 3
        tool_names = {tool.name for tool in tools}
        assert "camb_translated_tts" in tool_names
        assert "camb_translation" in tool_names
        assert "camb_transcription" in tool_names
