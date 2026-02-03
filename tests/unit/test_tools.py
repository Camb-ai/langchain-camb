"""Unit tests for CAMB AI tools."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from langchain_camb import (
    CambAudioSeparationTool,
    CambTextToSoundTool,
    CambTranscriptionTool,
    CambTranslatedTTSTool,
    CambTranslationTool,
    CambTTSTool,
    CambVoiceCloneTool,
    CambVoiceListTool,
    TTSInput,
    TranscriptionInput,
    TranslationInput,
)


@pytest.fixture(autouse=True)
def set_api_key():
    """Set API key for all tests."""
    with patch.dict(os.environ, {"CAMB_API_KEY": "test-api-key"}):
        yield


class TestTTSInput:
    """Tests for TTSInput schema."""

    def test_valid_input(self):
        """Test valid TTS input."""
        input_data = TTSInput(
            text="Hello, world!",
            language="en-us",
            voice_id=147320,
        )
        assert input_data.text == "Hello, world!"
        assert input_data.language == "en-us"
        assert input_data.voice_id == 147320

    def test_text_too_short(self):
        """Test that text must be at least 3 characters."""
        with pytest.raises(ValidationError):
            TTSInput(text="Hi", language="en-us", voice_id=147320)

    def test_text_too_long(self):
        """Test that text must be at most 3000 characters."""
        with pytest.raises(ValidationError):
            TTSInput(text="x" * 3001, language="en-us", voice_id=147320)

    def test_default_values(self):
        """Test default values are applied."""
        input_data = TTSInput(text="Hello, world!")
        assert input_data.language == "en-us"
        assert input_data.voice_id == 147320
        assert input_data.speech_model == "mars-flash"
        assert input_data.output_format == "file_path"
        assert input_data.speed == 1.0

    def test_speed_bounds(self):
        """Test speed must be between 0.5 and 2.0."""
        with pytest.raises(ValidationError):
            TTSInput(text="Hello", speed=0.4)
        with pytest.raises(ValidationError):
            TTSInput(text="Hello", speed=2.1)


class TestTranslationInput:
    """Tests for TranslationInput schema."""

    def test_valid_input(self):
        """Test valid translation input."""
        input_data = TranslationInput(
            text="Hello",
            source_language=1,
            target_language=2,
        )
        assert input_data.text == "Hello"
        assert input_data.source_language == 1
        assert input_data.target_language == 2

    def test_formality_options(self):
        """Test formality accepts valid options."""
        input_data = TranslationInput(
            text="Hello",
            source_language=1,
            target_language=2,
            formality="formal",
        )
        assert input_data.formality == "formal"


class TestTranscriptionInput:
    """Tests for TranscriptionInput schema."""

    def test_valid_url(self):
        """Test valid audio URL input."""
        input_data = TranscriptionInput(
            language=1,
            audio_url="https://example.com/audio.mp3",
        )
        assert input_data.audio_url == "https://example.com/audio.mp3"

    def test_valid_file_path(self):
        """Test valid audio file path input."""
        input_data = TranscriptionInput(
            language=1,
            audio_file_path="/path/to/audio.mp3",
        )
        assert input_data.audio_file_path == "/path/to/audio.mp3"

    def test_must_provide_audio_source(self):
        """Test that at least one audio source is required."""
        with pytest.raises(ValidationError):
            TranscriptionInput(language=1)

    def test_cannot_provide_both_sources(self):
        """Test that only one audio source can be provided."""
        with pytest.raises(ValidationError):
            TranscriptionInput(
                language=1,
                audio_url="https://example.com/audio.mp3",
                audio_file_path="/path/to/audio.mp3",
            )


class TestCambTTSTool:
    """Tests for CambTTSTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = CambTTSTool()
        assert tool.name == "camb_tts"
        assert "text to speech" in tool.description.lower()

    def test_run_returns_file_path(self):
        """Test _run returns file path."""
        # Mock the TTS streaming response
        mock_client = MagicMock()
        mock_client.text_to_speech.tts.return_value = iter([b"audio_data"])

        tool = CambTTSTool()
        tool._sync_client = mock_client

        result = tool._run(text="Hello, world!", language="en-us", voice_id=147320)

        assert isinstance(result, str)
        assert result.endswith(".wav")


class TestCambVoiceListTool:
    """Tests for CambVoiceListTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = CambVoiceListTool()
        assert tool.name == "camb_voice_list"
        assert "voice" in tool.description.lower()

    def test_gender_to_string(self):
        """Test gender integer to string conversion."""
        assert CambVoiceListTool._gender_to_string(0) == "not_specified"
        assert CambVoiceListTool._gender_to_string(1) == "male"
        assert CambVoiceListTool._gender_to_string(2) == "female"
        assert CambVoiceListTool._gender_to_string(9) == "not_applicable"
        assert CambVoiceListTool._gender_to_string(99) == "unknown"

    def test_run_returns_json(self):
        """Test _run returns valid JSON."""
        mock_voice = MagicMock()
        mock_voice.id = 1
        mock_voice.voice_name = "Test Voice"
        mock_voice.gender = 1
        mock_voice.age = 30
        mock_voice.language = "en-us"

        mock_client = MagicMock()
        mock_client.voice_cloning.list_voices.return_value = [mock_voice]

        tool = CambVoiceListTool()
        tool._sync_client = mock_client

        result = tool._run()
        parsed = json.loads(result)

        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["id"] == 1
        assert parsed[0]["name"] == "Test Voice"
        assert parsed[0]["gender"] == "male"


class TestCambTranslationTool:
    """Tests for CambTranslationTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = CambTranslationTool()
        assert tool.name == "camb_translation"
        assert "translate" in tool.description.lower()


class TestCambTranscriptionTool:
    """Tests for CambTranscriptionTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = CambTranscriptionTool()
        assert tool.name == "camb_transcription"
        assert "transcribe" in tool.description.lower()


class TestCambTranslatedTTSTool:
    """Tests for CambTranslatedTTSTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = CambTranslatedTTSTool()
        assert tool.name == "camb_translated_tts"
        assert "translate" in tool.description.lower()


class TestCambVoiceCloneTool:
    """Tests for CambVoiceCloneTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = CambVoiceCloneTool()
        assert tool.name == "camb_voice_clone"
        assert "clone" in tool.description.lower()


class TestCambTextToSoundTool:
    """Tests for CambTextToSoundTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = CambTextToSoundTool()
        assert tool.name == "camb_text_to_sound"
        assert "sound" in tool.description.lower() or "music" in tool.description.lower()


class TestCambAudioSeparationTool:
    """Tests for CambAudioSeparationTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = CambAudioSeparationTool()
        assert tool.name == "camb_audio_separation"
        assert "separate" in tool.description.lower()


class TestApiKeyValidation:
    """Tests for API key validation."""

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("CAMB_API_KEY", None)
            with pytest.raises(ValueError, match="API key is required"):
                CambTTSTool()

    def test_api_key_from_parameter(self):
        """Test API key can be passed as parameter."""
        tool = CambTTSTool(api_key="my-api-key")
        assert tool.api_key == "my-api-key"

    def test_api_key_from_environment(self):
        """Test API key is read from environment."""
        with patch.dict(os.environ, {"CAMB_API_KEY": "env-api-key"}):
            tool = CambTTSTool()
            assert tool.api_key == "env-api-key"
