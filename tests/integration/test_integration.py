"""Integration tests for CAMB AI tools.

These tests require a valid CAMB_API_KEY environment variable.
Run with: pytest tests/integration/ -m integration
"""

import json
import os

import pytest

from langchain_camb import (
    CambToolkit,
    CambTranslationTool,
    CambTTSTool,
    CambVoiceListTool,
)


# Skip all tests in this module if no API key is set
pytestmark = pytest.mark.integration


@pytest.fixture
def api_key():
    """Get API key from environment."""
    key = os.environ.get("CAMB_API_KEY")
    if not key:
        pytest.skip("CAMB_API_KEY not set")
    return key


class TestVoiceListIntegration:
    """Integration tests for voice list tool."""

    def test_list_voices(self, api_key):
        """Test listing available voices."""
        tool = CambVoiceListTool(api_key=api_key)
        result = tool.invoke({})

        # Should return valid JSON
        voices = json.loads(result)
        assert isinstance(voices, list)
        assert len(voices) > 0

        # Each voice should have required fields
        for voice in voices[:5]:  # Check first 5
            assert "id" in voice
            assert "name" in voice
            assert "gender" in voice


class TestTTSIntegration:
    """Integration tests for TTS tool."""

    def test_tts_basic(self, api_key):
        """Test basic TTS functionality."""
        tool = CambTTSTool(api_key=api_key)
        result = tool.invoke(
            {
                "text": "Hello, this is a test.",
                "language": "en-us",
                "voice_id": 147320,
                "output_format": "file_path",
            }
        )

        # Should return a file path
        assert isinstance(result, str)
        assert os.path.exists(result)

        # Clean up
        os.unlink(result)

    def test_tts_base64(self, api_key):
        """Test TTS with base64 output."""
        tool = CambTTSTool(api_key=api_key)
        result = tool.invoke(
            {
                "text": "Hello, this is a test.",
                "language": "en-us",
                "voice_id": 147320,
                "output_format": "base64",
            }
        )

        # Should return base64 string
        assert isinstance(result, str)
        # Base64 should be decodable
        import base64

        decoded = base64.b64decode(result)
        assert len(decoded) > 0


class TestTranslationIntegration:
    """Integration tests for translation tool."""

    def test_translate_english_to_spanish(self, api_key):
        """Test translating English to Spanish."""
        tool = CambTranslationTool(api_key=api_key)
        result = tool.invoke(
            {
                "text": "Hello, how are you?",
                "source_language": 1,  # English
                "target_language": 2,  # Spanish
            }
        )

        # Should return translated text
        assert isinstance(result, str)
        assert len(result) > 0
        # Spanish translation should contain common words
        result_lower = result.lower()
        assert any(
            word in result_lower
            for word in ["hola", "cómo", "estás", "qué", "tal"]
        )


class TestToolkitIntegration:
    """Integration tests for the toolkit."""

    def test_toolkit_get_tools(self, api_key):
        """Test getting tools from toolkit."""
        toolkit = CambToolkit(api_key=api_key)
        tools = toolkit.get_tools()

        assert len(tools) == 8

        # All tools should be configured with the API key
        for tool in tools:
            assert tool.api_key == api_key

    def test_toolkit_partial_tools(self, api_key):
        """Test getting partial tools from toolkit."""
        toolkit = CambToolkit(
            api_key=api_key,
            include_tts=True,
            include_voice_list=True,
            include_translation=False,
            include_transcription=False,
            include_translated_tts=False,
            include_voice_clone=False,
            include_text_to_sound=False,
            include_audio_separation=False,
        )
        tools = toolkit.get_tools()

        assert len(tools) == 2
        tool_names = {tool.name for tool in tools}
        assert "camb_tts" in tool_names
        assert "camb_voice_list" in tool_names


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Async integration tests."""

    async def test_async_voice_list(self, api_key):
        """Test async voice list."""
        tool = CambVoiceListTool(api_key=api_key)
        result = await tool.ainvoke({})

        voices = json.loads(result)
        assert isinstance(voices, list)
        assert len(voices) > 0

    async def test_async_tts(self, api_key):
        """Test async TTS."""
        tool = CambTTSTool(api_key=api_key)
        result = await tool.ainvoke(
            {
                "text": "Hello, async test.",
                "language": "en-us",
                "voice_id": 147320,
                "output_format": "base64",
            }
        )

        assert isinstance(result, str)
        assert len(result) > 0
