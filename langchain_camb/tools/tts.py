"""Text-to-Speech tool for CAMB AI."""

from __future__ import annotations

import base64
import tempfile
from typing import Any, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field

from langchain_camb.tools.base import CambBaseTool


class TTSInput(BaseModel):
    """Input schema for Text-to-Speech tool."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=3000,
        description="Text to convert to speech (3-3000 characters).",
    )
    language: str = Field(
        default="en-us",
        description="BCP-47 language code (e.g., 'en-us', 'es-es', 'fr-fr').",
    )
    voice_id: int = Field(
        default=147320,
        description="Voice ID to use. Get available voices with CambVoiceListTool.",
    )
    speech_model: str = Field(
        default="mars-flash",
        description="Speech model: 'mars-flash' (fast), 'mars-pro' (quality), 'mars-instruct' (with instructions).",
    )
    output_format: Literal["file_path", "base64", "bytes"] = Field(
        default="file_path",
        description="Output format: 'file_path' (save to file), 'base64' (encoded string), 'bytes' (raw bytes).",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5-2.0).",
    )
    user_instructions: Optional[str] = Field(
        default=None,
        description="Instructions for mars-instruct model (e.g., 'Speak with excitement').",
    )


class CambTTSTool(CambBaseTool):
    """Tool for converting text to speech using CAMB AI.

    This tool uses CAMB AI's streaming TTS API to convert text into natural
    speech in 140+ languages. Supports multiple voice models and output formats.

    Example:
        ```python
        from langchain_camb import CambTTSTool

        tts = CambTTSTool()
        result = tts.invoke({
            "text": "Hello, world!",
            "language": "en-us",
            "voice_id": 147320
        })
        print(result)  # Returns file path to audio
        ```
    """

    name: str = "camb_tts"
    description: str = (
        "Convert text to speech using CAMB AI. "
        "Supports 140+ languages and multiple voice models. "
        "Returns audio as file path, base64, or raw bytes."
    )
    args_schema: Type[BaseModel] = TTSInput

    def _run(
        self,
        text: str,
        language: str = "en-us",
        voice_id: int = 147320,
        speech_model: str = "mars-flash",
        output_format: str = "file_path",
        speed: float = 1.0,
        user_instructions: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, bytes]:
        """Run text-to-speech conversion synchronously.

        Returns:
            File path, base64 string, or raw bytes depending on output_format.
        """
        from camb import (
            StreamTtsOutputConfiguration,
            StreamTtsVoiceSettings,
        )

        # Build request parameters
        kwargs: dict[str, Any] = {
            "text": text,
            "language": language,
            "voice_id": voice_id,
            "speech_model": speech_model,
            "output_configuration": StreamTtsOutputConfiguration(format="wav"),
            "voice_settings": StreamTtsVoiceSettings(speed=speed),
        }

        if user_instructions and speech_model == "mars-instruct":
            kwargs["user_instructions"] = user_instructions

        # Stream audio chunks
        audio_chunks: list[bytes] = []
        for chunk in self.sync_client.text_to_speech.tts(**kwargs):
            audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)

        return self._format_output(audio_data, output_format)

    async def _arun(
        self,
        text: str,
        language: str = "en-us",
        voice_id: int = 147320,
        speech_model: str = "mars-flash",
        output_format: str = "file_path",
        speed: float = 1.0,
        user_instructions: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, bytes]:
        """Run text-to-speech conversion asynchronously.

        Returns:
            File path, base64 string, or raw bytes depending on output_format.
        """
        from camb import (
            StreamTtsOutputConfiguration,
            StreamTtsVoiceSettings,
        )

        # Build request parameters
        kwargs: dict[str, Any] = {
            "text": text,
            "language": language,
            "voice_id": voice_id,
            "speech_model": speech_model,
            "output_configuration": StreamTtsOutputConfiguration(format="wav"),
            "voice_settings": StreamTtsVoiceSettings(speed=speed),
        }

        if user_instructions and speech_model == "mars-instruct":
            kwargs["user_instructions"] = user_instructions

        # Stream audio chunks
        audio_chunks: list[bytes] = []
        async for chunk in self.async_client.text_to_speech.tts(**kwargs):
            audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)

        return self._format_output(audio_data, output_format)

    def _format_output(
        self, audio_data: bytes, output_format: str
    ) -> Union[str, bytes]:
        """Format audio data according to output_format."""
        if output_format == "bytes":
            return audio_data
        elif output_format == "base64":
            return base64.b64encode(audio_data).decode("utf-8")
        else:  # file_path
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as f:
                f.write(audio_data)
                return f.name
