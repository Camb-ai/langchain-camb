"""Translated TTS tool for CAMB AI."""

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


class TranslatedTTSInput(BaseModel):
    """Input schema for Translated TTS tool."""

    text: str = Field(
        ...,
        description="Text to translate and convert to speech.",
    )
    source_language: int = Field(
        ...,
        description="Source language code (integer). Common codes: 1=English, 2=Spanish, 3=French.",
    )
    target_language: int = Field(
        ...,
        description="Target language code (integer) for the output speech.",
    )
    voice_id: int = Field(
        default=147320,
        description="Voice ID for TTS. Get available voices with CambVoiceListTool.",
    )
    output_format: Literal["file_path", "base64"] = Field(
        default="file_path",
        description="Output format: 'file_path' or 'base64'.",
    )
    formality: Optional[int] = Field(
        default=None,
        description="Translation formality: 1=formal, 2=informal.",
    )


class CambTranslatedTTSTool(CambBaseTool):
    """Tool for translating text and converting to speech using CAMB AI.

    This tool combines translation and TTS in a single operation.
    It translates text to the target language and generates speech.

    Example:
        ```python
        from langchain_camb import CambTranslatedTTSTool

        translated_tts = CambTranslatedTTSTool()
        result = translated_tts.invoke({
            "text": "Hello, how are you?",
            "source_language": 1,  # English
            "target_language": 2,  # Spanish
            "voice_id": 147320
        })
        print(result)  # File path to Spanish audio
        ```
    """

    name: str = "camb_translated_tts"
    description: str = (
        "Translate text and convert to speech in one step. "
        "Provide source text, source language, target language, and voice ID. "
        "Returns audio file of the translated text spoken in the target language."
    )
    args_schema: Type[BaseModel] = TranslatedTTSInput

    def _run(
        self,
        text: str,
        source_language: int,
        target_language: int,
        voice_id: int = 147320,
        output_format: str = "file_path",
        formality: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Translate and convert to speech synchronously.

        Returns:
            File path or base64 encoded audio.
        """
        kwargs: dict[str, Any] = {
            "text": text,
            "voice_id": voice_id,
            "source_language": source_language,
            "target_language": target_language,
        }

        if formality:
            kwargs["formality"] = formality

        # Create translated TTS task
        result = self.sync_client.translated_tts.create_translated_tts(**kwargs)
        task_id = result.task_id

        # Poll for completion and get result from status
        status = self._poll_task_status_sync(
            self.sync_client.translated_tts.get_translated_tts_task_status,
            task_id,
        )

        # Get audio from status message (contains URL) or run_id
        audio_data, audio_format = self._get_audio_from_status(status)

        return self._format_output(audio_data, output_format, audio_format)

    async def _arun(
        self,
        text: str,
        source_language: int,
        target_language: int,
        voice_id: int = 147320,
        output_format: str = "file_path",
        formality: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Translate and convert to speech asynchronously.

        Returns:
            File path or base64 encoded audio.
        """
        kwargs: dict[str, Any] = {
            "text": text,
            "voice_id": voice_id,
            "source_language": source_language,
            "target_language": target_language,
        }

        if formality:
            kwargs["formality"] = formality

        # Create translated TTS task
        result = await self.async_client.translated_tts.create_translated_tts(**kwargs)
        task_id = result.task_id

        # Poll for completion and get result from status
        status = await self._poll_task_status(
            self.async_client.translated_tts.get_translated_tts_task_status,
            task_id,
        )

        # Get audio from status message (contains URL) or run_id
        audio_data, audio_format = await self._get_audio_from_status_async(status)

        return self._format_output(audio_data, output_format, audio_format)

    def _get_audio_from_status(self, status: Any) -> tuple[bytes, str]:
        """Extract audio from status response.

        Returns:
            Tuple of (audio_data, detected_format) where format is 'wav', 'mp3', 'flac', or 'pcm'.
        """
        import httpx

        # Get audio via run_id using the tts-result endpoint
        run_id = getattr(status, "run_id", None)
        if run_id:
            # Use direct API endpoint - more reliable than SDK method
            base_url = getattr(self.sync_client, '_client_wrapper', None)
            if base_url and hasattr(base_url, 'base_url'):
                result_url = f"{base_url.base_url}/tts-result/{run_id}"
            else:
                result_url = f"https://client.camb.ai/apis/tts-result/{run_id}"

            with httpx.Client() as client:
                response = client.get(
                    result_url,
                    headers={"x-api-key": self.api_key}
                )
                if response.status_code == 200:
                    audio_data = response.content
                    audio_format = self._detect_audio_format(
                        audio_data, response.headers.get("content-type", "")
                    )
                    return audio_data, audio_format

        # Fallback: check if message contains URL
        message = getattr(status, "message", None)
        if message:
            if isinstance(message, dict):
                url = message.get("output_url") or message.get("audio_url") or message.get("url")
            elif isinstance(message, str) and message.startswith("http"):
                url = message
            else:
                url = None

            if url:
                with httpx.Client() as client:
                    response = client.get(url)
                    audio_data = response.content
                    audio_format = self._detect_audio_format(
                        audio_data, response.headers.get("content-type", "")
                    )
                    return audio_data, audio_format

        return b"", "pcm"

    async def _get_audio_from_status_async(self, status: Any) -> tuple[bytes, str]:
        """Extract audio from status response (async).

        Returns:
            Tuple of (audio_data, detected_format) where format is 'wav', 'mp3', 'flac', or 'pcm'.
        """
        import httpx

        # Get audio via run_id using the tts-result endpoint
        run_id = getattr(status, "run_id", None)
        if run_id:
            # Use direct API endpoint - more reliable than SDK method
            base_url = getattr(self.async_client, '_client_wrapper', None)
            if base_url and hasattr(base_url, 'base_url'):
                result_url = f"{base_url.base_url}/tts-result/{run_id}"
            else:
                result_url = f"https://client.camb.ai/apis/tts-result/{run_id}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    result_url,
                    headers={"x-api-key": self.api_key}
                )
                if response.status_code == 200:
                    audio_data = response.content
                    audio_format = self._detect_audio_format(
                        audio_data, response.headers.get("content-type", "")
                    )
                    return audio_data, audio_format

        # Fallback: check if message contains URL
        message = getattr(status, "message", None)
        if message:
            if isinstance(message, dict):
                url = message.get("output_url") or message.get("audio_url") or message.get("url")
            elif isinstance(message, str) and message.startswith("http"):
                url = message
            else:
                url = None

            if url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    audio_data = response.content
                    audio_format = self._detect_audio_format(
                        audio_data, response.headers.get("content-type", "")
                    )
                    return audio_data, audio_format

        return b"", "pcm"

    def _detect_audio_format(self, audio_data: bytes, content_type: str) -> str:
        """Detect audio format from data bytes and content-type header.

        Returns:
            Detected format: 'wav', 'mp3', 'flac', or 'pcm' (raw).
        """
        # Check magic bytes first
        if audio_data.startswith(b"RIFF"):
            return "wav"
        if audio_data.startswith(b"\xff\xfb") or audio_data.startswith(b"\xff\xfa") or audio_data.startswith(b"ID3"):
            return "mp3"
        if audio_data.startswith(b"fLaC"):
            return "flac"
        if audio_data.startswith(b"OggS"):
            return "ogg"

        # Check content-type header as fallback
        content_type = content_type.lower()
        if "wav" in content_type or "wave" in content_type:
            return "wav"
        if "mpeg" in content_type or "mp3" in content_type:
            return "mp3"
        if "flac" in content_type:
            return "flac"
        if "ogg" in content_type:
            return "ogg"

        # Unknown format, assume raw PCM
        return "pcm"

    def _format_output(self, audio_data: bytes, output_format: str, audio_format: str = "pcm") -> str:
        """Format audio data according to output_format.

        Args:
            audio_data: Raw audio bytes.
            output_format: Desired output format ('file_path' or 'base64').
            audio_format: Detected audio format ('wav', 'mp3', 'flac', 'ogg', or 'pcm').
        """
        # Only add WAV header if it's raw PCM data
        if audio_format == "pcm" and audio_data:
            audio_data = self._add_wav_header(audio_data)
            audio_format = "wav"

        # Determine file extension
        ext_map = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg", "pcm": ".wav"}
        extension = ext_map.get(audio_format, ".wav")

        if output_format == "base64":
            return base64.b64encode(audio_data).decode("utf-8")
        else:  # file_path
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
                f.write(audio_data)
                return f.name

    def _add_wav_header(self, pcm_data: bytes) -> bytes:
        """Add WAV header to raw PCM data (16-bit, 24kHz, mono)."""
        import struct

        sample_rate = 24000
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_data)

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )

        return header + pcm_data
