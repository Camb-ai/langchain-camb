"""Text-to-Sound tool for CAMB AI."""

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


class TextToSoundInput(BaseModel):
    """Input schema for Text-to-Sound tool."""

    prompt: str = Field(
        ...,
        description="Description of the sound or music to generate.",
    )
    duration: Optional[float] = Field(
        default=None,
        description="Duration of the audio in seconds.",
    )
    audio_type: Optional[Literal["music", "sound"]] = Field(
        default=None,
        description="Type of audio: 'music' or 'sound'.",
    )
    output_format: Literal["file_path", "base64"] = Field(
        default="file_path",
        description="Output format: 'file_path' or 'base64'.",
    )


class CambTextToSoundTool(CambBaseTool):
    """Tool for generating sounds and music from text using CAMB AI.

    This tool creates audio from text descriptions. It can generate
    music, sound effects, or ambient soundscapes.

    Example:
        ```python
        from langchain_camb import CambTextToSoundTool

        sound_gen = CambTextToSoundTool()
        result = sound_gen.invoke({
            "prompt": "Upbeat electronic music with a driving beat",
            "duration": 30,
            "audio_type": "music"
        })
        print(result)  # File path to generated audio
        ```
    """

    name: str = "camb_text_to_sound"
    description: str = (
        "Generate sounds, music, or soundscapes from text descriptions using CAMB AI. "
        "Describe the audio you want and optionally specify duration and type "
        "(music, sound_effect, ambient). Returns audio file."
    )
    args_schema: Type[BaseModel] = TextToSoundInput

    def _run(
        self,
        prompt: str,
        duration: Optional[float] = None,
        audio_type: Optional[str] = None,
        output_format: str = "file_path",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate sound synchronously.

        Returns:
            File path or base64 encoded audio.
        """
        kwargs: dict[str, Any] = {"prompt": prompt}

        if duration:
            kwargs["duration"] = duration
        if audio_type:
            kwargs["audio_type"] = audio_type

        # Create task
        result = self.sync_client.text_to_audio.create_text_to_audio(**kwargs)
        task_id = result.task_id

        # Poll for completion and get run_id from status
        status = self._poll_task_status_sync(
            self.sync_client.text_to_audio.get_text_to_audio_status,
            task_id,
        )
        run_id = status.run_id

        # Get audio result (streaming)
        audio_chunks: list[bytes] = []
        for chunk in self.sync_client.text_to_audio.get_text_to_audio_result(run_id):
            audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)

        return self._format_output(audio_data, output_format)

    async def _arun(
        self,
        prompt: str,
        duration: Optional[float] = None,
        audio_type: Optional[str] = None,
        output_format: str = "file_path",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Generate sound asynchronously.

        Returns:
            File path or base64 encoded audio.
        """
        kwargs: dict[str, Any] = {"prompt": prompt}

        if duration:
            kwargs["duration"] = duration
        if audio_type:
            kwargs["audio_type"] = audio_type

        # Create task
        result = await self.async_client.text_to_audio.create_text_to_audio(**kwargs)
        task_id = result.task_id

        # Poll for completion and get run_id from status
        status = await self._poll_task_status(
            self.async_client.text_to_audio.get_text_to_audio_status,
            task_id,
        )
        run_id = status.run_id

        # Get audio result (streaming)
        audio_chunks: list[bytes] = []
        async for chunk in self.async_client.text_to_audio.get_text_to_audio_result(
            run_id
        ):
            audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)

        return self._format_output(audio_data, output_format)

    def _format_output(self, audio_data: bytes, output_format: str) -> str:
        """Format audio data according to output_format."""
        if output_format == "base64":
            return base64.b64encode(audio_data).decode("utf-8")
        else:  # file_path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                return f.name
