"""Transcription tool for CAMB AI."""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field, model_validator

from langchain_camb.tools.base import CambBaseTool


class TranscriptionInput(BaseModel):
    """Input schema for Transcription tool."""

    language: int = Field(
        ...,
        description="Language code (integer) for the audio. Common codes: 1=English, 2=Spanish, 3=French, 4=German, 5=Italian.",
    )
    audio_url: Optional[str] = Field(
        default=None,
        description="URL of the audio file to transcribe. Provide either audio_url or audio_file_path.",
    )
    audio_file_path: Optional[str] = Field(
        default=None,
        description="Local file path to the audio file. Provide either audio_url or audio_file_path.",
    )

    @model_validator(mode="after")
    def validate_audio_source(self) -> "TranscriptionInput":
        """Ensure exactly one audio source is provided."""
        if not self.audio_url and not self.audio_file_path:
            raise ValueError("Either audio_url or audio_file_path must be provided.")
        if self.audio_url and self.audio_file_path:
            raise ValueError(
                "Provide only one of audio_url or audio_file_path, not both."
            )
        return self


class CambTranscriptionTool(CambBaseTool):
    """Tool for transcribing audio using CAMB AI.

    This tool converts speech to text with speaker identification.
    Supports audio URLs or local files. Returns transcription with
    segments and speaker information.

    Example:
        ```python
        from langchain_camb import CambTranscriptionTool

        transcriber = CambTranscriptionTool()
        result = transcriber.invoke({
            "audio_url": "https://example.com/audio.mp3",
            "language": 1  # English
        })
        print(result)  # JSON with text, segments, speakers
        ```
    """

    name: str = "camb_transcription"
    description: str = (
        "Transcribe audio to text using CAMB AI. "
        "Supports audio URLs or local files. "
        "Returns transcription with segments and speaker identification. "
        "Provide language code (1=English, 2=Spanish, etc.) and audio source."
    )
    args_schema: Type[BaseModel] = TranscriptionInput

    def _run(
        self,
        language: int,
        audio_url: Optional[str] = None,
        audio_file_path: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Transcribe audio synchronously.

        Returns:
            JSON string with text, segments, and speakers.
        """
        kwargs: dict[str, Any] = {"language": language}

        if audio_url:
            kwargs["audio_url"] = audio_url
        elif audio_file_path:
            with open(audio_file_path, "rb") as f:
                kwargs["media_file"] = f
                # Create task
                result = self.sync_client.transcription.create_transcription(**kwargs)
        else:
            raise ValueError("No audio source provided")

        if audio_url:
            result = self.sync_client.transcription.create_transcription(**kwargs)

        task_id = result.task_id

        # Poll for completion and get run_id from status
        status = self._poll_task_status_sync(
            self.sync_client.transcription.get_transcription_task_status,
            task_id,
        )
        run_id = status.run_id

        # Get result
        transcription = self.sync_client.transcription.get_transcription_result(run_id)

        return self._format_result(transcription)

    async def _arun(
        self,
        language: int,
        audio_url: Optional[str] = None,
        audio_file_path: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Transcribe audio asynchronously.

        Returns:
            JSON string with text, segments, and speakers.
        """
        kwargs: dict[str, Any] = {"language": language}

        if audio_url:
            kwargs["audio_url"] = audio_url
        elif audio_file_path:
            with open(audio_file_path, "rb") as f:
                kwargs["media_file"] = f
                result = await self.async_client.transcription.create_transcription(
                    **kwargs
                )
        else:
            raise ValueError("No audio source provided")

        if audio_url:
            result = await self.async_client.transcription.create_transcription(**kwargs)

        task_id = result.task_id

        # Poll for completion and get run_id from status
        status = await self._poll_task_status(
            self.async_client.transcription.get_transcription_task_status,
            task_id,
        )
        run_id = status.run_id

        # Get result
        transcription = await self.async_client.transcription.get_transcription_result(
            run_id
        )

        return self._format_result(transcription)

    def _format_result(self, transcription: Any) -> str:
        """Format transcription result as JSON."""
        result = {
            "text": getattr(transcription, "text", ""),
            "segments": [],
            "speakers": [],
        }

        # Extract segments if available
        if hasattr(transcription, "segments"):
            for seg in transcription.segments:
                result["segments"].append(
                    {
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "text": getattr(seg, "text", ""),
                        "speaker": getattr(seg, "speaker", None),
                    }
                )

        # Extract unique speakers
        if hasattr(transcription, "speakers"):
            result["speakers"] = list(transcription.speakers)
        elif result["segments"]:
            speakers = set()
            for seg in result["segments"]:
                if seg.get("speaker"):
                    speakers.add(seg["speaker"])
            result["speakers"] = list(speakers)

        return json.dumps(result, indent=2)
