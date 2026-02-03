"""Audio separation tool for CAMB AI."""

from __future__ import annotations

import json
import tempfile
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field, model_validator

from langchain_camb.tools.base import CambBaseTool


class AudioSeparationInput(BaseModel):
    """Input schema for Audio Separation tool."""

    audio_url: Optional[str] = Field(
        default=None,
        description="URL of the audio file to separate. Provide either audio_url or audio_file_path.",
    )
    audio_file_path: Optional[str] = Field(
        default=None,
        description="Local file path to the audio. Provide either audio_url or audio_file_path.",
    )

    @model_validator(mode="after")
    def validate_audio_source(self) -> "AudioSeparationInput":
        """Ensure exactly one audio source is provided."""
        if not self.audio_url and not self.audio_file_path:
            raise ValueError("Either audio_url or audio_file_path must be provided.")
        if self.audio_url and self.audio_file_path:
            raise ValueError(
                "Provide only one of audio_url or audio_file_path, not both."
            )
        return self


class CambAudioSeparationTool(CambBaseTool):
    """Tool for separating vocals from background audio using CAMB AI.

    This tool isolates speech/vocals from background music or noise.
    Returns paths to separated audio files.

    Example:
        ```python
        from langchain_camb import CambAudioSeparationTool

        separator = CambAudioSeparationTool()
        result = separator.invoke({
            "audio_file_path": "/path/to/mixed_audio.mp3"
        })
        print(result)  # JSON with paths to vocals and background
        ```
    """

    name: str = "camb_audio_separation"
    description: str = (
        "Separate vocals/speech from background audio using CAMB AI. "
        "Provide an audio URL or file path. "
        "Returns separate files for vocals and background audio."
    )
    args_schema: Type[BaseModel] = AudioSeparationInput

    def _run(
        self,
        audio_url: Optional[str] = None,
        audio_file_path: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Separate audio synchronously.

        Returns:
            JSON string with paths to vocals and background audio.
        """
        kwargs: dict[str, Any] = {}

        if audio_file_path:
            with open(audio_file_path, "rb") as f:
                kwargs["media_file"] = f
                result = self.sync_client.audio_separation.create_audio_separation(
                    **kwargs
                )
        else:
            # For URL, we may need to handle differently depending on SDK
            result = self.sync_client.audio_separation.create_audio_separation(**kwargs)

        task_id = result.task_id

        # Poll for completion and get run_id from status
        status = self._poll_task_status_sync(
            self.sync_client.audio_separation.get_audio_separation_status,
            task_id,
        )
        run_id = status.run_id

        # Get result
        separation_result = self.sync_client.audio_separation.get_audio_separation_run_info(
            run_id
        )

        return self._format_result(separation_result)

    async def _arun(
        self,
        audio_url: Optional[str] = None,
        audio_file_path: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Separate audio asynchronously.

        Returns:
            JSON string with paths to vocals and background audio.
        """
        kwargs: dict[str, Any] = {}

        if audio_file_path:
            with open(audio_file_path, "rb") as f:
                kwargs["media_file"] = f
                result = await self.async_client.audio_separation.create_audio_separation(
                    **kwargs
                )
        else:
            result = await self.async_client.audio_separation.create_audio_separation(
                **kwargs
            )

        task_id = result.task_id

        # Poll for completion and get run_id from status
        status = await self._poll_task_status(
            self.async_client.audio_separation.get_audio_separation_status,
            task_id,
        )
        run_id = status.run_id

        # Get result
        separation_result = (
            await self.async_client.audio_separation.get_audio_separation_run_info(
                run_id
            )
        )

        return self._format_result(separation_result)

    def _format_result(self, result: Any) -> str:
        """Format separation result as JSON."""
        output = {
            "vocals": None,
            "background": None,
            "status": "completed",
        }

        # Extract URLs or data from result
        if hasattr(result, "vocals_url"):
            output["vocals"] = result.vocals_url
        elif hasattr(result, "vocals"):
            # If it's bytes, save to temp file
            if isinstance(result.vocals, bytes):
                with tempfile.NamedTemporaryFile(
                    suffix="_vocals.wav", delete=False
                ) as f:
                    f.write(result.vocals)
                    output["vocals"] = f.name
            else:
                output["vocals"] = result.vocals

        if hasattr(result, "background_url"):
            output["background"] = result.background_url
        elif hasattr(result, "background"):
            if isinstance(result.background, bytes):
                with tempfile.NamedTemporaryFile(
                    suffix="_background.wav", delete=False
                ) as f:
                    f.write(result.background)
                    output["background"] = f.name
            else:
                output["background"] = result.background

        # Handle alternative attribute names
        if hasattr(result, "instrumental_url"):
            output["background"] = result.instrumental_url
        if hasattr(result, "voice_url"):
            output["vocals"] = result.voice_url

        return json.dumps(output, indent=2)
