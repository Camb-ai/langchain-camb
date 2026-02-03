"""Voice clone tool for CAMB AI."""

from __future__ import annotations

import json
from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field

from langchain_camb.tools.base import CambBaseTool


class VoiceCloneInput(BaseModel):
    """Input schema for Voice Clone tool."""

    voice_name: str = Field(
        ...,
        description="Name for the new cloned voice.",
    )
    audio_file_path: str = Field(
        ...,
        description="Path to audio file (2+ seconds) to clone voice from.",
    )
    gender: int = Field(
        ...,
        description="Gender: 1=Male, 2=Female, 0=Not Specified, 9=Not Applicable.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the voice.",
    )
    age: Optional[int] = Field(
        default=None,
        description="Optional age of the voice.",
    )
    language: Optional[int] = Field(
        default=None,
        description="Optional language code for the voice.",
    )


class CambVoiceCloneTool(CambBaseTool):
    """Tool for cloning voices using CAMB AI.

    This tool creates a new voice from a 2+ second audio sample.
    The cloned voice can then be used with TTS tools.

    Example:
        ```python
        from langchain_camb import CambVoiceCloneTool

        clone = CambVoiceCloneTool()
        result = clone.invoke({
            "voice_name": "My Custom Voice",
            "audio_file_path": "/path/to/audio.wav",
            "gender": 2  # Female
        })
        print(result)  # JSON with new voice_id
        ```
    """

    name: str = "camb_voice_clone"
    description: str = (
        "Clone a voice from an audio sample using CAMB AI. "
        "Requires 2+ seconds of audio. "
        "Returns the new voice ID that can be used with TTS tools. "
        "Gender: 1=Male, 2=Female, 0=Not Specified."
    )
    args_schema: Type[BaseModel] = VoiceCloneInput

    def _run(
        self,
        voice_name: str,
        audio_file_path: str,
        gender: int,
        description: Optional[str] = None,
        age: Optional[int] = None,
        language: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Clone a voice synchronously.

        Returns:
            JSON string with new voice_id and details.
        """
        with open(audio_file_path, "rb") as f:
            kwargs = {
                "voice_name": voice_name,
                "gender": gender,
                "file": f,
            }

            if description:
                kwargs["description"] = description
            if age:
                kwargs["age"] = age
            if language:
                kwargs["language"] = language

            result = self.sync_client.voice_cloning.create_custom_voice(**kwargs)

        return self._format_result(result, voice_name)

    async def _arun(
        self,
        voice_name: str,
        audio_file_path: str,
        gender: int,
        description: Optional[str] = None,
        age: Optional[int] = None,
        language: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Clone a voice asynchronously.

        Returns:
            JSON string with new voice_id and details.
        """
        with open(audio_file_path, "rb") as f:
            kwargs = {
                "voice_name": voice_name,
                "gender": gender,
                "file": f,
            }

            if description:
                kwargs["description"] = description
            if age:
                kwargs["age"] = age
            if language:
                kwargs["language"] = language

            result = await self.async_client.voice_cloning.create_custom_voice(**kwargs)

        return self._format_result(result, voice_name)

    def _format_result(self, result, voice_name: str) -> str:
        """Format the voice clone result as JSON."""
        output = {
            "voice_id": getattr(result, "voice_id", getattr(result, "id", None)),
            "voice_name": voice_name,
            "status": "created",
        }

        if hasattr(result, "message"):
            output["message"] = result.message

        return json.dumps(output, indent=2)
