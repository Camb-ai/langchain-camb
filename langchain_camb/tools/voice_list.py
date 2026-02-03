"""Voice list tool for CAMB AI."""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel

from langchain_camb.tools.base import CambBaseTool


class VoiceListInput(BaseModel):
    """Input schema for Voice List tool (no parameters required)."""

    pass


class CambVoiceListTool(CambBaseTool):
    """Tool for listing available voices from CAMB AI.

    This tool retrieves all available voices that can be used with TTS tools.
    Returns voice ID, name, gender, age, and language information.

    Example:
        ```python
        from langchain_camb import CambVoiceListTool

        voice_list = CambVoiceListTool()
        voices = voice_list.invoke({})
        print(voices)  # JSON list of available voices
        ```
    """

    name: str = "camb_voice_list"
    description: str = (
        "List all available voices from CAMB AI. "
        "Returns voice IDs, names, genders, ages, and languages. "
        "Use this to find the right voice_id for TTS tools."
    )
    args_schema: Type[BaseModel] = VoiceListInput

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get list of available voices synchronously.

        Returns:
            JSON string containing list of voices with id, name, gender, age, language.
        """
        voices = self.sync_client.voice_cloning.list_voices()
        return self._format_voices(voices)

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Get list of available voices asynchronously.

        Returns:
            JSON string containing list of voices with id, name, gender, age, language.
        """
        voices = await self.async_client.voice_cloning.list_voices()
        return self._format_voices(voices)

    def _format_voices(self, voices: list[Any]) -> str:
        """Format voice list as JSON."""
        voice_list = []
        for voice in voices:
            # Handle both dict and object responses
            if isinstance(voice, dict):
                voice_list.append(
                    {
                        "id": voice.get("id"),
                        "name": voice.get("voice_name", voice.get("name", "Unknown")),
                        "gender": self._gender_to_string(voice.get("gender", 0)),
                        "age": voice.get("age"),
                        "language": voice.get("language"),
                    }
                )
            else:
                voice_list.append(
                    {
                        "id": getattr(voice, "id", None),
                        "name": getattr(voice, "voice_name", getattr(voice, "name", "Unknown")),
                        "gender": self._gender_to_string(getattr(voice, "gender", 0)),
                        "age": getattr(voice, "age", None),
                        "language": getattr(voice, "language", None),
                    }
                )

        return json.dumps(voice_list, indent=2)

    @staticmethod
    def _gender_to_string(gender: int) -> str:
        """Convert gender integer to string."""
        gender_map = {
            0: "not_specified",
            1: "male",
            2: "female",
            9: "not_applicable",
        }
        return gender_map.get(gender, "unknown")
