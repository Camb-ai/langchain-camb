"""CAMB AI toolkit that bundles all CAMB tools."""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_camb.tools import (
    CambAudioSeparationTool,
    CambTextToSoundTool,
    CambTranscriptionTool,
    CambTranslatedTTSTool,
    CambTranslationTool,
    CambTTSTool,
    CambVoiceCloneTool,
    CambVoiceListTool,
)


class CambToolkit(BaseModel):
    """Toolkit that bundles all CAMB AI tools.

    Provides convenient access to all CAMB AI services:
    - Text-to-Speech (TTS)
    - Translated TTS
    - Translation
    - Transcription
    - Voice Listing
    - Voice Cloning
    - Text-to-Sound
    - Audio Separation

    Example:
        ```python
        from langchain_camb import CambToolkit
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        toolkit = CambToolkit()
        tools = toolkit.get_tools()

        agent = create_react_agent(ChatOpenAI(), tools)
        agent.invoke({
            "messages": [{"role": "user", "content": "Say hello in Spanish"}]
        })
        ```
    """

    api_key: Optional[str] = Field(
        default=None,
        description="CAMB AI API key. Falls back to CAMB_API_KEY environment variable.",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Optional custom base URL for CAMB AI API.",
    )
    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds.",
    )
    include_tts: bool = Field(
        default=True,
        description="Include TTS tool.",
    )
    include_translated_tts: bool = Field(
        default=True,
        description="Include Translated TTS tool.",
    )
    include_translation: bool = Field(
        default=True,
        description="Include Translation tool.",
    )
    include_transcription: bool = Field(
        default=True,
        description="Include Transcription tool.",
    )
    include_voice_list: bool = Field(
        default=True,
        description="Include Voice List tool.",
    )
    include_voice_clone: bool = Field(
        default=True,
        description="Include Voice Clone tool.",
    )
    include_text_to_sound: bool = Field(
        default=True,
        description="Include Text-to-Sound tool.",
    )
    include_audio_separation: bool = Field(
        default=True,
        description="Include Audio Separation tool.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_api_key(self) -> str:
        """Get API key from field or environment."""
        key = self.api_key or os.environ.get("CAMB_API_KEY")
        if not key:
            raise ValueError(
                "CAMB AI API key is required. "
                "Set it via 'api_key' parameter or CAMB_API_KEY environment variable."
            )
        return key

    def get_tools(self) -> List[BaseTool]:
        """Get all enabled CAMB AI tools.

        Returns:
            List of LangChain tools configured with the toolkit's settings.
        """
        api_key = self._get_api_key()
        common_kwargs = {
            "api_key": api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }

        tools: List[BaseTool] = []

        if self.include_tts:
            tools.append(CambTTSTool(**common_kwargs))

        if self.include_translated_tts:
            tools.append(CambTranslatedTTSTool(**common_kwargs))

        if self.include_translation:
            tools.append(CambTranslationTool(**common_kwargs))

        if self.include_transcription:
            tools.append(CambTranscriptionTool(**common_kwargs))

        if self.include_voice_list:
            tools.append(CambVoiceListTool(**common_kwargs))

        if self.include_voice_clone:
            tools.append(CambVoiceCloneTool(**common_kwargs))

        if self.include_text_to_sound:
            tools.append(CambTextToSoundTool(**common_kwargs))

        if self.include_audio_separation:
            tools.append(CambAudioSeparationTool(**common_kwargs))

        return tools
