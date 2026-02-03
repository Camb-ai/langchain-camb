"""Translation tool for CAMB AI."""

from __future__ import annotations

from typing import Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field

from langchain_camb.tools.base import CambBaseTool


class TranslationInput(BaseModel):
    """Input schema for Translation tool."""

    text: str = Field(
        ...,
        description="Text to translate.",
    )
    source_language: int = Field(
        ...,
        description="Source language code (integer). Common codes: 1=English, 2=Spanish, 3=French, 4=German, 5=Italian, 6=Portuguese, 7=Dutch, 8=Russian, 9=Japanese, 10=Korean, 11=Chinese.",
    )
    target_language: int = Field(
        ...,
        description="Target language code (integer). Use same language codes as source_language.",
    )
    formality: Optional[int] = Field(
        default=None,
        description="Formality level: 1=formal, 2=informal. Optional.",
    )


class CambTranslationTool(CambBaseTool):
    """Tool for translating text using CAMB AI.

    This tool provides high-quality machine translation supporting 140+ languages.
    It uses CAMB AI's streaming translation API for fast responses.

    Example:
        ```python
        from langchain_camb import CambTranslationTool

        translator = CambTranslationTool()
        result = translator.invoke({
            "text": "Hello, how are you?",
            "source_language": 1,  # English
            "target_language": 2,  # Spanish
        })
        print(result)  # "Hola, ¿cómo estás?"
        ```
    """

    name: str = "camb_translation"
    description: str = (
        "Translate text between 140+ languages using CAMB AI. "
        "Provide source and target language codes (integers) and the text to translate. "
        "Common codes: 1=English, 2=Spanish, 3=French, 4=German, 5=Italian."
    )
    args_schema: Type[BaseModel] = TranslationInput

    def _run(
        self,
        text: str,
        source_language: int,
        target_language: int,
        formality: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Translate text synchronously.

        Returns:
            Translated text string.
        """
        from camb.core.api_error import ApiError

        kwargs = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
        }

        if formality:
            kwargs["formality"] = formality

        try:
            result = self.sync_client.translation.translation_stream(**kwargs)
            return self._extract_text(result)
        except ApiError as e:
            # SDK bug: translation_stream returns plain text but SDK tries to parse as JSON
            # If status is 200, the body contains the translated text
            if e.status_code == 200 and e.body:
                return str(e.body)
            raise

    async def _arun(
        self,
        text: str,
        source_language: int,
        target_language: int,
        formality: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Translate text asynchronously.

        Returns:
            Translated text string.
        """
        from camb.core.api_error import ApiError

        kwargs = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
        }

        if formality:
            kwargs["formality"] = formality

        try:
            result = await self.async_client.translation.translation_stream(**kwargs)
            return self._extract_text(result)
        except ApiError as e:
            # SDK bug: translation_stream returns plain text but SDK tries to parse as JSON
            # If status is 200, the body contains the translated text
            if e.status_code == 200 and e.body:
                return str(e.body)
            raise

    def _extract_text(self, result) -> str:
        """Extract text from various result types."""
        # Handle streaming response - collect all chunks
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            chunks = []
            for chunk in result:
                if hasattr(chunk, "text"):
                    chunks.append(chunk.text)
                elif isinstance(chunk, str):
                    chunks.append(chunk)
            return "".join(chunks)

        # Direct result
        if hasattr(result, "text"):
            return result.text
        if isinstance(result, str):
            return result
        return str(result)
