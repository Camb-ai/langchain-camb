"""Standard integration tests for CAMB AI tools using LangChain test framework.

These tests verify that the tools work correctly with the real CAMB AI API.
Run with: pytest tests/integration_tests/ -m integration

Requires CAMB_API_KEY environment variable to be set.
"""

import os
from typing import Type

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_camb import (
    CambTranslationTool,
    CambTTSTool,
    CambVoiceListTool,
)

# Skip all tests if no API key
pytestmark = pytest.mark.integration


def get_api_key() -> str:
    """Get API key from environment or skip test."""
    key = os.environ.get("CAMB_API_KEY")
    if not key:
        pytest.skip("CAMB_API_KEY not set")
    return key


class TestCambTTSToolIntegration(ToolsIntegrationTests):
    """Standard integration tests for CambTTSTool."""

    @property
    def tool_constructor(self) -> Type[CambTTSTool]:
        return CambTTSTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": get_api_key()}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "text": "Hello, this is a test.",
            "language": "en-us",
            "voice_id": 147320,
            "output_format": "base64",
        }


class TestCambVoiceListToolIntegration(ToolsIntegrationTests):
    """Standard integration tests for CambVoiceListTool."""

    @property
    def tool_constructor(self) -> Type[CambVoiceListTool]:
        return CambVoiceListTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": get_api_key()}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {}


class TestCambTranslationToolIntegration(ToolsIntegrationTests):
    """Standard integration tests for CambTranslationTool."""

    @property
    def tool_constructor(self) -> Type[CambTranslationTool]:
        return CambTranslationTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": get_api_key()}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "text": "Hello, how are you?",
            "source_language": 1,
            "target_language": 2,
        }
