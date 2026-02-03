"""Base class for CAMB AI LangChain tools."""

from __future__ import annotations

import asyncio
import os
from abc import ABC
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field, model_validator

from camb.client import AsyncCambAI, CambAI


class CambBaseTool(BaseTool, ABC):
    """Base class for CAMB AI tools.

    Provides shared client management and configuration for all CAMB AI tools.
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
    max_poll_attempts: int = Field(
        default=60,
        description="Maximum number of polling attempts for async tasks.",
    )
    poll_interval: float = Field(
        default=2.0,
        description="Interval between polling attempts in seconds.",
    )

    # Private attributes for lazy client initialization
    _sync_client: Optional[CambAI] = None
    _async_client: Optional[AsyncCambAI] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_api_key(self) -> "CambBaseTool":
        """Validate that API key is available."""
        if not self.api_key:
            self.api_key = os.environ.get("CAMB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "CAMB AI API key is required. "
                "Set it via 'api_key' parameter or CAMB_API_KEY environment variable."
            )
        return self

    @property
    def sync_client(self) -> CambAI:
        """Get or create synchronous CAMB AI client."""
        if self._sync_client is None:
            self._sync_client = CambAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncCambAI:
        """Get or create asynchronous CAMB AI client."""
        if self._async_client is None:
            self._async_client = AsyncCambAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._async_client

    async def _poll_task_status(
        self,
        get_status_fn: Any,
        task_id: str,
        *,
        run_id: Optional[int] = None,
    ) -> Any:
        """Poll for async task completion.

        Args:
            get_status_fn: Async function to check task status.
            task_id: The task ID to poll.
            run_id: Optional run ID for the request.

        Returns:
            The final status result when task completes.

        Raises:
            TimeoutError: If polling exceeds max attempts.
            RuntimeError: If task fails.
        """
        for attempt in range(self.max_poll_attempts):
            status = await get_status_fn(task_id, run_id=run_id)

            if hasattr(status, "status"):
                status_value = status.status
                if status_value in ("completed", "SUCCESS"):
                    return status
                elif status_value in ("failed", "FAILED", "error"):
                    error_msg = getattr(status, "error", "Unknown error")
                    raise RuntimeError(f"Task failed: {error_msg}")

            await asyncio.sleep(self.poll_interval)

        raise TimeoutError(
            f"Task {task_id} did not complete within "
            f"{self.max_poll_attempts * self.poll_interval} seconds"
        )

    def _poll_task_status_sync(
        self,
        get_status_fn: Any,
        task_id: str,
        *,
        run_id: Optional[int] = None,
    ) -> Any:
        """Poll for async task completion (synchronous version).

        Args:
            get_status_fn: Sync function to check task status.
            task_id: The task ID to poll.
            run_id: Optional run ID for the request.

        Returns:
            The final status result when task completes.

        Raises:
            TimeoutError: If polling exceeds max attempts.
            RuntimeError: If task fails.
        """
        import time

        for attempt in range(self.max_poll_attempts):
            status = get_status_fn(task_id, run_id=run_id)

            if hasattr(status, "status"):
                status_value = status.status
                if status_value in ("completed", "SUCCESS"):
                    return status
                elif status_value in ("failed", "FAILED", "error"):
                    error_msg = getattr(status, "error", "Unknown error")
                    raise RuntimeError(f"Task failed: {error_msg}")

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Task {task_id} did not complete within "
            f"{self.max_poll_attempts * self.poll_interval} seconds"
        )
