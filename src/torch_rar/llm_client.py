"""LLM client wrapper using LiteLLM for unified API access."""

import asyncio
import json
import logging
from typing import Any, Optional

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from torch_rar.config import LLMProvider, Settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting OpenRouter, vLLM, and LiteLLM proxy."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the LLM client.

        Args:
            settings: Configuration settings. If None, loads from environment.
        """
        self.settings = settings or Settings()
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        """Configure LiteLLM based on settings."""
        # Set API key for OpenRouter
        if self.settings.llm_provider == LLMProvider.OPENROUTER:
            if self.settings.openrouter_api_key:
                litellm.api_key = self.settings.openrouter_api_key

        # Enable verbose logging for debugging
        litellm.set_verbose = False

        # Configure drop params to handle provider-specific parameters
        litellm.drop_params = True

    def _get_completion_kwargs(self, model_type: str = "judge") -> dict[str, Any]:
        """Get kwargs for LiteLLM completion based on provider."""
        kwargs: dict[str, Any] = {
            "model": self.settings.get_litellm_model_name(model_type),
            "timeout": self.settings.request_timeout,
        }

        api_base = self.settings.get_api_base()
        if api_base:
            kwargs["api_base"] = api_base

        api_key = self.settings.get_api_key()
        if api_key:
            kwargs["api_key"] = api_key

        return kwargs

    @retry(
        retry=retry_if_exception_type((litellm.exceptions.RateLimitError, TimeoutError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
    )
    async def complete(
        self,
        messages: list[dict[str, str]],
        model_type: str = "judge",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model_type: Either 'judge' or 'rubric' to select the model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            response_format: Optional response format specification.

        Returns:
            Generated text response.

        Raises:
            Exception: If the API call fails after retries.
        """
        kwargs = self._get_completion_kwargs(model_type)
        kwargs.update(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        model_type: str = "judge",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate a JSON completion from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model_type: Either 'judge' or 'rubric' to select the model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            Parsed JSON response as a dictionary.
        """
        response = await self.complete(
            messages=messages,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())

    async def complete_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        model_type: str = "judge",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> list[str]:
        """Generate completions for multiple message sets concurrently.

        Args:
            messages_list: List of message lists.
            model_type: Either 'judge' or 'rubric' to select the model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            List of generated responses.
        """
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)

        async def limited_complete(messages: list[dict[str, str]]) -> str:
            async with semaphore:
                return await self.complete(
                    messages=messages,
                    model_type=model_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        tasks = [limited_complete(messages) for messages in messages_list]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def complete_sync(
        self,
        messages: list[dict[str, str]],
        model_type: str = "judge",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Synchronous wrapper for complete().

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model_type: Either 'judge' or 'rubric' to select the model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            Generated text response.
        """
        return asyncio.run(
            self.complete(
                messages=messages,
                model_type=model_type,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
