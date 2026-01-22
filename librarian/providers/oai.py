"""
OpenAI LLM client implementation.

Provides async completions and tool calling using OpenAI's API.
Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
"""

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from librarian.providers.base import BaseLLMClient, ToolCall, ToolResponse

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 4096


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client for completions and tool calls.

    Uses async API for non-blocking operations.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        default_max_tokens: int | None = None,
    ) -> None:
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key (default: from OPENAI_API_KEY env).
            model: Model to use (default: gpt-4o-mini).
            default_max_tokens: Default max tokens for completions.
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model = model or DEFAULT_MODEL
        self._default_max_tokens = default_max_tokens or DEFAULT_MAX_TOKENS
        self._client: AsyncOpenAI | None = None  # type: ignore[no-any-unimported]

        if not self._api_key:
            logger.warning("OPENAI_API_KEY not set - OpenAI client will fail")

    @property
    def client(self) -> "AsyncOpenAI":  # type: ignore[no-any-unimported]
        """Lazily initialize the async client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError as e:
                msg = "openai package required. Install with: pip install openai"
                raise ImportError(msg) from e
        return self._client

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a text completion.

        Args:
            prompt: User message/prompt.
            system: Optional system message.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content or ""

    async def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> ToolResponse:
        """
        Generate completion with potential tool calls.

        Args:
            prompt: User message/prompt.
            tools: Tool definitions in OpenAI function format.
            system: Optional system message.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            ToolResponse with content and/or tool calls.
        """
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        message = choice.message

        # Extract tool calls if present
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return ToolResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using tiktoken if available.

        Falls back to simple heuristic if tiktoken not installed.
        """
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self._model)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: ~4 chars per token for English
            return len(text) // 4
