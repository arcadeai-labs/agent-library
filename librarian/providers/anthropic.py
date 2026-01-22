"""
Anthropic LLM client implementation.

Provides async completions and tool calling using Anthropic's Claude API.
"""

import logging
import os
from typing import TYPE_CHECKING, Any

from librarian.providers.base import BaseLLMClient, ToolCall, ToolResponse

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 4096


class AnthropicClient(BaseLLMClient):
    """
    Anthropic API client for completions and tool calls.

    Uses async API for non-blocking operations.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        default_max_tokens: int | None = None,
    ) -> None:
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env).
            model: Model to use (default: claude-3-5-sonnet).
            default_max_tokens: Default max tokens for completions.
        """
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._model = model or DEFAULT_MODEL
        self._default_max_tokens = default_max_tokens or DEFAULT_MAX_TOKENS
        self._client: AsyncAnthropic | None = None  # type: ignore[no-any-unimported]

        if not self._api_key:
            logger.warning("ANTHROPIC_API_KEY not set - Anthropic client will fail")

    @property
    def client(self) -> "AsyncAnthropic":  # type: ignore[no-any-unimported]
        """Lazily initialize the async client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self._api_key)
            except ImportError as e:
                msg = "anthropic package required. Install with: pip install anthropic"
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
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._default_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system:
            kwargs["system"] = system
        if temperature != 1.0:  # Anthropic default is 1.0
            kwargs["temperature"] = temperature

        response = await self.client.messages.create(**kwargs)

        # Extract text from content blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return "".join(text_parts)

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
            tools: Tool definitions in Anthropic format.
            system: Optional system message.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            ToolResponse with content and/or tool calls.
        """
        # Convert to Anthropic tool format if needed
        anthropic_tools = self._convert_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._default_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": anthropic_tools,
        }

        if system:
            kwargs["system"] = system
        if temperature != 1.0:
            kwargs["temperature"] = temperature

        response = await self.client.messages.create(**kwargs)

        # Extract content and tool calls
        content_parts = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if hasattr(block, "text"):
                content_parts.append(block.text)
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return ToolResponse(
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert OpenAI-format tools to Anthropic format if needed.

        Anthropic uses a slightly different tool schema.
        """
        converted = []
        for tool in tools:
            # Check if already in Anthropic format
            if "input_schema" in tool:
                converted.append(tool)
            # Convert from OpenAI function format
            elif "function" in tool:
                func = tool["function"]
                converted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Assume it's a direct tool definition
                converted.append(tool)

        return converted

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Anthropic models.

        Uses a heuristic based on Claude's tokenization (~3.5 chars per token).
        """
        # Claude tends to use slightly fewer tokens than GPT
        return int(len(text) / 3.5)
