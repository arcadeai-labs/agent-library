"""
Base LLM client interface.

Defines the abstract interface for LLM providers, enabling
swappable backends for completions and tool calling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a tool call request from an LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResponse:
    """Response from a completion with tool calls."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Provides a unified interface for text completions and tool calling
    across different providers.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @abstractmethod
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
            prompt: The user prompt/message.
            system: Optional system message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0-2).

        Returns:
            Generated text response.
        """
        ...

    @abstractmethod
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
            prompt: The user prompt/message.
            tools: List of tool definitions in provider's format.
            system: Optional system message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            ToolResponse with content and/or tool calls.
        """
        ...

    async def summarize(
        self,
        text: str,
        max_length: int | None = None,
    ) -> str:
        """
        Summarize text content.

        Default implementation uses complete() with a summarization prompt.
        Override for provider-specific optimizations.

        Args:
            text: Text to summarize.
            max_length: Optional max length hint.

        Returns:
            Summarized text.
        """
        length_hint = f" in {max_length} words or less" if max_length else ""
        prompt = f"Summarize the following text{length_hint}:\n\n{text}"
        return await self.complete(prompt, max_tokens=max_length)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Default implementation uses a simple heuristic (4 chars per token).
        Override for provider-specific tokenizers.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        return len(text) // 4
