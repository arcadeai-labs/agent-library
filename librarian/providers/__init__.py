"""
LLM provider clients for librarian.

Provides unified interfaces for LLM completions and tool calls
across different providers (OpenAI, Anthropic, etc.).

Usage:
    from librarian.providers import OpenAIClient, AnthropicClient

    client = OpenAIClient()
    response = await client.complete("Hello, world!")
"""

from librarian.providers.base import BaseLLMClient, ToolCall, ToolResponse
from librarian.providers.oai import OpenAIClient

# Anthropic client imported conditionally (requires anthropic package)
try:
    from librarian.providers.anthropic import AnthropicClient
except ImportError:
    AnthropicClient = None  # type: ignore[misc, assignment]

__all__ = [
    "AnthropicClient",
    "BaseLLMClient",
    "OpenAIClient",
    "ToolCall",
    "ToolResponse",
]
