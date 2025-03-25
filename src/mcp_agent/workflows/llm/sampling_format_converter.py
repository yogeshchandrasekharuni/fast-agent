from typing import Generic, Protocol, TypeVar

from mcp.types import PromptMessage

# Define covariant type variables
MessageParamT_co = TypeVar("MessageParamT_co", covariant=True)
MessageT_co = TypeVar("MessageT_co", covariant=True)


class ProviderFormatConverter(Protocol, Generic[MessageParamT_co, MessageT_co]):
    """Conversions between LLM provider and MCP types"""

    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> MessageParamT_co:
        """Convert an MCP PromptMessage to a provider-specific message parameter."""
        ...


class BasicFormatConverter(ProviderFormatConverter[PromptMessage, PromptMessage]):
    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> PromptMessage:
        return message
