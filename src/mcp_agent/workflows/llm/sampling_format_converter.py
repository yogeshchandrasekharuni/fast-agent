from typing import Generic, List, Protocol, TypeVar

from mcp import CreateMessageResult, SamplingMessage

# Define type variables here instead of importing from augmented_llm
MessageParamT = TypeVar("MessageParamT")
"""A type representing an input message to an LLM."""

MessageT = TypeVar("MessageT")
"""A type representing an output message from an LLM."""


class SamplingFormatConverter(Protocol, Generic[MessageParamT, MessageT]):
    """Conversions between LLM provider and MCP types"""

    @classmethod
    def to_sampling_result(cls, result: MessageT) -> CreateMessageResult:
        """Convert an LLM response to an MCP message result type."""

    @classmethod
    def from_sampling_result(cls, result: CreateMessageResult) -> MessageT:
        """Convert an MCP message result to an LLM response type."""

    @classmethod
    def to_sampling_message(cls, param: MessageParamT) -> SamplingMessage:
        """Convert an LLM input to an MCP message (SamplingMessage) type."""

    @classmethod
    def from_sampling_message(cls, param: SamplingMessage) -> MessageParamT:
        """Convert an MCP message (SamplingMessage) to an LLM input type."""

    @classmethod
    def from_prompt_message(cls, message) -> MessageParamT:
        """Convert an MCP PromptMessage to a provider-specific message parameter."""


def typed_dict_extras(d: dict, exclude: List[str]):
    extras = {k: v for k, v in d.items() if k not in exclude}
    return extras
