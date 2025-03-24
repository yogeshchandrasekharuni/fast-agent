from typing import Generic, List, Protocol, TypeVar


# Define type variables here instead of importing from augmented_llm
MessageParamT = TypeVar("MessageParamT")
"""A type representing an input message to an LLM."""

MessageT = TypeVar("MessageT")
"""A type representing an output message from an LLM."""


class SamplingFormatConverter(Protocol, Generic[MessageParamT, MessageT]):
    """Conversions between LLM provider and MCP types"""

    @classmethod
    def from_prompt_message(cls, message) -> MessageParamT:
        """Convert an MCP PromptMessage to a provider-specific message parameter."""


def typed_dict_extras(d: dict, exclude: List[str]):
    extras = {k: v for k, v in d.items() if k not in exclude}
    return extras
