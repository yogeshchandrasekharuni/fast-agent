from typing import List, Union

from mcp.types import (
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    PromptMessage,
    Role,
    TextContent,
    TextResourceContents,
)
from pydantic import BaseModel


def get_text(content: Union[TextContent, ImageContent, EmbeddedResource]) -> str | None:
    """
    Extract text content from a content object if available.

    Args:
        content: A content object (TextContent, ImageContent, or EmbeddedResource)

    Returns:
        The text content as a string or None if not a text content
    """
    if isinstance(content, TextContent):
        return content.text

    if isinstance(content, EmbeddedResource):
        if isinstance(content.resource, TextResourceContents):
            return content.resource.text

    return None


class PromptMessageMultipart(BaseModel):
    """
    Extension of PromptMessage that handles multiple content parts.
    Internally converts to/from a sequence of standard PromptMessages.
    """

    role: Role
    content: List[Union[TextContent, ImageContent, EmbeddedResource]]

    @classmethod
    def to_multipart(cls, messages: List[PromptMessage]) -> List["PromptMessageMultipart"]:
        """Convert a sequence of PromptMessages into PromptMessageMultipart objects."""
        if not messages:
            return []

        result = []
        current_group = None
        current_role = None

        for msg in messages:
            if msg.role != current_role:
                # Role changed, start new message
                if current_group is not None:
                    result.append(current_group)
                current_role = msg.role
                current_group = cls(role=msg.role, content=[msg.content])
            else:
                # Same role, add to current message
                if current_group is not None:
                    current_group.content.append(msg.content)

        # Add the last group
        if current_group is not None:
            result.append(current_group)

        return result

    def from_multipart(self) -> List[PromptMessage]:
        """Convert this PromptMessageMultipart to a sequence of standard PromptMessages."""
        return [
            PromptMessage(role=self.role, content=content_part) for content_part in self.content
        ]

    def first_text(self) -> str:
        """
        Get the first available text content from a message.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            First text content or None if no text content exists
        """
        for content in self.content:
            text = get_text(content)
            if text is not None:
                return text

        return "<no text>"

    def all_text(self) -> str:
        """
        Get all the text available.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            First text content or None if no text content exists
        """
        result = []
        for content in self.content:
            text = get_text(content)
            if text is not None:
                result.append(text)

        return "\n".join(result)

    @classmethod
    def parse_get_prompt_result(cls, result: GetPromptResult) -> List["PromptMessageMultipart"]:
        """Parse a GetPromptResult into PromptMessageMultipart objects."""
        return cls.to_multipart(result.messages)
