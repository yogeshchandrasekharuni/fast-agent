"""
Prompt class for easily creating and working with MCP prompt content.
"""

from typing import List, Literal

from mcp.types import PromptMessage
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Import our content helper functions
from .mcp_content import User, Assistant, MCPPrompt


class Prompt:
    """
    A helper class for working with MCP prompt content.

    This class provides static methods to create:
    - PromptMessage instances
    - PromptMessageMultipart instances
    - Lists of messages for conversations

    All methods intelligently handle various content types:
    - Strings become TextContent
    - Image file paths become ImageContent
    - Other file paths become EmbeddedResource
    - Pre-formatted messages pass through unchanged
    """

    @classmethod
    def user(cls, *content_items) -> PromptMessageMultipart:
        """
        Create a user PromptMessageMultipart with various content items.

        Args:
            *content_items: Content items (strings, file paths, etc.)

        Returns:
            A PromptMessageMultipart with user role and the specified content
        """
        messages = User(*content_items)
        return PromptMessageMultipart(
            role="user", content=[msg["content"] for msg in messages]
        )

    @classmethod
    def assistant(cls, *content_items) -> PromptMessageMultipart:
        """
        Create an assistant PromptMessageMultipart with various content items.

        Args:
            *content_items: Content items (strings, file paths, etc.)

        Returns:
            A PromptMessageMultipart with assistant role and the specified content
        """
        messages = Assistant(*content_items)
        return PromptMessageMultipart(
            role="assistant", content=[msg["content"] for msg in messages]
        )

    @classmethod
    def message(
        cls, *content_items, role: Literal["user", "assistant"] = "user"
    ) -> PromptMessageMultipart:
        """
        Create a PromptMessageMultipart with the specified role and content items.

        Args:
            *content_items: Content items (strings, file paths, etc.)
            role: Role for the message (user or assistant)

        Returns:
            A PromptMessageMultipart with the specified role and content
        """
        messages = MCPPrompt(*content_items, role=role)
        return PromptMessageMultipart(
            role=messages[0]["role"] if messages else role,
            content=[msg["content"] for msg in messages],
        )

    @classmethod
    def conversation(cls, *messages) -> List[PromptMessage]:
        """
        Create a list of PromptMessages from various inputs.

        This method accepts:
        - PromptMessageMultipart instances
        - Dictionaries with role and content
        - Lists of dictionaries with role and content

        Args:
            *messages: Messages to include in the conversation

        Returns:
            A list of PromptMessage objects for the conversation
        """
        result = []

        for item in messages:
            if isinstance(item, PromptMessageMultipart):
                # Convert PromptMessageMultipart to a list of PromptMessages
                result.extend(item.to_prompt_messages())
            elif isinstance(item, dict) and "role" in item and "content" in item:
                # Convert a single message dict to PromptMessage
                result.append(PromptMessage(**item))
            elif isinstance(item, list):
                # Process each item in the list
                for msg in item:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        result.append(PromptMessage(**msg))
            # Ignore other types

        return result

    @classmethod
    def from_multipart(
        cls, multipart: List[PromptMessageMultipart]
    ) -> List[PromptMessage]:
        """
        Convert a list of PromptMessageMultipart objects to PromptMessages.

        Args:
            multipart: List of PromptMessageMultipart objects

        Returns:
            A flat list of PromptMessage objects
        """
        result = []
        for mp in multipart:
            result.extend(mp.to_prompt_messages())
        return result
