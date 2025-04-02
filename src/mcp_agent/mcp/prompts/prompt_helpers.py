"""
Helper functions for working with PromptMessage and PromptMessageMultipart objects.

These utilities simplify extracting content from nested message structures
without repetitive type checking.
"""

from typing import List, Optional, Union, cast

from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


def get_text(content: Union[TextContent, ImageContent, EmbeddedResource]) -> Optional[str]:
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


def get_image_data(content: Union[TextContent, ImageContent, EmbeddedResource]) -> Optional[str]:
    """
    Extract image data from a content object if available.

    Args:
        content: A content object (TextContent, ImageContent, or EmbeddedResource)

    Returns:
        The image data as a base64 string or None if not an image content
    """
    if isinstance(content, ImageContent):
        return content.data

    if isinstance(content, EmbeddedResource):
        if isinstance(content.resource, BlobResourceContents):
            # This assumes the blob might be an image, which isn't always true
            # Consider checking the mimeType if needed
            return content.resource.blob

    return None


def get_resource_uri(content: Union[TextContent, ImageContent, EmbeddedResource]) -> Optional[str]:
    """
    Extract resource URI from an EmbeddedResource if available.

    Args:
        content: A content object (TextContent, ImageContent, or EmbeddedResource)

    Returns:
        The resource URI as a string or None if not an embedded resource
    """
    if isinstance(content, EmbeddedResource):
        return str(content.resource.uri)

    return None


def is_text_content(content: Union[TextContent, ImageContent, EmbeddedResource]) -> bool:
    """
    Check if the content is text content.

    Args:
        content: A content object (TextContent, ImageContent, or EmbeddedResource)

    Returns:
        True if the content is TextContent, False otherwise
    """
    return isinstance(content, TextContent)


def is_image_content(content: Union[TextContent, ImageContent, EmbeddedResource]) -> bool:
    """
    Check if the content is image content.

    Args:
        content: A content object (TextContent, ImageContent, or EmbeddedResource)

    Returns:
        True if the content is ImageContent, False otherwise
    """
    return isinstance(content, ImageContent)


def is_resource_content(content: Union[TextContent, ImageContent, EmbeddedResource]) -> bool:
    """
    Check if the content is an embedded resource.

    Args:
        content: A content object (TextContent, ImageContent, or EmbeddedResource)

    Returns:
        True if the content is EmbeddedResource, False otherwise
    """
    return isinstance(content, EmbeddedResource)


class MessageContent:
    """
    Helper class for working with message content in both PromptMessage and
    PromptMessageMultipart objects.
    """

    @staticmethod
    def get_all_text(message: Union[PromptMessage, PromptMessageMultipart]) -> List[str]:
        """
        Extract all text content from a message.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            List of text strings from all text content parts
        """
        if isinstance(message, PromptMessage):
            text = get_text(message.content)
            return [text] if text is not None else []

        result = []
        for content in message.content:
            text = get_text(content)
            if text is not None:
                result.append(text)

        return result

    @staticmethod
    def join_text(
        message: Union[PromptMessage, PromptMessageMultipart], separator: str = "\n\n"
    ) -> str:
        """
        Join all text content in a message with a separator.

        Args:
            message: A PromptMessage or PromptMessageMultipart
            separator: String to use as separator (default: newlines)

        Returns:
            Joined text string
        """
        return separator.join(MessageContent.get_all_text(message))

    @staticmethod
    def get_first_text(message: Union[PromptMessage, PromptMessageMultipart]) -> Optional[str]:
        """
        Get the first available text content from a message.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            First text content or None if no text content exists
        """
        if isinstance(message, PromptMessage):
            return get_text(message.content)

        for content in message.content:
            text = get_text(content)
            if text is not None:
                return text

        return None

    @staticmethod
    def has_text_at_first_position(message: Union[PromptMessage, PromptMessageMultipart]) -> bool:
        """
        Check if a message has a TextContent at the first position.
        This is a common case when dealing with messages that start with text.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            True if the message has TextContent at first position, False otherwise
        """
        if isinstance(message, PromptMessage):
            return isinstance(message.content, TextContent)

        # For multipart messages, check if there's at least one item and the first is TextContent
        return len(message.content) > 0 and isinstance(message.content[0], TextContent)

    @staticmethod
    def get_text_at_first_position(
        message: Union[PromptMessage, PromptMessageMultipart],
    ) -> Optional[str]:
        """
        Get the text from the first position of a message if it's TextContent.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            The text content at the first position if it's TextContent,
            None otherwise
        """
        if not MessageContent.has_text_at_first_position(message):
            return None

        if isinstance(message, PromptMessage):
            return cast("TextContent", message.content).text

        # Safe to cast since we've verified the first item is TextContent
        return cast("TextContent", message.content[0]).text

    @staticmethod
    def get_all_images(message: Union[PromptMessage, PromptMessageMultipart]) -> List[str]:
        """
        Extract all image data from a message.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            List of image data strings from all image content parts
        """
        if isinstance(message, PromptMessage):
            img_data = get_image_data(message.content)
            return [img_data] if img_data is not None else []

        result = []
        for content in message.content:
            img_data = get_image_data(content)
            if img_data is not None:
                result.append(img_data)

        return result

    @staticmethod
    def get_first_image(message: Union[PromptMessage, PromptMessageMultipart]) -> Optional[str]:
        """
        Get the first available image data from a message.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            First image data or None if no image content exists
        """
        if isinstance(message, PromptMessage):
            return get_image_data(message.content)

        for content in message.content:
            img_data = get_image_data(content)
            if img_data is not None:
                return img_data

        return None

    @staticmethod
    def get_all_resources(
        message: Union[PromptMessage, PromptMessageMultipart],
    ) -> List[EmbeddedResource]:
        """
        Extract all embedded resources from a message.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            List of EmbeddedResource objects
        """
        if isinstance(message, PromptMessage):
            if isinstance(message.content, EmbeddedResource):
                return [message.content]
            return []

        return [content for content in message.content if isinstance(content, EmbeddedResource)]

    @staticmethod
    def has_text(message: Union[PromptMessage, PromptMessageMultipart]) -> bool:
        """
        Check if the message has any text content.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            True if the message has text content, False otherwise
        """
        return len(MessageContent.get_all_text(message)) > 0

    @staticmethod
    def has_images(message: Union[PromptMessage, PromptMessageMultipart]) -> bool:
        """
        Check if the message has any image content.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            True if the message has image content, False otherwise
        """
        return len(MessageContent.get_all_images(message)) > 0

    @staticmethod
    def has_resources(message: Union[PromptMessage, PromptMessageMultipart]) -> bool:
        """
        Check if the message has any embedded resources.

        Args:
            message: A PromptMessage or PromptMessageMultipart

        Returns:
            True if the message has embedded resources, False otherwise
        """
        return len(MessageContent.get_all_resources(message)) > 0
