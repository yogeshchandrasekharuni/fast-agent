from typing import List, Union, Optional

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    Role,
    TextResourceContents,
    BlobResourceContents,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.mime_utils import (
    guess_mime_type,
    is_text_mime_type,
    is_image_mime_type,
)

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ImageBlockParam,
    DocumentBlockParam,
    Base64ImageSourceParam,
    URLImageSourceParam,
    Base64PDFSourceParam,
    URLPDFSourceParam,
    PlainTextSourceParam,
    ContentBlockParam,
)
from mcp_agent.logging.logger import get_logger

_logger = get_logger("mutlipart_converter_anthropic")
# List of image MIME types supported by Anthropic API
SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class AnthropicConverter:
    """Converts MCP message types to Anthropic API format."""

    @staticmethod
    def convert_to_anthropic(multipart_msg: PromptMessageMultipart) -> MessageParam:
        """
        Convert a PromptMessageMultipart message to Anthropic API format.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert

        Returns:
            An Anthropic API MessageParam object
        """
        # Extract role
        role = multipart_msg.role

        # Convert content blocks
        anthropic_blocks = []

        for content_item in multipart_msg.content:
            try:
                if isinstance(content_item, TextContent):
                    anthropic_block = AnthropicConverter._convert_text_content(
                        content_item
                    )
                    anthropic_blocks.append(anthropic_block)
                elif isinstance(content_item, ImageContent):
                    # Check if image MIME type is supported
                    if content_item.mimeType not in SUPPORTED_IMAGE_MIME_TYPES:
                        _logger.warning(
                            f"Skipping image with unsupported MIME type: {content_item.mimeType}. "
                            f"Anthropic only supports: {', '.join(SUPPORTED_IMAGE_MIME_TYPES)}"
                        )
                        continue  # Skip this content item
                    anthropic_block = AnthropicConverter._convert_image_content(
                        content_item
                    )
                    anthropic_blocks.append(anthropic_block)
                elif isinstance(content_item, EmbeddedResource):
                    try:
                        anthropic_block = AnthropicConverter._convert_embedded_resource(
                            content_item
                        )
                        anthropic_blocks.append(anthropic_block)
                    except ValueError as e:
                        _logger.warning(f"Skipping embedded resource: {e}")
                        continue  # Skip this content item
                else:
                    _logger.warning(
                        f"Skipping unsupported content type: {type(content_item)}"
                    )
                    continue  # Skip this content item
            except Exception as e:
                _logger.warning(f"Error converting content item: {e}")
                continue  # Skip this content item

        # Filter blocks based on role (assistant can only have text blocks)
        if role == "assistant":
            text_blocks = []
            for block in anthropic_blocks:
                if block.get("type") == "text":
                    text_blocks.append(block)
                else:
                    _logger.warning(
                        f"Removing non-text block from assistant message: {block.get('type')}"
                    )
            anthropic_blocks = text_blocks

        # Create the Anthropic message
        return MessageParam(role=role, content=anthropic_blocks)

    @staticmethod
    def _convert_text_content(content: TextContent) -> TextBlockParam:
        """Convert TextContent to Anthropic TextBlockParam."""
        return TextBlockParam(type="text", text=content.text)

    @staticmethod
    def _convert_image_content(content: ImageContent) -> ImageBlockParam:
        """Convert ImageContent to Anthropic ImageBlockParam."""
        # MIME type validation already done in the main convert method
        return ImageBlockParam(
            type="image",
            source=Base64ImageSourceParam(
                type="base64", media_type=content.mimeType, data=content.data
            ),
        )

    @staticmethod
    def _convert_embedded_resource(
        resource: EmbeddedResource,
    ) -> Union[ImageBlockParam, DocumentBlockParam, TextBlockParam]:
        """Convert EmbeddedResource to appropriate Anthropic block type."""
        resource_content = resource.resource
        uri = resource_content.uri

        # Use mime_utils to guess MIME type if not provided
        if resource_content.mimeType is None and uri:
            mime_type = guess_mime_type(str(uri))
            _logger.info(f"MIME type not provided, guessed {mime_type} for {uri}")
        else:
            mime_type = resource_content.mimeType or "application/octet-stream"

        is_url = str(uri).startswith(("http://", "https://"))

        # Extract title from URI
        title = extract_title_from_uri(uri) if uri else None

        # Handle image resources
        if is_image_mime_type(mime_type) and mime_type != "image/svg+xml":
            # Check if image MIME type is supported
            if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
                raise ValueError(
                    f"Unsupported image MIME type: {mime_type}. "
                    f"Anthropic only supports: {', '.join(SUPPORTED_IMAGE_MIME_TYPES)}"
                )

            if is_url:
                return ImageBlockParam(
                    type="image", source=URLImageSourceParam(type="url", url=str(uri))
                )
            elif hasattr(resource_content, "blob"):
                return ImageBlockParam(
                    type="image",
                    source=Base64ImageSourceParam(
                        type="base64", media_type=mime_type, data=resource_content.blob
                    ),
                )

        # Handle PDF resources
        elif mime_type == "application/pdf":
            if is_url:
                return DocumentBlockParam(
                    type="document",
                    title=title,
                    source=URLPDFSourceParam(type="url", url=str(uri)),
                )
            elif hasattr(resource_content, "blob"):
                return DocumentBlockParam(
                    type="document",
                    title=title,
                    source=Base64PDFSourceParam(
                        type="base64",
                        media_type="application/pdf",
                        data=resource_content.blob,
                    ),
                )

        # Handle text resources (default for all other text mime types)
        elif is_text_mime_type(mime_type):
            if hasattr(resource_content, "text"):
                return DocumentBlockParam(
                    type="document",
                    title=title,
                    source=PlainTextSourceParam(
                        type="text", media_type="text/plain", data=resource_content.text
                    ),
                )

        # Default fallback - convert to text if possible
        if hasattr(resource_content, "text"):
            return TextBlockParam(type="text", text=resource_content.text)

        raise ValueError(f"Unable to convert resource with MIME type: {mime_type}")


def extract_title_from_uri(uri: str) -> str:
    """Extract a readable title from a URI."""
    # Simple attempt to get filename from path
    uri_str = str(uri)
    try:
        from urllib.parse import urlparse

        parsed = urlparse(uri_str)

        # For HTTP(S) URLs
        if parsed.scheme in ("http", "https"):
            # Get the last part of the path
            path_parts = parsed.path.split("/")
            filename = next((p for p in reversed(path_parts) if p), "")
            return filename if filename else uri_str

        # For file URLs or other schemes
        elif parsed.path:
            import os.path

            return os.path.basename(parsed.path)

    except Exception:
        pass

    # Fallback to the full URI if parsing fails
    return uri_str


def normalize_uri(uri_or_filename: str) -> str:
    """
    Normalize a URI or filename to ensure it's a valid URI.
    Converts simple filenames to file:// URIs if needed.

    Args:
        uri_or_filename: A URI string or simple filename

    Returns:
        A properly formatted URI string
    """
    if not uri_or_filename:
        return ""

    # Check if it's already a valid URI with a scheme
    if "://" in uri_or_filename:
        return uri_or_filename

    # Handle Windows-style paths with backslashes
    normalized_path = uri_or_filename.replace("\\", "/")

    # If it's a simple filename or relative path, convert to file:// URI
    # Make sure it has three slashes for an absolute path
    if normalized_path.startswith("/"):
        return f"file://{normalized_path}"
    else:
        return f"file:///{normalized_path}"
