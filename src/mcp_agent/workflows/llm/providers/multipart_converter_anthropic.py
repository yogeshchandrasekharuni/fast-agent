from typing import List, Union, Sequence

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolResult,
    TextResourceContents,
    BlobResourceContents,
)
from pydantic import AnyUrl
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
    ToolResultBlockParam,
    ContentBlockParam,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.resource_utils import extract_title_from_uri

_logger = get_logger("multipart_converter_anthropic")
# List of image MIME types supported by Anthropic API
SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class AnthropicConverter:
    """Converts MCP message types to Anthropic API format."""

    @staticmethod
    def _convert_content_items(
        content_items: Sequence[Union[TextContent, ImageContent, EmbeddedResource]],
        documentMode: bool = True,
    ) -> List[ContentBlockParam]:
        """
        Helper method to convert a list of content items to Anthropic format.

        Args:
            content_items: Sequence of MCP content items
            documentMode: Whether to convert text resources to document blocks (True) or text blocks (False)

        Returns:
            List of Anthropic content blocks
        """

        anthropic_blocks: List[ContentBlockParam] = []

        for content_item in content_items:
            if isinstance(content_item, TextContent):
                anthropic_block = AnthropicConverter._convert_text_content(content_item)
                anthropic_blocks.append(anthropic_block)
            elif isinstance(content_item, ImageContent):
                # Check if image MIME type is supported
                if content_item.mimeType not in SUPPORTED_IMAGE_MIME_TYPES:
                    anthropic_block = AnthropicConverter._format_fail_message(
                        content_item, content_item.mimeType
                    )
                else:
                    anthropic_block = AnthropicConverter._convert_image_content(
                        content_item
                    )
                anthropic_blocks.append(anthropic_block)
            elif isinstance(content_item, EmbeddedResource):
                anthropic_block = AnthropicConverter._convert_embedded_resource(
                    content_item, documentMode
                )
                anthropic_blocks.append(anthropic_block)

        return anthropic_blocks

    @staticmethod
    def _format_fail_message(
        resource: Union[TextContent, ImageContent, EmbeddedResource], mimetype: str
    ) -> TextBlockParam:
        """Create a fallback text block for unsupported resource types"""
        fallback_text: str = f"Unknown resource with format {mimetype}"
        if resource.type == "image":
            fallback_text = f"Image with unsupported format '{mimetype}' ({len(resource.data)} characters)"
        if isinstance(resource, EmbeddedResource):
            if isinstance(resource.resource, BlobResourceContents):
                fallback_text = f"Embedded Resource {resource.resource.uri._url} with unsupported format {resource.resource.mimeType} ({len(resource.resource.blob)} characters)"

        return TextBlockParam(type="text", text=fallback_text)

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
        role: str = multipart_msg.role

        # Convert content blocks
        anthropic_blocks: List[MessageParam] = (
            AnthropicConverter._convert_content_items(multipart_msg.content)
        )

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
    def _determine_mime_type(
        resource: TextResourceContents | BlobResourceContents,
    ) -> str:
        if resource.mimeType:
            return resource.mimeType

        if resource.uri:
            return guess_mime_type(resource.uri.serialize_url)

        if resource.blob:
            return "application/octet-stream"
        else:
            return "text/plain"

    @staticmethod
    def _convert_embedded_resource(
        resource: EmbeddedResource,
        documentMode: bool = True,
    ) -> ContentBlockParam:
        """Convert EmbeddedResource to appropriate Anthropic block type.

        Args:
            resource: The embedded resource to convert
            documentMode: Whether to convert text resources to Document blocks (True) or Text blocks (False)

        Returns:
            An appropriate ContentBlockParam for the resource
        """
        resource_content: TextResourceContents | BlobResourceContents = (
            resource.resource
        )
        uri: AnyUrl = resource_content.uri
        is_url: bool = uri.scheme in ("http", "https")
        mime_type = AnthropicConverter._determine_mime_type(resource_content)
        # Extract title from URI
        title = extract_title_from_uri(uri) if uri else None

        # Special case for SVG - it's actually text/XML, so extract as text
        if mime_type == "image/svg+xml":
            if hasattr(resource_content, "text"):
                # For SVG from text resource
                svg_content = resource_content.text
                return TextBlockParam(type="text", text=f"```xml\n{svg_content}\n```")

        # Handle image resources
        if is_image_mime_type(mime_type):
            # Check if image MIME type is supported
            if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
                return AnthropicConverter._format_fail_message(resource, mime_type)

            # Handle supported image types
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
            if documentMode:
                if hasattr(resource_content, "text"):
                    return DocumentBlockParam(
                        type="document",
                        title=title,
                        source=PlainTextSourceParam(
                            type="text",
                            media_type="text/plain",
                            data=resource_content.text,
                        ),
                    )
            # Return as text block when documentMode is False
            if hasattr(resource_content, "text"):
                return TextBlockParam(type="text", text=resource_content.text)

        # Default fallback - convert to text if possible
        if hasattr(resource_content, "text"):
            return TextBlockParam(type="text", text=resource_content.text)

        return AnthropicConverter._format_fail_message(resource, mime_type)

    @staticmethod
    def convert_tool_result_to_anthropic(
        tool_result: CallToolResult, tool_use_id: str
    ) -> ToolResultBlockParam:
        """
        Convert an MCP CallToolResult to an Anthropic ToolResultBlockParam.

        Args:
            tool_result: The tool result from a tool call
            tool_use_id: The ID of the associated tool use

        Returns:
            An Anthropic ToolResultBlockParam ready to be included in a user message
        """
        # For tool results, we always use documentMode=False to get text blocks instead of document blocks
        anthropic_content = []

        for item in tool_result.content:
            if isinstance(item, EmbeddedResource):
                # For embedded resources, always use text mode in tool results
                resource_block = AnthropicConverter._convert_embedded_resource(
                    item, documentMode=False
                )
                anthropic_content.append(resource_block)
            else:
                # For other types (Text, Image), use standard conversion
                blocks = AnthropicConverter._convert_content_items(
                    [item], documentMode=False
                )
                anthropic_content.extend(blocks)

        # If we ended up with no valid content blocks, create a placeholder
        if not anthropic_content:
            anthropic_content = [
                TextBlockParam(type="text", text="[No content in tool result]")
            ]

        # Create the tool result block
        return ToolResultBlockParam(
            type="tool_result",
            tool_use_id=tool_use_id,
            content=anthropic_content,
            is_error=tool_result.isError,
        )

    @staticmethod
    def create_tool_results_message(
        tool_results: List[tuple[str, CallToolResult]],
    ) -> MessageParam:
        """
        Create a user message containing tool results.

        Args:
            tool_results: List of (tool_use_id, tool_result) tuples

        Returns:
            A MessageParam with role='user' containing all tool results
        """
        content_blocks = []

        for tool_use_id, result in tool_results:
            # Split into text/image content vs other content
            tool_content = []
            separate_blocks = []

            for item in result.content:
                # Text and images go in tool results, other resources (PDFs) go as separate blocks
                if isinstance(item, (TextContent, ImageContent)):
                    tool_content.append(item)
                elif isinstance(item, EmbeddedResource):
                    # If it's a text resource, keep it in tool_content
                    if isinstance(item.resource, TextResourceContents):
                        tool_content.append(item)
                    else:
                        # For binary resources like PDFs, convert and add as separate block
                        block = AnthropicConverter._convert_embedded_resource(
                            item, documentMode=True
                        )
                        separate_blocks.append(block)
                else:
                    tool_content.append(item)

            # Always create a tool result block, even if empty
            # If tool_content is empty, we'll get a placeholder text block added in convert_tool_result_to_anthropic
            tool_result = CallToolResult(content=tool_content, isError=result.isError)
            content_blocks.append(
                AnthropicConverter.convert_tool_result_to_anthropic(
                    tool_result, tool_use_id
                )
            )

            # Add separate blocks directly to the message
            content_blocks.extend(separate_blocks)

        return MessageParam(role="user", content=content_blocks)
