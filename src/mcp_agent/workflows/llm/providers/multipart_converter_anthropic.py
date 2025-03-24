from typing import List, Union, Sequence

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolResult,
    TextResourceContents,
    BlobResourceContents,
    PromptMessage,
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
    def _is_supported_image_type(mime_type: str) -> bool:
        """Check if the given MIME type is supported by Anthropic's image API.
        
        Args:
            mime_type: The MIME type to check
            
        Returns:
            True if the MIME type is supported, False otherwise
        """
        return mime_type in SUPPORTED_IMAGE_MIME_TYPES
        
    @staticmethod
    def _convert_content_items(
        content_items: Sequence[Union[TextContent, ImageContent, EmbeddedResource]],
        document_mode: bool = True,
    ) -> List[ContentBlockParam]:
        """
        Helper method to convert a list of content items to Anthropic format.

        Args:
            content_items: Sequence of MCP content items
            document_mode: Whether to convert text resources to document blocks (True) or text blocks (False)

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
                if not AnthropicConverter._is_supported_image_type(content_item.mimeType):
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
                    content_item, document_mode
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
            AnthropicConverter._convert_content_items(multipart_msg.content, document_mode=True)
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
    def convert_prompt_message_to_anthropic(message: PromptMessage) -> MessageParam:
        """
        Convert a standard PromptMessage to Anthropic API format.
        
        Args:
            message: The PromptMessage to convert
            
        Returns:
            An Anthropic API MessageParam object
        """
        # Convert the PromptMessage to a PromptMessageMultipart containing a single content item
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])
        
        # Use the existing conversion method
        return AnthropicConverter.convert_to_anthropic(multipart)

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
    def _convert_svg_resource(resource_content) -> TextBlockParam:
        """Convert SVG resource to text block with XML code formatting.
        
        Args:
            resource_content: The resource content containing SVG data
            
        Returns:
            A TextBlockParam with formatted SVG content
        """
        if hasattr(resource_content, "text"):
            svg_content = resource_content.text
            return TextBlockParam(type="text", text=f"```xml\n{svg_content}\n```")
        return TextBlockParam(type="text", text="[SVG content could not be extracted]")
    
    @staticmethod
    def _convert_image_resource(resource, resource_content, mime_type, is_url, uri) -> ContentBlockParam:
        """Convert image resource to appropriate Anthropic image block.
        
        Args:
            resource: The original embedded resource
            resource_content: The resource contents
            mime_type: The detected MIME type
            is_url: Whether the resource is referenced by URL
            uri: The resource URI
            
        Returns:
            An ImageBlockParam or fallback TextBlockParam
        """
        # Check if image MIME type is supported
        if not AnthropicConverter._is_supported_image_type(mime_type):
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
        
        return AnthropicConverter._format_fail_message(resource, mime_type)
        
    @staticmethod
    def _convert_pdf_resource(resource_content, title, is_url, uri) -> DocumentBlockParam:
        """Convert PDF resource to appropriate Anthropic document block.
        
        Args:
            resource_content: The resource contents
            title: The document title
            is_url: Whether the resource is referenced by URL
            uri: The resource URI
            
        Returns:
            A DocumentBlockParam for the PDF
        """
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
        
        # PDF resource cannot be properly converted
        return TextBlockParam(type="text", text=f"[PDF resource could not be converted: {uri}]")
    
    @staticmethod
    def _convert_text_resource(resource_content, title, document_mode) -> ContentBlockParam:
        """Convert text resource to appropriate Anthropic block based on mode.
        
        Args:
            resource_content: The resource contents
            title: The document title
            document_mode: Whether to convert to document block (True) or text block (False)
            
        Returns:
            Either a DocumentBlockParam or TextBlockParam depending on document_mode
        """
        if not hasattr(resource_content, "text"):
            return TextBlockParam(type="text", text="[Text content could not be extracted]")
            
        # Create document block when in document mode
        if document_mode:
            return DocumentBlockParam(
                type="document",
                title=title,
                source=PlainTextSourceParam(
                    type="text",
                    media_type="text/plain",
                    data=resource_content.text,
                ),
            )
        
        # Return as simple text block when not in document mode
        return TextBlockParam(type="text", text=resource_content.text)
    
    @staticmethod
    def _convert_embedded_resource(
        resource: EmbeddedResource,
        document_mode: bool = True,
    ) -> ContentBlockParam:
        """Convert EmbeddedResource to appropriate Anthropic block type.

        Args:
            resource: The embedded resource to convert
            document_mode: Whether to convert text resources to Document blocks (True) or Text blocks (False)

        Returns:
            An appropriate ContentBlockParam for the resource
        """
        resource_content: TextResourceContents | BlobResourceContents = resource.resource
        uri: AnyUrl = resource_content.uri
        is_url: bool = uri.scheme in ("http", "https") if uri else False
        mime_type = AnthropicConverter._determine_mime_type(resource_content)
        # Extract title from URI
        title = extract_title_from_uri(uri) if uri else None

        # Dispatch based on MIME type to specialized handlers
        if mime_type == "image/svg+xml":
            return AnthropicConverter._convert_svg_resource(resource_content)
        elif is_image_mime_type(mime_type):
            return AnthropicConverter._convert_image_resource(resource, resource_content, mime_type, is_url, uri)
        elif mime_type == "application/pdf":
            return AnthropicConverter._convert_pdf_resource(resource_content, title, is_url, uri)
        elif is_text_mime_type(mime_type):
            return AnthropicConverter._convert_text_resource(resource_content, title, document_mode)

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
        # For tool results, we always use document_mode=False to get text blocks instead of document blocks
        anthropic_content = []

        for item in tool_result.content:
            if isinstance(item, EmbeddedResource):
                # For embedded resources, always use text mode in tool results
                resource_block = AnthropicConverter._convert_embedded_resource(
                    item, document_mode=False
                )
                anthropic_content.append(resource_block)
            else:
                # For other types (Text, Image), use standard conversion
                blocks = AnthropicConverter._convert_content_items(
                    [item], document_mode=False
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
            # Initialize collection for blocks that should be separate from the tool result
            separate_blocks = []
            filtered_content = []

            for item in result.content:
                # Text and images go in tool results, other resources (PDFs) go as separate blocks
                if isinstance(item, (TextContent, ImageContent)):
                    filtered_content.append(item)
                elif isinstance(item, EmbeddedResource):
                    # If it's a text resource, keep it in tool_content
                    if isinstance(item.resource, TextResourceContents):
                        filtered_content.append(item)
                    else:
                        # For binary resources like PDFs, convert and add as separate block
                        block = AnthropicConverter._convert_embedded_resource(
                            item, document_mode=True
                        )
                        separate_blocks.append(block)
                else:
                    filtered_content.append(item)

            # Process the filtered content directly without creating a new CallToolResult
            if filtered_content:
                # Create a tool result block with filtered content
                anthropic_content = []
                
                for item in filtered_content:
                    if isinstance(item, EmbeddedResource):
                        resource_block = AnthropicConverter._convert_embedded_resource(
                            item, document_mode=False
                        )
                        anthropic_content.append(resource_block)
                    else:
                        blocks = AnthropicConverter._convert_content_items(
                            [item], document_mode=False
                        )
                        anthropic_content.extend(blocks)
                        
                if not anthropic_content:
                    anthropic_content = [
                        TextBlockParam(type="text", text="[No content in tool result]")
                    ]
                    
                content_blocks.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_use_id,
                        content=anthropic_content,
                        is_error=result.isError,
                    )
                )
            else:
                # If there's no filtered content, still create a placeholder tool result
                content_blocks.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_use_id,
                        content=[TextBlockParam(type="text", text="[No content in tool result]")],
                        is_error=result.isError,
                    )
                )

            # Add separate blocks directly to the message
            content_blocks.extend(separate_blocks)

        return MessageParam(role="user", content=content_blocks)
