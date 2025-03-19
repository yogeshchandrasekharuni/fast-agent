from typing import List, Union, Optional, Dict, Any, Tuple

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    TextResourceContents,
    BlobResourceContents,
    CallToolResult,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.mime_utils import (
    guess_mime_type,
    is_text_mime_type,
    is_image_mime_type,
)

from mcp_agent.logging.logger import get_logger

_logger = get_logger("multipart_converter_openai")

# Define the types for OpenAI API
OpenAIContentBlock = Dict[str, Any]
OpenAIMessage = Dict[str, Any]


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


class OpenAIConverter:
    """Converts MCP message types to OpenAI API format."""

    @staticmethod
    def convert_to_openai(
        multipart_msg: PromptMessageMultipart, concatenate_text_blocks: bool = False
    ) -> OpenAIMessage:
        """
        Convert a PromptMessageMultipart message to OpenAI API format.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            An OpenAI API message object
        """
        role = multipart_msg.role

        # Handle empty content
        if not multipart_msg.content:
            return {"role": role, "content": ""}

        # Assistant messages in OpenAI only support string content, not array of content blocks
        if role == "assistant":
            # Extract text from all text content blocks
            content_text = ""
            for item in multipart_msg.content:
                if isinstance(item, TextContent):
                    content_text += item.text
                # Other types are ignored for assistant messages in OpenAI

            return {"role": role, "content": content_text}

        # For user messages, convert each content block
        content_blocks = []

        for item in multipart_msg.content:
            try:
                if isinstance(item, TextContent):
                    content_blocks.append(OpenAIConverter._convert_text_content(item))

                elif isinstance(item, ImageContent):
                    content_blocks.append(OpenAIConverter._convert_image_content(item))

                elif isinstance(item, EmbeddedResource):
                    block = OpenAIConverter._convert_embedded_resource(item)
                    if block:
                        content_blocks.append(block)

                # Handle input_audio if implemented
                elif hasattr(item, "type") and getattr(item, "type") == "input_audio":
                    # This assumes an InputAudioContent class structure with input_audio attribute
                    if hasattr(item, "input_audio"):
                        content_blocks.append(
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": item.input_audio.get("data", ""),
                                    "format": item.input_audio.get("format", "wav"),
                                },
                            }
                        )
                    else:
                        _logger.warning(
                            "InputAudio content missing input_audio attribute"
                        )
                        fallback_text = "[Audio content missing data]"
                        content_blocks.append({"type": "text", "text": fallback_text})

                else:
                    _logger.warning(f"Unsupported content type: {type(item)}")
                    # Create a text block with information about the skipped content
                    fallback_text = f"[Unsupported content type: {type(item).__name__}]"
                    content_blocks.append({"type": "text", "text": fallback_text})

            except Exception as e:
                _logger.warning(f"Error converting content item: {e}")
                # Create a text block with information about the conversion error
                fallback_text = f"[Content conversion error: {str(e)}]"
                content_blocks.append({"type": "text", "text": fallback_text})

        # Special case: empty content list or only empty text blocks
        if not content_blocks:
            return {"role": role, "content": ""}

        # If we only have one text content and it's empty, return an empty string for content
        if (
            len(content_blocks) == 1
            and content_blocks[0]["type"] == "text"
            and not content_blocks[0]["text"]
        ):
            return {"role": role, "content": ""}

        # If concatenate_text_blocks is True, combine adjacent text blocks
        if concatenate_text_blocks:
            combined_blocks = []
            current_text = ""

            for block in content_blocks:
                if block["type"] == "text":
                    # Add to current text accumulator
                    if current_text:
                        current_text += " " + block["text"]
                    else:
                        current_text = block["text"]
                else:
                    # Non-text block found, flush accumulated text if any
                    if current_text:
                        combined_blocks.append({"type": "text", "text": current_text})
                        current_text = ""
                    # Add the non-text block
                    combined_blocks.append(block)

            # Don't forget any remaining text
            if current_text:
                combined_blocks.append({"type": "text", "text": current_text})

            content_blocks = combined_blocks

        return {"role": role, "content": content_blocks}

    @staticmethod
    def _convert_text_content(content: TextContent) -> OpenAIContentBlock:
        """Convert TextContent to OpenAI text content block."""
        return {"type": "text", "text": content.text}

    @staticmethod
    def _convert_image_content(content: ImageContent) -> OpenAIContentBlock:
        """Convert ImageContent to OpenAI image_url content block."""
        # OpenAI requires image URLs or data URIs for images
        image_url = {"url": f"data:{content.mimeType};base64,{content.data}"}

        # Check if the image has annotations for detail level
        # This would depend on your ImageContent implementation
        # If annotations are available, use them for the detail parameter
        if hasattr(content, "annotations") and content.annotations:
            if hasattr(content.annotations, "detail"):
                detail = content.annotations.detail
                if detail in ("auto", "low", "high"):
                    image_url["detail"] = detail

        return {"type": "image_url", "image_url": image_url}

    @staticmethod
    def _convert_embedded_resource(
        resource: EmbeddedResource,
    ) -> Optional[OpenAIContentBlock]:
        """Convert EmbeddedResource to appropriate OpenAI content block."""
        resource_content = resource.resource
        uri = resource_content.uri

        # Use mime_utils to guess MIME type if not provided
        if resource_content.mimeType is None and uri:
            mime_type = guess_mime_type(str(uri))
            _logger.info(f"MIME type not provided, guessed {mime_type} for {uri}")
        else:
            mime_type = resource_content.mimeType or "application/octet-stream"

        is_url: bool = str(uri).startswith(("http://", "https://"))
        title = extract_title_from_uri(uri) if uri else "resource"

        # Handle image resources
        if is_image_mime_type(mime_type) and mime_type != "image/svg+xml":
            image_url = {}

            if is_url:
                image_url["url"] = str(uri)
            elif hasattr(resource_content, "blob"):
                image_url["url"] = f"data:{mime_type};base64,{resource_content.blob}"
            else:
                _logger.warning(f"Image resource missing both URL and blob data: {uri}")
                return {"type": "text", "text": f"[Image missing data: {title}]"}

            # Check for detail level in annotations if available
            if hasattr(resource, "annotations") and resource.annotations:
                if hasattr(resource.annotations, "detail"):
                    detail = resource.annotations.detail
                    if detail in ("auto", "low", "high"):
                        image_url["detail"] = detail

            return {"type": "image_url", "image_url": image_url}

        # Handle PDF resources - OpenAI has specific file format for PDFs
        elif mime_type == "application/pdf":
            if is_url:
                # OpenAI doesn't directly support PDF URLs, only file_id or base64
                _logger.warning(f"PDF URL not directly supported in OpenAI API: {uri}")
                fallback_text = f"[PDF URL: {uri}]\nOpenAI requires PDF files to be uploaded or provided as base64 data."
                return {"type": "text", "text": fallback_text}
            elif hasattr(resource_content, "blob"):
                return {
                    "type": "file",
                    "file": {
                        "file_name": title or "document.pdf",
                        "file_data": resource_content.blob,
                    },
                }

        # Handle SVG as text with fastagent:file tags
        elif mime_type == "image/svg+xml":
            if hasattr(resource_content, "text"):
                file_text = (
                    f'<fastagent:file title="{title}" mimetype="{mime_type}">\n'
                    f"{resource_content.text}\n"
                    f"</fastagent:file>"
                )
                return {"type": "text", "text": file_text}

        # Handle text resources with fastagent:file tags
        elif is_text_mime_type(mime_type):
            if hasattr(resource_content, "text"):
                # Wrap in fastagent:file tags for text resources
                file_text = (
                    f'<fastagent:file title="{title}" mimetype="{mime_type}">\n'
                    f"{resource_content.text}\n"
                    f"</fastagent:file>"
                )
                return {"type": "text", "text": file_text}

        # Handle other binary formats that OpenAI supports with file type
        # Currently, OpenAI supports PDFs for comprehensive viewing, but we can try
        # to use the file type for other binary formats as well for future compatibility
        elif hasattr(resource_content, "blob"):
            # For now, we'll use file type for PDFs only, and use fallback for others
            if mime_type == "application/pdf":
                return {
                    "type": "file",
                    "file": {"file_name": title, "file_data": resource_content.blob},
                }
            else:
                # For other binary formats, create a text message mentioning the resource
                return {
                    "type": "text",
                    "text": f"[Binary resource: {title} ({mime_type})]",
                }

        # Default fallback - convert to text if possible
        if hasattr(resource_content, "text"):
            # For anything with text content that isn't handled specially above,
            # use the raw text without special formatting
            return {"type": "text", "text": resource_content.text}

        _logger.warning(f"Unable to convert resource with MIME type: {mime_type}")
        return {
            "type": "text",
            "text": f"[Unsupported resource: {title} ({mime_type})]",
        }

    @staticmethod
    def convert_tool_result_to_openai(
        tool_result: CallToolResult,
        tool_call_id: str,
        concatenate_text_blocks: bool = False,
    ) -> OpenAIMessage:
        """
        Convert a CallToolResult to an OpenAI tool message.
        Non text elements are returned as User messages (tool results can only be text)

        Args:
            tool_result: The tool result from a tool call
            tool_call_id: The ID of the associated tool use
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            An OpenAI API message for the tool response
        """
        # Create a temporary PromptMessageMultipart to reuse the conversion logic
        temp_multipart = PromptMessageMultipart(
            role="user", content=tool_result.content
        )

        # Convert using the same logic as user messages
        converted = OpenAIConverter.convert_to_openai(
            temp_multipart, concatenate_text_blocks=concatenate_text_blocks
        )

        # If the conversion resulted in a string content (e.g., for empty content)
        if isinstance(converted["content"], str):
            content = converted["content"]
        else:
            # For compatibility with OpenAI's tool message format, we may need to
            # convert the array of content blocks to a single string for simple cases
            # Check if all blocks are text
            all_text = all(
                block.get("type") == "text" for block in converted["content"]
            )

            if all_text and len(converted["content"]) > 0:
                # Combine all text blocks
                content = " ".join(
                    block.get("text", "") for block in converted["content"]
                )
            else:
                # For mixed content, we have to use the content blocks
                # OpenAI tool messages can have content blocks just like user messages
                content = converted["content"]

        # Create a tool message with the converted content
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    @staticmethod
    def convert_function_results_to_openai(
        results: List[Tuple[str, CallToolResult]],
        concatenate_text_blocks: bool = False,
    ) -> List[OpenAIMessage]:
        """
        Convert a list of function call results to OpenAI tool messages.

        Args:
            results: List of (tool_call_id, result) tuples
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            List of OpenAI API messages for tool responses
        """
        messages = []

        for tool_call_id, result in results:
            tool_message = OpenAIConverter.convert_tool_result_to_openai(
                tool_result=result,
                tool_call_id=tool_call_id,
                concatenate_text_blocks=concatenate_text_blocks,
            )
            messages.append(tool_message)

        return messages
