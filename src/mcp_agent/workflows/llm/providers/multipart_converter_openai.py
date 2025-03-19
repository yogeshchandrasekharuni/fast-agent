from typing import List, Union, Optional, Dict, Any, Tuple

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolResult,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.mime_utils import (
    guess_mime_type,
    is_text_mime_type,
    is_image_mime_type,
)
from mcp_agent.mcp.resource_utils import extract_title_from_uri

from mcp_agent.logging.logger import get_logger

_logger = get_logger("multipart_converter_openai")

# Define the types for OpenAI API
OpenAIContentBlock = Dict[str, Any]
OpenAIMessage = Dict[str, Any]


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
                        "filename": title or "document.pdf",
                        "file_data": f"data:application/pdf;base64,{resource_content.blob}",
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
    ) -> Union[OpenAIMessage, Tuple[OpenAIMessage, List[OpenAIMessage]]]:
        """
        Convert a CallToolResult to an OpenAI tool message.
        If the result contains non-text elements, those are converted to separate messages
        since OpenAI tool messages can only contain text.

        Args:
            tool_result: The tool result from a tool call
            tool_call_id: The ID of the associated tool use
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            Either a single OpenAI message for the tool response (if text only),
            or a tuple containing the tool message and a list of additional messages for non-text content
        """
        # Handle empty content case
        if not tool_result.content:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "[No content in tool result]",
            }

        # First, separate text and non-text content
        text_content = []
        non_text_content = []

        for item in tool_result.content:
            if isinstance(item, TextContent):
                text_content.append(item)
            else:
                non_text_content.append(item)

        # If we only have text content, process as before
        if not non_text_content:
            # Create a temporary PromptMessageMultipart to reuse the conversion logic
            temp_multipart = PromptMessageMultipart(role="user", content=text_content)

            # Convert using the same logic as user messages
            converted = OpenAIConverter.convert_to_openai(
                temp_multipart, concatenate_text_blocks=concatenate_text_blocks
            )

            # For tool messages, we need to extract and combine all text content
            if isinstance(converted["content"], str):
                content = converted["content"]
            else:
                # For compatibility with OpenAI's tool message format, combine all text blocks
                all_text = all(
                    block.get("type") == "text" for block in converted["content"]
                )

                if all_text and len(converted["content"]) > 0:
                    # Combine all text blocks
                    content = " ".join(
                        block.get("text", "") for block in converted["content"]
                    )
                else:
                    # Fallback for unexpected cases
                    content = "[Complex content converted to text]"

            # Create a tool message with the converted content
            return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

        # If we have mixed content or only non-text content

        # Process text content for the tool message
        tool_message_content = ""
        if text_content:
            temp_multipart = PromptMessageMultipart(role="user", content=text_content)
            converted = OpenAIConverter.convert_to_openai(
                temp_multipart, concatenate_text_blocks=True
            )

            if isinstance(converted["content"], str):
                tool_message_content = converted["content"]
            else:
                # Combine all text blocks
                all_text = [
                    block.get("text", "")
                    for block in converted["content"]
                    if block.get("type") == "text"
                ]
                tool_message_content = " ".join(all_text)

        if not tool_message_content:
            tool_message_content = "[Tool returned non-text content]"

        # Create the tool message with just the text
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_message_content,
        }

        # Process non-text content as a separate user message
        if non_text_content:
            # Create a multipart message with the non-text content
            non_text_multipart = PromptMessageMultipart(
                role="user", content=non_text_content
            )

            # Convert to OpenAI format
            user_message = OpenAIConverter.convert_to_openai(non_text_multipart)
            # Add tool_call_id to associate with the tool call
            user_message["tool_call_id"] = tool_call_id

            return (tool_message, [user_message])

        return tool_message

    @staticmethod
    def convert_function_results_to_openai(
        results: List[Tuple[str, CallToolResult]],
        concatenate_text_blocks: bool = False,
    ) -> List[OpenAIMessage]:
        """
        Convert a list of function call results to OpenAI messages.
        Handles cases where tool results contain non-text content by creating
        additional user messages as needed.

        Args:
            results: List of (tool_call_id, result) tuples
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            List of OpenAI API messages for tool responses
        """
        messages = []

        for tool_call_id, result in results:
            converted = OpenAIConverter.convert_tool_result_to_openai(
                tool_result=result,
                tool_call_id=tool_call_id,
                concatenate_text_blocks=concatenate_text_blocks,
            )

            # Handle the case where we have mixed content and get back a tuple
            if isinstance(converted, tuple):
                tool_message, additional_messages = converted
                messages.append(tool_message)
                messages.extend(additional_messages)
            else:
                # Single message case (text-only)
                messages.append(converted)

        return messages
