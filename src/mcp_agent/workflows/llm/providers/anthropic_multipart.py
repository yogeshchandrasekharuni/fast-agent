# anthropic_multipart.py
"""
Clean utilities for converting between PromptMessageMultipart and Anthropic message formats.
Each function handles all content types consistently and is designed for simple testing.
"""

import json
from typing import List, Dict, Any, Union, cast

from anthropic.types import (
    Message,
    MessageParam,
    TextBlockParam,
    ImageBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
)

from mcp.types import TextContent, ImageContent, EmbeddedResource, TextResourceContents

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


def multipart_to_anthropic(
    multipart: Union[PromptMessageMultipart, List[PromptMessageMultipart]],
) -> Union[MessageParam, List[MessageParam]]:
    """
    Convert PromptMessageMultipart to Anthropic MessageParam format.

    Args:
        multipart: Single multipart message or list of multipart messages

    Returns:
        Equivalent message(s) in Anthropic format
    """
    if isinstance(multipart, list):
        return [_multipart_message_to_anthropic(mp) for mp in multipart]
    return _multipart_message_to_anthropic(multipart)


def anthropic_to_multipart(
    message: Union[Message, MessageParam, List[Union[Message, MessageParam]]],
) -> Union[PromptMessageMultipart, List[PromptMessageMultipart]]:
    """
    Convert Anthropic messages to PromptMessageMultipart format.

    Args:
        message: Anthropic Message, MessageParam, or list of them

    Returns:
        Equivalent message(s) in PromptMessageMultipart format
    """
    if isinstance(message, list):
        return [_anthropic_message_to_multipart(m) for m in message]
    return _anthropic_message_to_multipart(message)


def _multipart_message_to_anthropic(multipart: PromptMessageMultipart) -> MessageParam:
    """Convert a single PromptMessageMultipart to Anthropic MessageParam."""
    role = multipart.role
    content_blocks = []

    for content in multipart.content:
        if content.type == "text":
            # Simple text content
            content_blocks.append(TextBlockParam(type="text", text=content.text))

        elif content.type == "image":
            # Image content
            content_blocks.append(
                ImageBlockParam(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": content.mimeType,
                        "data": content.data,
                    },
                )
            )

        elif content.type == "resource":
            # Handle embedded resources based on their type
            if hasattr(content.resource, "text"):
                # Text-based resource
                mime_type = getattr(content.resource, "mimeType", "text/plain")
                resource_text = content.resource.text
                uri = str(getattr(content.resource, "uri", "resource://unknown"))

                # Plain text resources are just text blocks
                if mime_type == "text/plain":
                    content_blocks.append(
                        TextBlockParam(type="text", text=resource_text)
                    )
                else:
                    # Other text resources get special formatting
                    resource_block = TextBlockParam(
                        type="text",
                        text=f"[Resource: {uri}, MIME: {mime_type}]\n{resource_text}",
                    )
                    content_blocks.append(resource_block)

            elif hasattr(content.resource, "blob"):
                # Binary resource
                mime_type = getattr(
                    content.resource, "mimeType", "application/octet-stream"
                )
                blob_data = content.resource.blob
                uri = str(getattr(content.resource, "uri", "resource://unknown"))

                # Images become image blocks
                if mime_type.startswith("image/") and mime_type != "image/svg+xml":
                    content_blocks.append(
                        ImageBlockParam(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": mime_type,
                                "data": blob_data,
                            },
                        )
                    )
                else:
                    # Other binary resources become informational text
                    content_blocks.append(
                        TextBlockParam(
                            type="text",
                            text=f"[Binary Resource: {uri}, MIME: {mime_type}]",
                        )
                    )

    # Handle special case: user message with single text block
    if (
        role == "user"
        and len(content_blocks) == 1
        and isinstance(content_blocks[0], dict)
        and content_blocks[0].get("type") == "text"
    ):
        # User messages can use simplified string format
        return {"role": role, "content": content_blocks[0]["text"]}

    # Normal case: message with content blocks
    return {"role": role, "content": content_blocks}


def _anthropic_message_to_multipart(
    message: Union[Message, MessageParam],
) -> PromptMessageMultipart:
    """Convert a single Anthropic message to PromptMessageMultipart."""
    # Extract role and content depending on input type
    if isinstance(message, dict):
        role = cast(str, message.get("role", "user"))
        content = message.get("content", "")
    else:
        role = message.role
        content = message.content

    multipart_content = []

    # Handle string content (user messages can be simple strings)
    if isinstance(content, str):
        multipart_content.append(TextContent(type="text", text=content))
        return PromptMessageMultipart(role=role, content=multipart_content)

    # Process content blocks
    for block in content:
        block_type = (
            block.get("type")
            if isinstance(block, dict)
            else getattr(block, "type", None)
        )

        if block_type == "text":
            # Handle text block
            text = (
                block.get("text")
                if isinstance(block, dict)
                else getattr(block, "text", "")
            )

            # Check for special resource marker format
            if (
                isinstance(text, str)
                and (
                    text.startswith("[Resource:")
                    or text.startswith("[Binary Resource:")
                )
                and "\n" in text
            ):
                header, content_text = text.split("\n", 1)

                if "MIME:" in header:
                    # Extract resource metadata
                    mime_match = header.split("MIME:", 1)[1].split("]")[0].strip()
                    if (
                        mime_match != "text/plain"
                        and "Resource:" in header
                        and "Binary Resource:" not in header
                    ):
                        # It's a text resource with non-plain MIME type
                        uri = header.split("Resource:", 1)[1].split(",")[0].strip()
                        multipart_content.append(
                            EmbeddedResource(
                                type="resource",
                                resource=TextResourceContents(
                                    uri=uri, mimeType=mime_match, text=content_text
                                ),
                            )
                        )
                        continue

            # Plain text content
            multipart_content.append(TextContent(type="text", text=text))

        elif block_type == "image":
            # Handle image block
            if isinstance(block, dict) and "source" in block:
                source = block["source"]
                if isinstance(source, dict) and source.get("type") == "base64":
                    multipart_content.append(
                        ImageContent(
                            type="image",
                            data=source.get("data", ""),
                            mimeType=source.get("media_type", "image/png"),
                        )
                    )
            elif hasattr(block, "source") and hasattr(block.source, "data"):
                multipart_content.append(
                    ImageContent(
                        type="image",
                        data=block.source.data,
                        mimeType=block.source.media_type,
                    )
                )

        elif block_type == "tool_use":
            # Convert tool_use to structured text
            name = (
                block.get("name")
                if isinstance(block, dict)
                else getattr(block, "name", "")
            )
            input_data = (
                block.get("input")
                if isinstance(block, dict)
                else getattr(block, "input", {})
            )
            id = (
                block.get("id") if isinstance(block, dict) else getattr(block, "id", "")
            )

            tool_text = f"Tool Call: {name}\nID: {id}\nInput: {json.dumps(input_data, indent=2)}"
            multipart_content.append(TextContent(type="text", text=tool_text))

        elif block_type == "tool_result":
            # Convert tool_result to embedded resource
            tool_id = (
                block.get("tool_use_id")
                if isinstance(block, dict)
                else getattr(block, "tool_use_id", "")
            )
            is_error = (
                block.get("is_error")
                if isinstance(block, dict)
                else getattr(block, "is_error", False)
            )
            result_content = (
                block.get("content")
                if isinstance(block, dict)
                else getattr(block, "content", [])
            )

            result_text = "Tool Result:\n"
            if is_error:
                result_text += "Error: "

            # Extract text from content
            if isinstance(result_content, list):
                for item in result_content:
                    if isinstance(item, dict) and "text" in item:
                        result_text += item["text"]
                    elif hasattr(item, "text"):
                        result_text += item.text
            else:
                result_text += str(result_content)

            multipart_content.append(
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"resource://tool_result_{tool_id}",
                        mimeType="application/json",
                        text=result_text,
                    ),
                )
            )

    return PromptMessageMultipart(role=role, content=multipart_content)
