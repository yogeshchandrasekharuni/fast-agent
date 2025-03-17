"""
Utility functions for Anthropic integration with MCP.

Provides conversion between Anthropic message formats and PromptMessageMultipart,
leveraging existing code for resource handling and delimited formats.
"""

from typing import List

from anthropic.types import (
    Message,
    MessageParam,
    ContentBlockParam,
    TextBlockParam,
    ImageBlockParam,
)

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    TextResourceContents,
)

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


def anthropic_message_param_to_prompt_message_multipart(
    message_param: MessageParam,
) -> PromptMessageMultipart:
    """
    Convert an Anthropic MessageParam to a PromptMessageMultipart.

    Args:
        message_param: The Anthropic MessageParam to convert

    Returns:
        A PromptMessageMultipart representation
    """
    role = message_param["role"]
    content = message_param["content"]

    # Handle string content (user messages can be simple strings)
    if isinstance(content, str):
        return PromptMessageMultipart(
            role=role, content=[TextContent(type="text", text=content)]
        )

    # Convert content blocks to MCP content types
    mcp_contents = []

    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text = block.get("text", "")

                # Check if this is a resource marker
                if (
                    text
                    and (
                        text.startswith("[Resource:")
                        or text.startswith("[Binary Resource:")
                    )
                    and "\n" in text
                ):
                    header, content_text = text.split("\n", 1)
                    if "MIME:" in header:
                        mime_match = header.split("MIME:", 1)[1].split("]")[0].strip()
                        if (
                            mime_match != "text/plain"
                        ):  # Only process non-plain text resources
                            if (
                                "Resource:" in header
                                and "Binary Resource:" not in header
                            ):
                                uri = (
                                    header.split("Resource:", 1)[1]
                                    .split(",")[0]
                                    .strip()
                                )
                                mcp_contents.append(
                                    EmbeddedResource(
                                        type="resource",
                                        resource=TextResourceContents(
                                            uri=uri,
                                            mimeType=mime_match,
                                            text=content_text,
                                        ),
                                    )
                                )
                                continue

                # Regular text content
                mcp_contents.append(TextContent(type="text", text=text))

            elif block.get("type") == "image":
                # Image content
                source = block.get("source", {})
                if isinstance(source, dict) and source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    mcp_contents.append(
                        ImageContent(type="image", data=data, mimeType=media_type)
                    )

    return PromptMessageMultipart(role=role, content=mcp_contents)


def prompt_message_multipart_to_anthropic_message_param(
    multipart: PromptMessageMultipart,
) -> MessageParam:
    """
    Convert a PromptMessageMultipart to an Anthropic MessageParam.

    Args:
        multipart: The PromptMessageMultipart to convert

    Returns:
        An Anthropic MessageParam representation
    """
    # Convert MCP content to Anthropic content blocks
    content_blocks: List[ContentBlockParam] = []

    for content in multipart.content:
        if content.type == "text":
            # TextContent -> TextBlockParam
            content_blocks.append(TextBlockParam(type="text", text=content.text))

        elif content.type == "image":
            # ImageContent -> ImageBlockParam
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
            # Handle embedded resources
            if hasattr(content, "resource"):
                # Text resources
                if hasattr(content.resource, "text"):
                    mime_type = (
                        content.resource.mimeType
                        if hasattr(content.resource, "mimeType")
                        else "text/plain"
                    )
                    uri = (
                        content.resource.uri
                        if hasattr(content.resource, "uri")
                        else "resource://unknown"
                    )

                    if mime_type == "text/plain":
                        # Plain text resource becomes regular text content
                        content_blocks.append(
                            TextBlockParam(type="text", text=content.resource.text)
                        )
                    else:
                        # Non-plain text resource - add special format for round-trip conversion
                        content_blocks.append(
                            TextBlockParam(
                                type="text",
                                text=f"[Resource: {uri}, MIME: {mime_type}]\n{content.resource.text}",
                            )
                        )

                # Binary resources that are images
                elif (
                    hasattr(content.resource, "blob")
                    and hasattr(content.resource, "mimeType")
                    and content.resource.mimeType.startswith("image/")
                    and content.resource.mimeType != "image/svg+xml"
                ):
                    # Convert image resources to image blocks
                    content_blocks.append(
                        ImageBlockParam(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": content.resource.mimeType,
                                "data": content.resource.blob,
                            },
                        )
                    )
                # Other binary resources
                elif hasattr(content.resource, "blob"):
                    # Non-image binary resource - add a text note
                    mime_type = (
                        content.resource.mimeType
                        if hasattr(content.resource, "mimeType")
                        else "application/octet-stream"
                    )
                    uri = (
                        content.resource.uri
                        if hasattr(content.resource, "uri")
                        else "resource://unknown"
                    )
                    content_blocks.append(
                        TextBlockParam(
                            type="text",
                            text=f"[Binary Resource: {uri}, MIME: {mime_type}]",
                        )
                    )

    return {"role": multipart.role, "content": content_blocks}
