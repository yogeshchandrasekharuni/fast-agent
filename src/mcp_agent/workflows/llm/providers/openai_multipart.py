# openai_multipart.py
"""
Clean utilities for converting between PromptMessageMultipart and OpenAI message formats.
Each function handles all content types consistently and is designed for simple testing.
"""

from typing import Dict, Any, Union, List, cast

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    TextResourceContents,
    BlobResourceContents,
)

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


def multipart_to_openai(
    multipart: Union[PromptMessageMultipart, List[PromptMessageMultipart]],
) -> Union[ChatCompletionMessageParam, List[ChatCompletionMessageParam]]:
    """
    Convert PromptMessageMultipart to OpenAI MessageParam format.

    Args:
        multipart: Single multipart message or list of multipart messages

    Returns:
        Equivalent message(s) in OpenAI format
    """
    if isinstance(multipart, list):
        return [_multipart_message_to_openai(mp) for mp in multipart]
    return _multipart_message_to_openai(multipart)


def openai_to_multipart(
    message: Union[
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        List[Union[ChatCompletionMessage, ChatCompletionMessageParam]],
    ],
) -> Union[PromptMessageMultipart, List[PromptMessageMultipart]]:
    """
    Convert OpenAI messages to PromptMessageMultipart format.

    Args:
        message: OpenAI Message, MessageParam, or list of them

    Returns:
        Equivalent message(s) in PromptMessageMultipart format
    """
    if isinstance(message, list):
        return [_openai_message_to_multipart(m) for m in message]
    return _openai_message_to_multipart(message)


def _multipart_message_to_openai(
    multipart: PromptMessageMultipart,
) -> ChatCompletionMessageParam:
    """Convert a single PromptMessageMultipart to an OpenAI ChatCompletionMessageParam."""
    # Simple case: single text content
    if len(multipart.content) == 1 and multipart.content[0].type == "text":
        content = multipart.content[0].text

        # Create the appropriate message type based on role
        if multipart.role == "user":
            return ChatCompletionUserMessageParam(role="user", content=content)
        elif multipart.role == "assistant":
            return ChatCompletionAssistantMessageParam(
                role="assistant", content=content
            )

        # not possible with PromptMessageMultipart
        # elif multipart.role == "system":
        #     return ChatCompletionSystemMessageParam(role="system", content=content)
        # else:
        #     # Default to user for unknown roles
        #     return ChatCompletionUserMessageParam(role="user", content=content)

    # Complex case: multiple content parts
    content_parts = []

    for content in multipart.content:
        if content.type == "text":
            # Text content
            content_parts.append(
                ChatCompletionContentPartTextParam(type="text", text=content.text)
            )

        elif content.type == "image":
            # Image content
            data_url = f"data:{content.mimeType};base64,{content.data}"
            content_parts.append(
                ChatCompletionContentPartImageParam(
                    type="image_url", image_url={"url": data_url}
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
                        # Plain text resource - regular text
                        content_parts.append(
                            ChatCompletionContentPartTextParam(
                                type="text", text=content.resource.text
                            )
                        )
                    else:
                        # Non-plain text resource - use our extension format
                        content_parts.append(
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text=f"<fastagent:file uri='{uri}' mimetype='{mime_type}'>\n{content.resource.text}\n</fastagent:file>",
                            )
                        )

                # Binary resources
                elif hasattr(content.resource, "blob"):
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

                    if mime_type.startswith("image/") and mime_type != "image/svg+xml":
                        # Image content
                        data_url = f"data:{mime_type};base64,{content.resource.blob}"
                        content_parts.append(
                            ChatCompletionContentPartImageParam(
                                type="image_url", image_url={"url": data_url}
                            )
                        )
                    else:
                        # Other binary resource
                        content_parts.append(
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text=f"[Binary Resource: {uri}, MIME: {mime_type}]",
                            )
                        )

    # Create appropriate message type based on role
    if multipart.role == "user":
        return ChatCompletionUserMessageParam(role="user", content=content_parts)
    elif multipart.role == "assistant":
        return ChatCompletionAssistantMessageParam(
            role="assistant", content=content_parts
        )
    elif multipart.role == "system":
        return ChatCompletionSystemMessageParam(role="system", content=content_parts)
    else:
        # Default to user for any other role
        return ChatCompletionUserMessageParam(role="user", content=content_parts)


def _openai_message_to_multipart(
    message: Union[ChatCompletionMessage, Dict[str, Any]],
) -> PromptMessageMultipart:
    """Convert a single OpenAI message to PromptMessageMultipart."""
    # Get role and content from message
    if isinstance(message, dict):
        role = message.get("role", "assistant")
        content = message.get("content", "")
    else:
        role = message.role
        content = message.content

    mcp_contents = []

    # Handle string content (simple case)
    if isinstance(content, str):
        mcp_contents.append(TextContent(type="text", text=content))

    # Handle list of content parts
    elif isinstance(content, list):
        for part in content:
            part_type = (
                part.get("type")
                if isinstance(part, dict)
                else getattr(part, "type", None)
            )

            # Handle text content
            if part_type == "text":
                text = (
                    part.get("text")
                    if isinstance(part, dict)
                    else getattr(part, "text", "")
                )

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

                        # If not text/plain, create an embedded resource
                        if mime_match != "text/plain":
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

            # Handle image content
            elif part_type == "image_url":
                image_url = (
                    part.get("image_url", {})
                    if isinstance(part, dict)
                    else getattr(part, "image_url", None)
                )
                if image_url:
                    url = (
                        image_url.get("url")
                        if isinstance(image_url, dict)
                        else getattr(image_url, "url", "")
                    )
                    if url and url.startswith("data:image/"):
                        # Handle base64 data URLs
                        mime_type = url.split(";")[0].replace("data:", "")
                        data = url.split(",")[1]
                        mcp_contents.append(
                            ImageContent(type="image", data=data, mimeType=mime_type)
                        )

            # Handle explicit resource types
            elif (
                part_type == "resource"
                and isinstance(part, dict)
                and "resource" in part
            ):
                resource = part["resource"]
                if isinstance(resource, dict):
                    # Text resource
                    if "text" in resource and "mimeType" in resource:
                        mime_type = resource["mimeType"]
                        uri = resource.get("uri", "resource://unknown")

                        if mime_type == "text/plain":
                            mcp_contents.append(
                                TextContent(type="text", text=resource["text"])
                            )
                        else:
                            mcp_contents.append(
                                EmbeddedResource(
                                    type="resource",
                                    resource=TextResourceContents(
                                        text=resource["text"],
                                        mimeType=mime_type,
                                        uri=uri,
                                    ),
                                )
                            )
                    # Binary resource
                    elif "blob" in resource and "mimeType" in resource:
                        mime_type = resource["mimeType"]
                        uri = resource.get("uri", "resource://unknown")

                        if (
                            mime_type.startswith("image/")
                            and mime_type != "image/svg+xml"
                        ):
                            mcp_contents.append(
                                ImageContent(
                                    type="image",
                                    data=resource["blob"],
                                    mimeType=mime_type,
                                )
                            )
                        else:
                            mcp_contents.append(
                                EmbeddedResource(
                                    type="resource",
                                    resource=BlobResourceContents(
                                        blob=resource["blob"],
                                        mimeType=mime_type,
                                        uri=uri,
                                    ),
                                )
                            )

    return PromptMessageMultipart(role=role, content=mcp_contents)
