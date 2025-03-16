from typing import List, Union
from pydantic import BaseModel

from mcp.types import (
    PromptMessage,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Role,
    GetPromptResult,
)


class PromptMessageMultipart(BaseModel):
    """
    Extension of PromptMessage that handles multiple content parts.
    Internally converts to/from a sequence of standard PromptMessages.
    """

    role: Role
    content: List[Union[TextContent, ImageContent, EmbeddedResource]]

    @classmethod
    def from_prompt_messages(
        cls, messages: List[PromptMessage]
    ) -> List["PromptMessageMultipart"]:
        """Convert a sequence of PromptMessages into PromptMessageMultipart objects."""
        if not messages:
            return []

        result = []
        current_group = None
        current_role = None

        for msg in messages:
            if msg.role != current_role:
                # Role changed, start new message
                if current_group is not None:
                    result.append(current_group)
                current_role = msg.role
                current_group = cls(role=msg.role, content=[msg.content])
            else:
                # Same role, add to current message
                current_group.content.append(msg.content)

        # Add the last group
        if current_group is not None:
            result.append(current_group)

        return result

    def to_prompt_messages(self) -> List[PromptMessage]:
        """Convert this PromptMessageMultipart to a sequence of standard PromptMessages."""
        return [
            PromptMessage(role=self.role, content=content_part)
            for content_part in self.content
        ]

    @classmethod
    def parse_get_prompt_result(
        cls, result: GetPromptResult
    ) -> List["PromptMessageMultipart"]:
        """Parse a GetPromptResult into PromptMessageMultipart objects."""
        return cls.from_prompt_messages(result.messages)


def multipart_messages_to_delimited_format(
    messages: List[PromptMessageMultipart], 
    user_delimiter: str = "---USER",
    assistant_delimiter: str = "---ASSISTANT",
    system_delimiter: str = "---SYSTEM"
) -> List[str]:
    """
    Convert a list of PromptMessageMultipart objects to delimited format for saving.
    
    Args:
        messages: List of PromptMessageMultipart objects
        user_delimiter: The delimiter to use for user messages
        assistant_delimiter: The delimiter to use for assistant messages
        system_delimiter: The delimiter to use for system messages (not currently used)
        
    Returns:
        A list of strings in delimited format suitable for saving to a file
    """
    delimited_content = []
    
    for message in messages:
        # Add role delimiter
        if message.role == "user":
            delimited_content.append(user_delimiter)
        elif message.role == "assistant":
            delimited_content.append(assistant_delimiter)
        elif message.role == "system":
            # Skip system messages - MCP only supports user and assistant roles
            continue
        else:
            # Skip other unsupported roles
            continue
        
        # Add text content, joining multiple parts with newlines
        # Only include parts that actually have text
        text_parts = []
        for content in message.content:
            if content.type == "text":
                text_parts.append(content.text)
            elif content.type == "resource" and hasattr(content.resource, "text"):
                text_parts.append(content.resource.text)
        
        # Join all text parts with double newlines
        message_text = "\n\n".join(text_parts)
        delimited_content.append(message_text)
    
    return delimited_content


def delimited_format_to_multipart_messages(
    content: str,
    user_delimiter: str = "---USER",
    assistant_delimiter: str = "---ASSISTANT",
    system_delimiter: str = "---SYSTEM",
) -> List[PromptMessageMultipart]:
    """
    Parse delimited content into a list of PromptMessageMultipart objects.
    
    Args:
        content: The delimited content string to parse
        user_delimiter: The delimiter for user messages
        assistant_delimiter: The delimiter for assistant messages
        system_delimiter: The delimiter for system messages (not currently used)
        
    Returns:
        A list of PromptMessageMultipart objects
    """
    lines = content.split("\n")
    messages = []
    
    current_role = None
    current_content = []
    
    for line in lines:
        if line.strip() == user_delimiter:
            # Save previous message if it exists
            if current_role is not None and current_content:
                messages.append(
                    PromptMessageMultipart(
                        role=current_role,
                        content=[TextContent(type="text", text="\n".join(current_content).strip())]
                    )
                )
            
            # Start a new user message
            current_role = "user"
            current_content = []
        
        elif line.strip() == assistant_delimiter:
            # Save previous message if it exists
            if current_role is not None and current_content:
                messages.append(
                    PromptMessageMultipart(
                        role=current_role,
                        content=[TextContent(type="text", text="\n".join(current_content).strip())]
                    )
                )
            
            # Start a new assistant message
            current_role = "assistant"
            current_content = []
        
        elif current_role is not None:
            # Add to current message content
            current_content.append(line)
    
    # Add the final message if it exists
    if current_role is not None and current_content:
        messages.append(
            PromptMessageMultipart(
                role=current_role,
                content=[TextContent(type="text", text="\n".join(current_content).strip())]
            )
        )
    
    return messages
