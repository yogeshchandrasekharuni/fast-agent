"""
Tests for serializing PromptMessageMultipart objects to delimited format.
"""

from mcp.types import TextContent, ImageContent, EmbeddedResource, TextResourceContents
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.prompt_serialization import multipart_messages_to_delimited_format
from mcp_agent.workflows.llm.providers.anthropic_multipart import (
    anthropic_to_multipart, multipart_to_anthropic
)


class TestPromptSerialization:
    """Tests for prompt serialization and delimited format conversion."""

    def test_multipart_to_delimited_format(self):
        """Test converting PromptMessageMultipart to delimited format for saving."""
        # Create multipart messages
        multipart_messages = [
            PromptMessageMultipart(
                role="user",
                content=[
                    TextContent(type="text", text="Hello!"),
                    TextContent(type="text", text="Can you help me?"),
                ],
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[
                    TextContent(type="text", text="Sure, I'd be happy to help."),
                    TextContent(type="text", text="What do you need assistance with?"),
                ],
            ),
        ]

        # Convert to delimited format
        delimited_content = multipart_messages_to_delimited_format(multipart_messages)

        # Verify results
        assert len(delimited_content) == 4
        assert delimited_content[0] == "---USER"
        assert delimited_content[1] == "Hello!\n\nCan you help me?"
        assert delimited_content[2] == "---ASSISTANT"
        assert (
            delimited_content[3]
            == "Sure, I'd be happy to help.\n\nWhat do you need assistance with?"
        )

    def test_multipart_with_resources_to_delimited_format(self):
        """Test converting PromptMessageMultipart with resources to delimited format."""
        # Create multipart messages with resources
        multipart_messages = [
            PromptMessageMultipart(
                role="user",
                content=[
                    TextContent(type="text", text="Check this code:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://example.py",
                            mimeType="text/x-python",
                            text="def hello():\n    print('Hello, world!')",
                        ),
                    ),
                ],
            ),
        ]

        # Convert to delimited format
        delimited_content = multipart_messages_to_delimited_format(multipart_messages)

        # Verify results
        assert len(delimited_content) == 3
        assert delimited_content[0] == "---USER"
        assert "Check this code:" in delimited_content[1]
        assert "def hello():" in delimited_content[1]
        assert delimited_content[2] == "---RESOURCE"
    
    def test_multi_role_messages_to_delimited_format(self):
        """Test converting a list of PromptMessageMultipart objects with different roles to delimited format."""
        # Create multipart messages with different roles
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="You are a helpful assistant.")],
            ),
            PromptMessageMultipart(
                role="user",
                content=[
                    TextContent(type="text", text="Hello!"),
                    TextContent(type="text", text="Can you help me?"),
                ],
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[
                    TextContent(type="text", text="I'd be happy to help."),
                    TextContent(type="text", text="What can I assist you with today?"),
                ],
            ),
        ]

        # Convert to delimited format
        delimited = multipart_messages_to_delimited_format(messages)

        # Verify results
        assert len(delimited) == 6  # 3 delimiters + 3 content blocks
        assert delimited[0] == "---USER"
        assert delimited[1] == "You are a helpful assistant."
        assert delimited[2] == "---USER"
        assert delimited[3] == "Hello!\n\nCan you help me?"
        assert delimited[4] == "---ASSISTANT"
        assert (
            delimited[5] == "I'd be happy to help.\n\nWhat can I assist you with today?"
        )
