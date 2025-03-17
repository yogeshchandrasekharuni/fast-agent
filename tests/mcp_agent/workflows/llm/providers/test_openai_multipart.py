# test_openai_multipart.py
import pytest
from typing import List, Dict, Any

from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    TextResourceContents,
    BlobResourceContents,
)

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.workflows.llm.providers.openai_multipart import (
    multipart_to_openai,
    openai_to_multipart,
)


class TestOpenAIMultipart:
    """Tests for OpenAI ⟷ PromptMessageMultipart conversions."""

    def test_simple_text_conversion(self):
        """Test conversion of simple text messages."""
        # Create a simple text multipart message
        multipart = PromptMessageMultipart(
            role="user", content=[TextContent(type="text", text="Hello, world!")]
        )

        # Convert to OpenAI format
        openai_msg = multipart_to_openai(multipart)

        # Verify simplified format for user messages
        assert openai_msg["role"] == "user"
        assert openai_msg["content"] == "Hello, world!"

        # Convert back to multipart
        result = openai_to_multipart(openai_msg)

        # Verify round-trip conversion
        assert result.role == "user"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello, world!"

    def test_assistant_message_conversion(self):
        """Test conversion of assistant messages."""
        # Create an assistant message
        multipart = PromptMessageMultipart(
            role="assistant",
            content=[
                TextContent(type="text", text="I'm an AI assistant."),
                TextContent(type="text", text="How can I help you today?"),
            ],
        )

        # Convert to OpenAI format
        openai_msg = multipart_to_openai(multipart)

        # Verify structured format for assistant messages
        assert openai_msg["role"] == "assistant"
        assert isinstance(openai_msg["content"], list)
        assert len(openai_msg["content"]) == 2
        assert openai_msg["content"][0]["type"] == "text"
        assert openai_msg["content"][0]["text"] == "I'm an AI assistant."
        assert openai_msg["content"][1]["type"] == "text"
        assert openai_msg["content"][1]["text"] == "How can I help you today?"

        # Convert back to multipart
        result = openai_to_multipart(openai_msg)

        # Verify round-trip conversion
        assert result.role == "assistant"
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[0].text == "I'm an AI assistant."
        assert result.content[1].type == "text"
        assert result.content[1].text == "How can I help you today?"

    def test_image_content_conversion(self):
        """Test conversion of messages with image content."""
        # Create a multipart message with an image
        multipart = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="What's in this image?"),
                ImageContent(
                    type="image",
                    data="iVBORw0KGgoAAAANSUhEUgAA",  # Partial base64
                    mimeType="image/png",
                ),
            ],
        )

        # Convert to OpenAI format
        openai_msg = multipart_to_openai(multipart)

        # Verify structured format with image
        assert openai_msg["role"] == "user"
        assert isinstance(openai_msg["content"], list)
        assert len(openai_msg["content"]) == 2
        assert openai_msg["content"][0]["type"] == "text"
        assert openai_msg["content"][1]["type"] == "image_url"
        assert (
            openai_msg["content"][1]["image_url"]["url"]
            == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA"
        )

        # Convert back to multipart
        result = openai_to_multipart(openai_msg)

        # Verify round-trip conversion
        assert result.role == "user"
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[1].type == "image"
        assert result.content[1].data == "iVBORw0KGgoAAAANSUhEUgAA"
        assert result.content[1].mimeType == "image/png"

    def test_text_resource_conversion(self):
        """Test conversion of text resources."""
        # Create a multipart message with a text resource
        multipart = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="Here's some code:"),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="resource://code.py",
                        mimeType="text/x-python",
                        text="def hello():\n    print('Hello, world!')",
                    ),
                ),
            ],
        )

        # Convert to OpenAI format
        openai_msg = multipart_to_openai(multipart)

        # Verify structured format with resource
        assert openai_msg["role"] == "user"
        assert isinstance(openai_msg["content"], list)
        assert len(openai_msg["content"]) == 2
        assert openai_msg["content"][0]["type"] == "text"
        assert openai_msg["content"][1]["type"] == "text"
        assert (
            "<fastagent:file uri='resource://code.py' mimetype='text/x-python'>"
            in openai_msg["content"][1]["text"]
        )
        assert "MIME: text/x-python" in openai_msg["content"][1]["text"]
        assert "def hello():" in openai_msg["content"][1]["text"]

        # Convert back to multipart
        result = openai_to_multipart(openai_msg)

        # Verify text resource information is preserved
        assert result.role == "user"
        assert len(result.content) == 2
        assert result.content[0].type == "text"

        # Second content should be converted back to a resource
        assert result.content[1].type == "resource"
        assert hasattr(result.content[1], "resource")
        assert result.content[1].resource.mimeType == "text/x-python"
        assert "def hello():" in result.content[1].resource.text

    def test_binary_resource_conversion(self):
        """Test conversion of binary resources."""
        # Create a multipart message with a binary resource
        multipart = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="Here's a binary file:"),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="resource://data.bin",
                        mimeType="application/octet-stream",
                        blob="SGVsbG8gV29ybGQ=",  # "Hello World" in base64
                    ),
                ),
            ],
        )

        # Convert to OpenAI format
        openai_msg = multipart_to_openai(multipart)

        # Verify structured format with binary resource
        assert openai_msg["role"] == "user"
        assert isinstance(openai_msg["content"], list)
        assert len(openai_msg["content"]) == 2
        assert openai_msg["content"][0]["type"] == "text"
        assert openai_msg["content"][1]["type"] == "text"
        assert (
            "Binary Resource: resource://data.bin" in openai_msg["content"][1]["text"]
        )
        assert "MIME: application/octet-stream" in openai_msg["content"][1]["text"]

        # Convert back to multipart - note binary resources become text representations
        result = openai_to_multipart(openai_msg)

        # Verify important info is present in text
        assert result.role == "user"
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[1].type == "text"
        assert "Binary Resource:" in result.content[1].text

    def test_message_list_conversion(self):
        """Test conversion of message lists."""
        # Create a list of multipart messages
        multiparts = [
            PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text="Hello")]
            ),
            PromptMessageMultipart(
                role="assistant", content=[TextContent(type="text", text="Hi there!")]
            ),
        ]

        # Convert to OpenAI format
        openai_msgs = multipart_to_openai(multiparts)

        # Verify list conversion
        assert isinstance(openai_msgs, list)
        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "user"
        assert openai_msgs[1]["role"] == "assistant"

        # Convert back to multipart
        results = openai_to_multipart(openai_msgs)

        # Verify round-trip conversion
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].role == "user"
        assert results[1].role == "assistant"

    def test_system_message_conversion(self):
        """Test conversion of system messages."""
        # Create a system message
        multipart = PromptMessageMultipart(
            role="user",
            content=[TextContent(type="text", text="You are a helpful assistant.")],
        )

        # Convert to OpenAI format
        openai_msg = multipart_to_openai(multipart)

        # Verify system role is preserved
        assert openai_msg["role"] == "user"
        assert openai_msg["content"] == "You are a helpful assistant."

        # Convert back to multipart
        result = openai_to_multipart(openai_msg)

        # Verify role and content
        assert result.role == "user"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "You are a helpful assistant."
