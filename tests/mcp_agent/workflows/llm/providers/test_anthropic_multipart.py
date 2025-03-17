# test_anthropic_multipart.py


from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    TextResourceContents,
    BlobResourceContents,
)

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.workflows.llm.providers.anthropic_multipart import (
    multipart_to_anthropic,
    anthropic_to_multipart,
)


class TestAnthropicMultipart:
    """Tests for Anthropic ⟷ PromptMessageMultipart conversions."""

    def test_simple_text_conversion(self):
        """Test conversion of simple text messages."""
        # Create a simple text multipart message
        multipart = PromptMessageMultipart(
            role="user", content=[TextContent(type="text", text="Hello, world!")]
        )

        # Convert to Anthropic format
        anthropic_msg = multipart_to_anthropic(multipart)

        # Verify simplified format for user messages
        assert anthropic_msg["role"] == "user"
        assert anthropic_msg["content"] == "Hello, world!"

        # Convert back to multipart
        result = anthropic_to_multipart(anthropic_msg)

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
                TextContent(type="text", text="I'm Claude, an AI assistant."),
                TextContent(type="text", text="How can I help you today?"),
            ],
        )

        # Convert to Anthropic format
        anthropic_msg = multipart_to_anthropic(multipart)

        # Verify structured format for assistant messages
        assert anthropic_msg["role"] == "assistant"
        assert isinstance(anthropic_msg["content"], list)
        assert len(anthropic_msg["content"]) == 2
        assert anthropic_msg["content"][0]["type"] == "text"
        assert anthropic_msg["content"][0]["text"] == "I'm Claude, an AI assistant."
        assert anthropic_msg["content"][1]["type"] == "text"
        assert anthropic_msg["content"][1]["text"] == "How can I help you today?"

        # Convert back to multipart
        result = anthropic_to_multipart(anthropic_msg)

        # Verify round-trip conversion
        assert result.role == "assistant"
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[0].text == "I'm Claude, an AI assistant."
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

        # Convert to Anthropic format
        anthropic_msg = multipart_to_anthropic(multipart)

        # Verify structured format with image
        assert anthropic_msg["role"] == "user"
        assert isinstance(anthropic_msg["content"], list)
        assert len(anthropic_msg["content"]) == 2
        assert anthropic_msg["content"][0]["type"] == "text"
        assert anthropic_msg["content"][1]["type"] == "image"
        assert anthropic_msg["content"][1]["source"]["type"] == "base64"
        assert anthropic_msg["content"][1]["source"]["media_type"] == "image/png"
        assert (
            anthropic_msg["content"][1]["source"]["data"] == "iVBORw0KGgoAAAANSUhEUgAA"
        )

        # Convert back to multipart
        result = anthropic_to_multipart(anthropic_msg)

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

        # Convert to Anthropic format
        anthropic_msg = multipart_to_anthropic(multipart)

        # Verify structured format with resource
        assert anthropic_msg["role"] == "user"
        assert isinstance(anthropic_msg["content"], list)
        assert len(anthropic_msg["content"]) == 2
        assert anthropic_msg["content"][0]["type"] == "text"
        assert anthropic_msg["content"][1]["type"] == "text"
        assert "Resource: resource://code.py" in anthropic_msg["content"][1]["text"]
        assert "MIME: text/x-python" in anthropic_msg["content"][1]["text"]
        assert "def hello():" in anthropic_msg["content"][1]["text"]

        # Convert back to multipart
        result = anthropic_to_multipart(anthropic_msg)

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

        # Convert to Anthropic format
        anthropic_msg = multipart_to_anthropic(multipart)

        # Verify structured format with binary resource
        assert anthropic_msg["role"] == "user"
        assert isinstance(anthropic_msg["content"], list)
        assert len(anthropic_msg["content"]) == 2
        assert anthropic_msg["content"][0]["type"] == "text"
        assert anthropic_msg["content"][1]["type"] == "text"
        assert (
            "Binary Resource: resource://data.bin"
            in anthropic_msg["content"][1]["text"]
        )
        assert "MIME: application/octet-stream" in anthropic_msg["content"][1]["text"]

        # Convert back to multipart - note binary resources become text representations
        result = anthropic_to_multipart(anthropic_msg)

        # Verify important info is present in text
        assert result.role == "user"
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[1].type == "text"
        assert "Binary Resource:" in result.content[1].text

    def test_tool_use_conversion(self):
        """Test conversion of tool_use blocks."""
        # Create an Anthropic message with a tool_use block
        anthropic_msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll help you with that."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "web_search",
                    "input": {"query": "weather in Paris"},
                },
            ],
        }

        # Convert to multipart
        multipart = anthropic_to_multipart(anthropic_msg)

        # Verify tool use is converted to a structured text representation
        assert multipart.role == "assistant"
        assert len(multipart.content) == 2
        assert multipart.content[0].type == "text"
        assert multipart.content[0].text == "I'll help you with that."
        assert multipart.content[1].type == "text"
        assert "Tool Call: web_search" in multipart.content[1].text
        assert "ID: tool_123" in multipart.content[1].text
        assert "weather in Paris" in multipart.content[1].text

    def test_tool_result_conversion(self):
        """Test conversion of tool_result blocks."""
        # Create an Anthropic message with a tool_result block
        anthropic_msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_123",
                    "content": [{"type": "text", "text": "Sunny, 25°C"}],
                    "is_error": False,
                }
            ],
        }

        # Convert to multipart
        multipart = anthropic_to_multipart(anthropic_msg)

        # Verify tool result is converted to an embedded resource
        assert multipart.role == "user"
        assert len(multipart.content) == 1
        assert multipart.content[0].type == "resource"
        assert multipart.content[0].resource.mimeType == "application/json"
        assert "Tool Result:" in multipart.content[0].resource.text
        assert "Sunny, 25°C" in multipart.content[0].resource.text

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

        # Convert to Anthropic format
        anthropic_msgs = multipart_to_anthropic(multiparts)

        # Verify list conversion
        assert isinstance(anthropic_msgs, list)
        assert len(anthropic_msgs) == 2
        assert anthropic_msgs[0]["role"] == "user"
        assert anthropic_msgs[1]["role"] == "assistant"

        # Convert back to multipart
        results = anthropic_to_multipart(anthropic_msgs)

        # Verify round-trip conversion
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].role == "user"
        assert results[1].role == "assistant"
