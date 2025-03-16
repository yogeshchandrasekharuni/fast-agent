"""
Tests for converting between Anthropic SDK messages and PromptMessageMultipart.
"""

from typing import List, Dict, Any

from anthropic.types import (
    ContentBlock,
    Message,
    MessageParam,
    TextBlock,
)

from mcp.types import (
    PromptMessage,
    TextContent,
    ImageContent,
)

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


def create_anthropic_message(role: str, contents: List[ContentBlock]) -> Message:
    """Helper to create an Anthropic Message object."""
    return Message(
        id="msg_test",
        model="claude-3-7-sonnet-20250219",
        role=role,
        type="message",
        content=contents,
        stop_reason="end_turn",
        usage={"input_tokens": 100, "output_tokens": 100},
    )


def create_anthropic_message_param(
    role: str, content: List[Dict[str, Any]] | str
) -> MessageParam:
    """Helper to create an Anthropic MessageParam object."""
    return {"role": role, "content": content}


class TestAnthropicMultipartConversion:
    """Tests for converting between Anthropic types and PromptMessageMultipart."""

    def test_anthropic_message_to_prompt_message_multipart_text_only(self):
        """Test converting an Anthropic Message with text content to PromptMessageMultipart."""
        # Create an Anthropic Message with text content
        anthropic_message = create_anthropic_message(
            role="assistant",
            contents=[
                TextBlock(type="text", text="Hello!"),
                TextBlock(type="text", text="How can I help you today?"),
            ],
        )

        # Convert to MCP PromptMessages
        mcp_messages = []
        for content_block in anthropic_message.content:
            if content_block.type == "text":
                mcp_messages.append(
                    PromptMessage(
                        role=anthropic_message.role,
                        content=TextContent(type="text", text=content_block.text),
                    )
                )

        # Convert to PromptMessageMultipart
        multipart_messages = PromptMessageMultipart.from_prompt_messages(mcp_messages)

        # Verify results
        assert len(multipart_messages) == 1
        assert multipart_messages[0].role == "assistant"
        assert len(multipart_messages[0].content) == 2
        assert multipart_messages[0].content[0].text == "Hello!"
        assert multipart_messages[0].content[1].text == "How can I help you today?"

    def test_anthropic_message_param_to_prompt_message_multipart(self):
        """Test converting an Anthropic MessageParam with text blocks to PromptMessageMultipart."""
        # Create an Anthropic MessageParam with text content
        message_param = create_anthropic_message_param(
            role="user",
            content=[
                {"type": "text", "text": "I have a question."},
                {"type": "text", "text": "What's the capital of France?"},
            ],
        )

        # Convert to MCP PromptMessages
        mcp_messages = []
        for content_block in message_param["content"]:
            if isinstance(content_block, dict) and content_block.get("type") == "text":
                mcp_messages.append(
                    PromptMessage(
                        role=message_param["role"],
                        content=TextContent(type="text", text=content_block["text"]),
                    )
                )

        # Convert to PromptMessageMultipart
        multipart_messages = PromptMessageMultipart.from_prompt_messages(mcp_messages)

        # Verify results
        assert len(multipart_messages) == 1
        assert multipart_messages[0].role == "user"
        assert len(multipart_messages[0].content) == 2
        assert multipart_messages[0].content[0].text == "I have a question."
        assert multipart_messages[0].content[1].text == "What's the capital of France?"

    def test_prompt_message_multipart_to_anthropic_message_param(self):
        """Test converting a PromptMessageMultipart to Anthropic MessageParam."""
        # Create a PromptMessageMultipart with text content
        multipart = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="Hello"),
                TextContent(type="text", text="I need help with a problem."),
            ],
        )

        # Convert to Anthropic MessageParam
        content_blocks = []
        for content in multipart.content:
            if content.type == "text":
                content_blocks.append({"type": "text", "text": content.text})

        message_param = {"role": multipart.role, "content": content_blocks}

        # Verify results
        assert message_param["role"] == "user"
        assert len(message_param["content"]) == 2
        assert message_param["content"][0]["text"] == "Hello"
        assert message_param["content"][1]["text"] == "I need help with a problem."

    def test_prompt_message_multipart_to_anthropic_message(self):
        """Test conceptual conversion from PromptMessageMultipart to Anthropic Message."""
        # Create a PromptMessageMultipart with text content
        multipart = PromptMessageMultipart(
            role="assistant",
            content=[
                TextContent(type="text", text="Here's what I found:"),
                TextContent(type="text", text="The answer is 42."),
            ],
        )

        # Convert to content blocks
        content_blocks = []
        for content in multipart.content:
            if content.type == "text":
                content_blocks.append(TextBlock(type="text", text=content.text))

        # Create an Anthropic Message
        message = create_anthropic_message(role=multipart.role, contents=content_blocks)

        # Verify results
        assert message.role == "assistant"
        assert len(message.content) == 2
        assert message.content[0].text == "Here's what I found:"
        assert message.content[1].text == "The answer is 42."

    def test_mixed_content_conversion(self):
        """Test converting messages with mixed content types (images and text)."""
        # Create a multipart message with text and image content
        multipart = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="Look at this image:"),
                ImageContent(
                    type="image", data="base64_encoded_image_data", mimeType="image/png"
                ),
                TextContent(type="text", text="What do you see?"),
            ],
        )

        # Convert to Anthropic MessageParam
        # Note: For images, we'd convert differently based on Anthropic's API requirements
        content_blocks = []
        for content in multipart.content:
            if content.type == "text":
                content_blocks.append({"type": "text", "text": content.text})
            elif content.type == "image":
                # Simplified representation - actual Anthropic format would be different
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content.mimeType,
                            "data": content.data,
                        },
                    }
                )

        message_param = {"role": multipart.role, "content": content_blocks}

        # Verify results
        assert message_param["role"] == "user"
        assert len(message_param["content"]) == 3
        assert message_param["content"][0]["type"] == "text"
        assert message_param["content"][0]["text"] == "Look at this image:"
        assert message_param["content"][1]["type"] == "image"
        assert (
            message_param["content"][1]["source"]["data"] == "base64_encoded_image_data"
        )
        assert message_param["content"][2]["type"] == "text"
        assert message_param["content"][2]["text"] == "What do you see?"

    def test_round_trip_conversion(self):
        """Test round-trip conversion from Anthropic Message to PromptMessageMultipart and back."""
        # Create an Anthropic Message
        original_message = create_anthropic_message(
            role="assistant",
            contents=[
                TextBlock(type="text", text="I've analyzed your request."),
                TextBlock(type="text", text="Here are my findings..."),
            ],
        )

        # Convert to MCP PromptMessages
        mcp_messages = []
        for content_block in original_message.content:
            if content_block.type == "text":
                mcp_messages.append(
                    PromptMessage(
                        role=original_message.role,
                        content=TextContent(type="text", text=content_block.text),
                    )
                )

        # Convert to PromptMessageMultipart
        multipart_messages = PromptMessageMultipart.from_prompt_messages(mcp_messages)

        # Convert back to content blocks
        content_blocks = []
        for content in multipart_messages[0].content:
            if content.type == "text":
                content_blocks.append(TextBlock(type="text", text=content.text))

        # Create a new Anthropic Message
        new_message = create_anthropic_message(
            role=multipart_messages[0].role, contents=content_blocks
        )

        # Verify the content is preserved
        assert new_message.role == original_message.role
        assert len(new_message.content) == len(original_message.content)

        for i, block in enumerate(original_message.content):
            assert new_message.content[i].text == block.text

    def test_save_delimited_format(self):
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
        delimited_content = []

        for message in multipart_messages:
            # Add role delimiter
            if message.role == "user":
                delimited_content.append("---USER")
            elif message.role == "assistant":
                delimited_content.append("---ASSISTANT")

            # Add text content, joining multiple parts with newlines
            message_text = "\n\n".join(
                [
                    content.text
                    for content in message.content
                    if hasattr(content, "text")
                ]
            )
            delimited_content.append(message_text)

        # Verify results
        assert len(delimited_content) == 4
        assert delimited_content[0] == "---USER"
        assert delimited_content[1] == "Hello!\n\nCan you help me?"
        assert delimited_content[2] == "---ASSISTANT"
        assert (
            delimited_content[3]
            == "Sure, I'd be happy to help.\n\nWhat do you need assistance with?"
        )
