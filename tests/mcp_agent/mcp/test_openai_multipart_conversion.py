"""
Tests for converting between OpenAI message types and PromptMessageMultipart.
"""

from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from mcp.types import (
    TextContent,
    ImageContent,
)

from mcp_agent.mcp.prompt_message_multipart import (
    PromptMessageMultipart,
    multipart_messages_to_delimited_format,
)
from mcp_agent.workflows.llm.openai_utils import (
    openai_message_to_prompt_message_multipart,
    openai_message_param_to_prompt_message_multipart,
    prompt_message_multipart_to_openai_message_param,
)


class TestOpenAIMultipartConversion:
    """Tests for converting between OpenAI types and PromptMessageMultipart."""

    def test_openai_message_to_prompt_message_multipart_text_only(self):
        """Test converting an OpenAI message with text content to PromptMessageMultipart."""
        # Since ChatCompletionMessage is restricted to role="assistant",
        # we'll use a ChatCompletionAssistantMessageParam which is what's actually
        # returned by the API and handled in augmented_llm_openai.py
        message = ChatCompletionAssistantMessageParam(
            role="assistant", content="I'm an AI assistant. How can I help you today?"
        )

        # Convert to PromptMessageMultipart
        multipart = openai_message_to_prompt_message_multipart(message)

        # Verify results
        assert multipart.role == "assistant"
        assert len(multipart.content) == 1
        assert multipart.content[0].type == "text"
        assert (
            multipart.content[0].text
            == "I'm an AI assistant. How can I help you today?"
        )

    def test_openai_message_param_to_prompt_message_multipart(self):
        """Test converting an OpenAI MessageParam with text content to PromptMessageMultipart."""
        # Create a simple message param using the actual type
        message_param = ChatCompletionAssistantMessageParam(
            role="assistant",
            content="I'm an AI assistant and I can help with various tasks.",
        )

        # Convert to PromptMessageMultipart
        multipart = openai_message_param_to_prompt_message_multipart(message_param)

        # Verify results
        assert multipart.role == "assistant"
        assert len(multipart.content) == 1
        assert multipart.content[0].type == "text"
        assert (
            multipart.content[0].text
            == "I'm an AI assistant and I can help with various tasks."
        )

    def test_openai_message_param_with_parts_to_prompt_message_multipart(self):
        """Test converting an OpenAI MessageParam with content parts to PromptMessageMultipart."""
        # Create a message param with multiple content parts
        # We'll use a dictionary directly since the exact structure matters more than the type
        message_param = ChatCompletionUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "Look at this image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                },
                {"type": "text", "text": "What do you see?"},
            ],
        )

        # Convert to PromptMessageMultipart
        multipart = openai_message_param_to_prompt_message_multipart(message_param)

        # Verify results
        assert multipart.role == "user"
        assert len(multipart.content) == 3
        assert multipart.content[0].type == "text"
        assert multipart.content[0].text == "Look at this image:"
        assert multipart.content[1].type == "image"
        assert multipart.content[1].data == "iVBORw0KGgo="
        assert multipart.content[1].mimeType == "image/png"
        assert multipart.content[2].type == "text"
        assert multipart.content[2].text == "What do you see?"

    def test_prompt_message_multipart_to_openai_message_param_single(self):
        """Test converting a PromptMessageMultipart with a single text item to OpenAI MessageParam."""
        # Create a multipart message with a single text content
        # Note: MCP's Role only supports 'user' and 'assistant', not 'system'
        multipart = PromptMessageMultipart(
            role="user",
            content=[TextContent(type="text", text="You are a helpful assistant.")],
        )

        # Convert to OpenAI MessageParam
        message_param = prompt_message_multipart_to_openai_message_param(multipart)

        # Verify results - should use the simpler string representation
        assert message_param["role"] == "user"
        assert message_param["content"] == "You are a helpful assistant."
        assert isinstance(message_param["content"], str)

    def test_prompt_message_multipart_to_openai_message_param_multiple(self):
        """Test converting a PromptMessageMultipart with multiple items to OpenAI MessageParam."""
        # Create a multipart message with multiple content items
        multipart = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="Hello!"),
                TextContent(type="text", text="Can you help me with this?"),
            ],
        )

        # Convert to OpenAI MessageParam
        message_param = prompt_message_multipart_to_openai_message_param(multipart)

        # Verify results - should use the content list
        assert message_param["role"] == "user"
        assert isinstance(message_param["content"], list)
        assert len(message_param["content"]) == 2
        assert message_param["content"][0]["type"] == "text"
        assert message_param["content"][0]["text"] == "Hello!"
        assert message_param["content"][1]["type"] == "text"
        assert message_param["content"][1]["text"] == "Can you help me with this?"

    def test_prompt_message_multipart_to_openai_message_param_with_image(self):
        """Test converting a PromptMessageMultipart with image content to OpenAI MessageParam."""
        # Create a multipart message with text and image content
        multipart = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="Look at this:"),
                ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png"),
            ],
        )

        # Convert to OpenAI MessageParam
        message_param = prompt_message_multipart_to_openai_message_param(multipart)

        # Verify results
        assert message_param["role"] == "user"
        assert isinstance(message_param["content"], list)
        assert len(message_param["content"]) == 2

        assert message_param["content"][0]["type"] == "text"
        assert message_param["content"][0]["text"] == "Look at this:"

        assert message_param["content"][1]["type"] == "image_url"
        assert "image_url" in message_param["content"][1]
        assert (
            message_param["content"][1]["image_url"]["url"]
            == "data:image/png;base64,iVBORw0KGgo="
        )

    def test_multipart_messages_to_delimited_format(self):
        """Test converting a list of PromptMessageMultipart objects to delimited format."""
        # Create multipart messages with different roles
        # Note: Using user role instead of system since MCP only supports user and assistant
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

    def test_round_trip_conversion(self):
        """Test round-trip conversion from OpenAI MessageParam to PromptMessageMultipart and back."""
        # Original OpenAI message param
        original_param = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=[
                {"type": "text", "text": "Here's what I found:"},
                {"type": "text", "text": "The answer is 42."},
            ],
        )

        # Convert to PromptMessageMultipart
        multipart = openai_message_param_to_prompt_message_multipart(original_param)

        # Convert back to OpenAI MessageParam
        result_param = prompt_message_multipart_to_openai_message_param(multipart)

        # Verify the result matches the original
        assert result_param["role"] == original_param["role"]
        assert isinstance(result_param["content"], list)
        assert len(result_param["content"]) == 2

        for i in range(2):
            assert result_param["content"][i]["type"] == "text"
            assert (
                result_param["content"][i]["text"]
                == original_param["content"][i]["text"]
            )
