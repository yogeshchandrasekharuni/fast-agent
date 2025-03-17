"""
Tests for using PromptMessageMultipart in augmented LLMs.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import PromptMessage, TextContent, GetPromptResult
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM


@pytest.mark.asyncio
async def test_apply_prompt_template_with_multipart():
    """Test applying a prompt template using PromptMessageMultipart."""

    # Create a mock AugmentedLLM instance
    llm = MagicMock(spec=AugmentedLLM)
    llm.logger = MagicMock()
    llm.history = MagicMock()
    llm.type_converter = MagicMock()
    llm.show_prompt_loaded = AsyncMock()
    llm.generate_str = AsyncMock(return_value="Generated response")
    llm.provider = "Test"  # Add a provider that will use the fallback path

    # Make the methods accessible for testing
    llm.apply_prompt_template = AugmentedLLM.apply_prompt_template.__get__(
        llm, AugmentedLLM
    )
    llm._apply_prompt_template_provider_specific = (
        AugmentedLLM._apply_prompt_template_provider_specific.__get__(llm, AugmentedLLM)
    )

    # Create test prompt messages
    prompt_messages = [
        PromptMessage(
            role="assistant", content=TextContent(type="text", text="I'm an assistant")
        ),
        PromptMessage(
            role="user", content=TextContent(type="text", text="Tell me about Python")
        ),
    ]

    # Create a GetPromptResult
    prompt_result = GetPromptResult(messages=prompt_messages, description="Test prompt")

    # Test the method with user as last message (should generate a response)
    result = await llm.apply_prompt_template(prompt_result, "test_prompt")

    # Verify the method converted to PromptMessageMultipart
    llm.show_prompt_loaded.assert_awaited_once()

    # Verify it extracted the user message and generated a response
    llm.generate_str.assert_awaited_once_with("Tell me about Python")
    assert result == "Generated response"

    # Reset mocks
    llm.show_prompt_loaded.reset_mock()
    llm.generate_str.reset_mock()
    llm.history.reset_mock()

    # Test with assistant as last message (should return the text directly)
    prompt_messages = [
        PromptMessage(
            role="user", content=TextContent(type="text", text="Tell me about Python")
        ),
        PromptMessage(
            role="assistant",
            content=TextContent(type="text", text="Python is a programming language"),
        ),
    ]

    prompt_result = GetPromptResult(messages=prompt_messages, description="Test prompt")

    result = await llm.apply_prompt_template(prompt_result, "test_prompt")

    # Verify it didn't generate a response
    llm.generate_str.assert_not_awaited()

    # Verify it returned the assistant's response directly
    assert result == "Python is a programming language"


@pytest.mark.asyncio
async def test_apply_prompt_template_with_multiple_content():
    """Test applying a prompt template with multiple content parts."""

    # Create a mock AugmentedLLM instance
    llm = MagicMock(spec=AugmentedLLM)
    llm.logger = MagicMock()
    llm.history = MagicMock()
    llm.type_converter = MagicMock()
    llm.show_prompt_loaded = AsyncMock()
    llm.generate_str = AsyncMock(return_value="Generated response")
    llm.provider = "Test"  # Add a provider that will use the fallback path

    # Make the methods accessible for testing
    llm.apply_prompt_template = AugmentedLLM.apply_prompt_template.__get__(
        llm, AugmentedLLM
    )
    llm._apply_prompt_template_provider_specific = (
        AugmentedLLM._apply_prompt_template_provider_specific.__get__(llm, AugmentedLLM)
    )

    # Create test prompt messages with multiple content parts
    # This simulates a scenario where the same user sent multiple messages
    prompt_messages = [
        PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
        PromptMessage(
            role="user",
            content=TextContent(type="text", text="I have a question about Python"),
        ),
        PromptMessage(
            role="user",
            content=TextContent(type="text", text="How do I create a function?"),
        ),
    ]

    # Create a GetPromptResult
    prompt_result = GetPromptResult(
        messages=prompt_messages, description="Test prompt with multiple parts"
    )

    # Test the method
    await llm.apply_prompt_template(prompt_result, "test_prompt")

    # In PromptMessageMultipart, these should be combined into a single message
    # with three content parts, and we should extract all text parts
    "Hello\nI have a question about Python\nHow do I create a function?"
    llm.generate_str.assert_awaited_once()

    # The arguments passed to generate_str depend on the implementation
    # Our implementation should join the text parts with newlines
    args = llm.generate_str.call_args[0][0]
    assert "Hello" in args
    assert "I have a question about Python" in args
    assert "How do I create a function?" in args


@pytest.mark.asyncio
async def test_apply_prompt_template_with_images():
    """Test applying a prompt template with image content."""

    # Create a mock AugmentedLLM instance
    llm = MagicMock(spec=AugmentedLLM)
    llm.logger = MagicMock()
    llm.history = MagicMock()
    llm.type_converter = MagicMock()
    llm.show_prompt_loaded = AsyncMock()
    llm.generate_str = AsyncMock(return_value="Generated response")
    llm.provider = "Test"  # Use the fallback path

    # Make the methods accessible for testing
    llm.apply_prompt_template = AugmentedLLM.apply_prompt_template.__get__(
        llm, AugmentedLLM
    )
    llm._apply_prompt_template_provider_specific = (
        AugmentedLLM._apply_prompt_template_provider_specific.__get__(llm, AugmentedLLM)
    )

    # Create a GetPromptResult with a message containing an image
    from mcp.types import ImageContent

    prompt_result = GetPromptResult(
        messages=[
            PromptMessage(
                role="user",
                content=ImageContent(
                    type="image", data="base64encodeddata", mimeType="image/png"
                ),
            ),
        ],
        description="Test prompt with image",
    )

    # Test the method
    result = await llm.apply_prompt_template(prompt_result, "test_prompt")

    # Verify the LLM received a text representation of the image
    llm.generate_str.assert_awaited_once()
    args = llm.generate_str.call_args[0][0]
    assert "[Image: image/png]" in args

    # Test with assistant message containing an image
    llm.generate_str.reset_mock()
    prompt_result = GetPromptResult(
        messages=[
            PromptMessage(
                role="assistant",
                content=ImageContent(
                    type="image", data="base64encodeddata", mimeType="image/jpeg"
                ),
            ),
        ],
        description="Test prompt with assistant image",
    )

    result = await llm.apply_prompt_template(prompt_result, "test_prompt")

    # Verify we didn't call generate_str (assistant message is returned directly)
    llm.generate_str.assert_not_awaited()

    # Verify the result contains a description of the image
    assert "[Image: image/jpeg]" in result
    assert "message contained non-text content" in result


@pytest.mark.asyncio
async def test_apply_prompt_template_with_mixed_content():
    """Test applying a prompt template with mixed content (text + image)."""

    # Create a mock AugmentedLLM instance
    llm = MagicMock(spec=AugmentedLLM)
    llm.logger = MagicMock()
    llm.history = MagicMock()
    llm.type_converter = MagicMock()
    llm.show_prompt_loaded = AsyncMock()
    llm.generate_str = AsyncMock(return_value="Generated response")

    # Test with provider-specific path - we want to make sure Anthropic messages are passed directly
    llm.provider = "Anthropic"

    # Make the methods accessible for testing
    llm.apply_prompt_template = AugmentedLLM.apply_prompt_template.__get__(
        llm, AugmentedLLM
    )

    # Import the Anthropic-specific implementation
    from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

    llm._apply_prompt_template_provider_specific = (
        AnthropicAugmentedLLM._apply_prompt_template_provider_specific.__get__(
            llm, AnthropicAugmentedLLM
        )
    )

    # Create messages with mixed content types
    from mcp.types import ImageContent, TextContent

    prompt_result = GetPromptResult(
        messages=[
            PromptMessage(
                role="assistant", content=TextContent(type="text", text="Hello")
            ),
            PromptMessage(
                role="user",
                content=TextContent(type="text", text="Look at this image:"),
            ),
            PromptMessage(
                role="user",
                content=ImageContent(
                    type="image", data="base64encodeddata", mimeType="image/png"
                ),
            ),
        ],
        description="Test prompt with mixed content",
    )

    # Mock the anthropic conversion function
    with patch(
        "mcp_agent.workflows.llm.anthropic_utils.prompt_message_multipart_to_anthropic_message_param"
    ) as mock_convert:
        # Set up the mock to return a properly formatted message
        mock_convert.return_value = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image:"},
                {
                    "type": "image_url",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "base64encodeddata",
                    },
                },
            ],
        }

        # Test the method
        await llm.apply_prompt_template(prompt_result, "test_prompt")

        # Verify we're using the provider-specific path for multimodal content
        mock_convert.assert_called()
        # The Anthropic LLM should be called with a complete message object
        assert isinstance(llm.generate_str.call_args[0][0], dict)
        assert llm.generate_str.call_args[0][0]["role"] == "user"


@pytest.mark.asyncio
async def test_apply_prompt_template_with_anthropic():
    """Test applying a prompt template using Anthropic-specific conversion."""

    # Create a mock AugmentedLLM instance for Anthropic
    llm = MagicMock(spec=AugmentedLLM)
    llm.logger = MagicMock()
    llm.history = MagicMock()
    llm.type_converter = MagicMock()
    llm.show_prompt_loaded = AsyncMock()
    llm.generate_str = AsyncMock(return_value="Generated response")
    llm.provider = "Anthropic"  # Set provider to Anthropic

    # Make the methods accessible for testing
    llm.apply_prompt_template = AugmentedLLM.apply_prompt_template.__get__(
        llm, AugmentedLLM
    )

    # Import the Anthropic-specific implementation
    from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

    llm._apply_prompt_template_provider_specific = (
        AnthropicAugmentedLLM._apply_prompt_template_provider_specific.__get__(
            llm, AnthropicAugmentedLLM
        )
    )

    # Create a GetPromptResult with multiple messages to ensure we hit the conversion path
    prompt_result = GetPromptResult(
        messages=[
            PromptMessage(
                role="assistant", content=TextContent(type="text", text="I'm Claude")
            ),
            PromptMessage(
                role="user",
                content=TextContent(type="text", text="Tell me about Claude"),
            ),
        ],
        description="Test prompt for Anthropic",
    )

    # Mock the anthropic conversion function to verify it's called
    with patch(
        "mcp_agent.workflows.llm.anthropic_utils.prompt_message_multipart_to_anthropic_message_param"
    ) as mock_convert:
        # Set up the mock to return a properly formatted message
        mock_convert.return_value = {
            "role": "user",
            "content": [{"type": "text", "text": "Tell me about Claude"}],
        }

        # Call apply_prompt_template
        await llm.apply_prompt_template(prompt_result, "test_prompt")

        # Verify that our anthropic conversion utility was called
        assert mock_convert.called


@pytest.mark.asyncio
async def test_apply_prompt_template_with_openai():
    """Test applying a prompt template using OpenAI-specific conversion."""

    # Create a mock AugmentedLLM instance for OpenAI
    llm = MagicMock(spec=AugmentedLLM)
    llm.logger = MagicMock()
    llm.history = MagicMock()
    llm.type_converter = MagicMock()
    llm.show_prompt_loaded = AsyncMock()
    llm.generate_str = AsyncMock(return_value="Generated response")
    llm.provider = "OpenAI"  # Set provider to OpenAI
    llm.instruction = None
    llm.default_request_params = None

    # Make the methods accessible for testing
    llm.apply_prompt_template = AugmentedLLM.apply_prompt_template.__get__(
        llm, AugmentedLLM
    )

    # Import the OpenAI-specific implementation
    from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

    llm._apply_prompt_template_provider_specific = (
        OpenAIAugmentedLLM._apply_prompt_template_provider_specific.__get__(
            llm, OpenAIAugmentedLLM
        )
    )

    # Create a GetPromptResult with multiple messages to ensure we hit the conversion path
    prompt_result = GetPromptResult(
        messages=[
            PromptMessage(
                role="assistant", content=TextContent(type="text", text="I'm GPT-4")
            ),
            PromptMessage(
                role="user",
                content=TextContent(type="text", text="Tell me about GPT-4"),
            ),
        ],
        description="Test prompt for OpenAI",
    )

    # Mock the openai conversion function to verify it's called
    with patch(
        "mcp_agent.workflows.llm.openai_utils.prompt_message_multipart_to_openai_message_param"
    ) as mock_convert:
        # Set up the mock to return a properly formatted message
        mock_convert.return_value = {"role": "user", "content": "Tell me about GPT-4"}

        # Call apply_prompt_template
        await llm.apply_prompt_template(prompt_result, "test_prompt")

        # Verify that our openai conversion utility was called
        assert mock_convert.called
