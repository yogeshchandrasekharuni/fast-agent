"""
Tests for OpenAIMCPTypeConverter.
"""

from openai.types.chat import ChatCompletionMessage, ChatCompletionUserMessageParam

from mcp.types import (
    TextContent,
    PromptMessage,
    CreateMessageResult,
    SamplingMessage,
)

from mcp_agent.workflows.llm.providers import OpenAISamplingConverter


class TestOpenAIMCPTypeConverter:
    def test_from_mcp_message_result_simple_text(self):
        """Test converting a simple text MCP message result to OpenAI ChatCompletionMessage."""
        # Create a simple MCP message result with text content
        mcp_result = CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text="Hello, this is a test!"),
            model="gpt-4o",
            stopReason="length",
        )

        # Convert to OpenAI ChatCompletionMessage
        openai_message = OpenAISamplingConverter.from_sampling_result(mcp_result)

        # Verify the conversion
        assert isinstance(openai_message, ChatCompletionMessage)
        assert openai_message.role == "assistant"
        assert openai_message.content == "Hello, this is a test!"

    def test_to_mcp_message_result_simple_text(self):
        """Test converting a simple OpenAI ChatCompletionMessage to MCP message result."""
        # Create a simple OpenAI ChatCompletionMessage with text content
        openai_message = ChatCompletionMessage(
            role="assistant",
            content="Hello, this is a test!",
            function_call=None,
            tool_calls=None,
        )

        # Convert to MCP message result
        mcp_result = OpenAISamplingConverter.to_sampling_result(openai_message)

        # Verify the conversion
        assert isinstance(mcp_result, CreateMessageResult)
        assert mcp_result.role == "assistant"
        assert mcp_result.content.type == "text"
        assert mcp_result.content.text == "Hello, this is a test!"

    def test_from_mcp_message_param_simple_text(self):
        """Test converting a simple MCP message param to OpenAI ChatCompletionMessageParam."""
        # Create a simple MCP message param with text content
        mcp_param = SamplingMessage(
            role="user",
            content=TextContent(type="text", text="How does this conversion work?"),
        )

        # Convert to OpenAI ChatCompletionMessageParam
        openai_param = OpenAISamplingConverter.from_sampling_message(mcp_param)

        # Verify the conversion
        assert isinstance(
            openai_param, dict
        )  # ChatCompletionMessageParam is a TypedDict
        assert openai_param["role"] == "user"
        assert openai_param["content"]["text"] == "How does this conversion work?"

    def test_to_mcp_message_param_simple_text(self):
        """Test converting a simple OpenAI ChatCompletionMessageParam to MCP message param."""
        # Create a simple OpenAI ChatCompletionMessageParam with text content

        param = ChatCompletionUserMessageParam(
            role="user", content="How does this conversion work?"
        )

        # Convert to MCP message param
        mcp_param = OpenAISamplingConverter.to_sampling_message(param)

        # Verify the conversion
        assert isinstance(mcp_param, SamplingMessage)
        assert mcp_param.role == "user"
        assert mcp_param.content.type == "text"
        assert mcp_param.content.text == "How does this conversion work?"

    def test_from_mcp_prompt_message_user(self):
        """Test converting a user PromptMessage to OpenAI ChatCompletionMessageParam."""
        # Create a user PromptMessage
        prompt_message = PromptMessage(
            role="user",
            content=TextContent(type="text", text="Please explain this concept."),
        )

        # Convert to OpenAI ChatCompletionMessageParam
        openai_param = OpenAISamplingConverter.from_prompt_message(prompt_message)

        # Verify the conversion
        assert isinstance(
            openai_param, dict
        )  # ChatCompletionMessageParam is a TypedDict
        assert openai_param["role"] == "user"
        assert openai_param["content"] == "Please explain this concept."

    def test_from_mcp_prompt_message_assistant(self):
        """Test converting an assistant PromptMessage to OpenAI ChatCompletionMessageParam."""
        # Create an assistant PromptMessage
        prompt_message = PromptMessage(
            role="assistant",
            content=TextContent(type="text", text="Here's the explanation..."),
        )

        # Convert to OpenAI ChatCompletionMessageParam
        openai_param = OpenAISamplingConverter.from_prompt_message(prompt_message)

        # Verify the conversion
        assert isinstance(
            openai_param, dict
        )  # ChatCompletionMessageParam is a TypedDict
        assert openai_param["role"] == "assistant"
        assert openai_param["content"] == "Here's the explanation..."

    def test_edge_cases_null_content(self):
        """Test handling of null/None content."""
        # Test handling None content in OpenAI message
        openai_message = ChatCompletionMessage(
            role="assistant",
            content=None,
            function_call=None,
            tool_calls=None,
        )

        mcp_result = OpenAISamplingConverter.to_sampling_result(openai_message)
        assert mcp_result.content.text == ""
