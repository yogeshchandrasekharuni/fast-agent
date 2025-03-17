"""
Tests for AnthropicMCPTypeConverter.
"""

from anthropic.types import (
    Message,
    MessageParam,
    TextBlock,
)

from mcp.types import (
    TextContent,
    PromptMessage,
    CreateMessageResult,
    SamplingMessage,
)

from mcp_agent.workflows.llm.providers import AnthropicSamplingConverter


class TestAnthropicMCPTypeConverter:
    def test_from_sampling_result_to_anthropic(self):
        """Test converting a simple text MCP message result to Anthropic Message."""
        # Create a simple MCP message result with text content
        mcp_result = CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text="Hello, this is a test!"),
            model="claude-3-opus-20240229",
            stopReason="endTurn",
        )

        # Convert to Anthropic Message
        anthropic_message = AnthropicSamplingConverter.from_sampling_result(mcp_result)

        # Verify the conversion
        assert isinstance(anthropic_message, Message)
        assert anthropic_message.role == "assistant"
        assert len(anthropic_message.content) == 1
        assert anthropic_message.content[0].type == "text"
        assert anthropic_message.content[0].text == "Hello, this is a test!"
        assert anthropic_message.model == "claude-3-opus-20240229"
        assert (
            anthropic_message.stop_reason == "end_turn"
        )  # Note the conversion from camelCase to snake_case

    def test_to_sampling_result_from_anthropic(self):
        """Test converting a simple Anthropic Message to MCP message result."""
        # Create a simple Anthropic Message with text content
        anthropic_message = Message(
            role="assistant",
            content=[TextBlock(type="text", text="Hello, this is a test!")],
            model="claude-3-opus-20240229",
            id="msg_123",
            stop_reason="end_turn",
            type="message",
            usage={"input_tokens": 10, "output_tokens": 20},
        )

        # Convert to MCP message result
        sampling_result = AnthropicSamplingConverter.to_sampling_result(
            anthropic_message
        )

        # Verify the conversion
        assert isinstance(sampling_result, CreateMessageResult)
        assert sampling_result.role == "assistant"
        assert sampling_result.content.type == "text"
        assert sampling_result.content.text == "Hello, this is a test!"
        assert sampling_result.model == "claude-3-opus-20240229"
        assert (
            sampling_result.stopReason == "endTurn"
        )  # Note the conversion from snake_case to camelCase

    def test_from_sampling_message_to_anthropic(self):
        """Test converting a simple MCP message param to Anthropic MessageParam."""
        # Create a simple MCP message param with text content
        message = SamplingMessage(
            role="user",
            content=TextContent(type="text", text="How does this conversion work?"),
        )

        # Convert to Anthropic MessageParam
        anthropic_param = AnthropicSamplingConverter.from_sampling_message(message)

        # Verify the conversion
        assert isinstance(anthropic_param, dict)  # MessageParam is a TypedDict
        assert anthropic_param["role"] == "user"
        assert len(anthropic_param["content"]) == 1
        assert anthropic_param["content"][0].type == "text"
        assert anthropic_param["content"][0].text == "How does this conversion work?"

    def test_to_sampling_message_from_anthropic(self):
        """Test converting a simple Anthropic MessageParam to MCP message param."""
        # Create a simple Anthropic MessageParam with text content
        #
        anthropic_param = MessageParam(
            role="user",
            content=[TextBlock(text="How does this conversion work?", type="text")],
        )

        # Convert to MCP message param
        mcp_param = AnthropicSamplingConverter.to_sampling_message(anthropic_param)

        # Verify the conversion
        assert isinstance(mcp_param, SamplingMessage)
        assert mcp_param.role == "user"
        assert mcp_param.content.type == "text"
        assert mcp_param.content.text == "How does this conversion work?"

    def test_from_prompt_message_user_to_anthropic(self):
        """Test converting a user PromptMessage to Anthropic MessageParam."""
        # Create a user PromptMessage
        prompt_message = PromptMessage(
            role="user",
            content=TextContent(type="text", text="Please explain this concept."),
        )

        # Convert to Anthropic MessageParam
        anthropic_param = AnthropicSamplingConverter.from_prompt_message(prompt_message)

        # Verify the conversion
        assert isinstance(anthropic_param, dict)  # MessageParam is a TypedDict
        assert anthropic_param["role"] == "user"
        assert anthropic_param["content"] == "Please explain this concept."

    def test_from_prompt_message_assistant(self):
        """Test converting an assistant PromptMessage to Anthropic MessageParam."""
        # Create an assistant PromptMessage
        prompt_message = PromptMessage(
            role="assistant",
            content=TextContent(type="text", text="Here's the explanation..."),
        )

        # Convert to Anthropic MessageParam
        anthropic_param = AnthropicSamplingConverter.from_prompt_message(prompt_message)

        # Verify the conversion
        assert isinstance(anthropic_param, dict)  # MessageParam is a TypedDict
        assert anthropic_param["role"] == "assistant"
        assert len(anthropic_param["content"]) == 1
        assert anthropic_param["content"][0]["type"] == "text"
        assert anthropic_param["content"][0]["text"] == "Here's the explanation..."

    def test_stop_reason_conversions(self):
        """Test various stop reason conversions."""
        from mcp_agent.workflows.llm.providers.sampling_converter_anthropic import (
            mcp_stop_reason_to_anthropic_stop_reason,
            anthropic_stop_reason_to_mcp_stop_reason,
        )

        # Test MCP to Anthropic conversions
        assert mcp_stop_reason_to_anthropic_stop_reason("endTurn") == "end_turn"
        assert mcp_stop_reason_to_anthropic_stop_reason("maxTokens") == "max_tokens"
        assert (
            mcp_stop_reason_to_anthropic_stop_reason("stopSequence") == "stop_sequence"
        )
        assert mcp_stop_reason_to_anthropic_stop_reason("toolUse") == "tool_use"
        assert mcp_stop_reason_to_anthropic_stop_reason(None) is None
        assert mcp_stop_reason_to_anthropic_stop_reason("unknown") == "unknown"

        # Test Anthropic to MCP conversions
        assert anthropic_stop_reason_to_mcp_stop_reason("end_turn") == "endTurn"
        assert anthropic_stop_reason_to_mcp_stop_reason("max_tokens") == "maxTokens"
        assert (
            anthropic_stop_reason_to_mcp_stop_reason("stop_sequence") == "stopSequence"
        )
        assert anthropic_stop_reason_to_mcp_stop_reason("tool_use") == "toolUse"
        assert anthropic_stop_reason_to_mcp_stop_reason(None) is None
        assert anthropic_stop_reason_to_mcp_stop_reason("unknown") == "unknown"
