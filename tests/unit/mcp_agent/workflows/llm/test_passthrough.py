import pytest
import asyncio
from mcp.types import Role, TextContent
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.workflows.llm.augmented_llm_passthrough import PassthroughLLM
from mcp.types import GetPromptResult, PromptMessage


async def test_apply_prompt_template_concatenates_all_content():
    # Create a PassthroughLLM instance
    passthrough = PassthroughLLM(name="TestPassthrough")
    
    # Create test messages
    messages = [
        PromptMessage(role=Role.ASSISTANT, content=TextContent(text="message 1")),
        PromptMessage(role=Role.USER, content=TextContent(text="message 2")),
        PromptMessage(role=Role.ASSISTANT, content=TextContent(text="message 3")),
    ]
    
    # Create GetPromptResult
    prompt_result = GetPromptResult(
        id="test-prompt",
        name="test-prompt",
        description="Test prompt",
        messages=messages
    )
    
    # Apply the template
    result = await passthrough.apply_prompt_template(prompt_result, "test-prompt")
    
    # Assert that all message content is present in the result
    assert "message 1" in result
    assert "message 2" in result
    assert "message 3" in result


async def test_apply_prompt_concatenates_all_content():
    # Create a PassthroughLLM instance
    passthrough = PassthroughLLM(name="TestPassthrough")
    
    # Create multipart messages directly
    multipart_messages = [
        PromptMessageMultipart(
            role=Role.ASSISTANT,
            content=[TextContent(text="message 1")]
        ),
        PromptMessageMultipart(
            role=Role.USER,
            content=[TextContent(text="message 2")]
        ),
        PromptMessageMultipart(
            role=Role.ASSISTANT,
            content=[TextContent(text="message 3")]
        ),
    ]
    
    # Apply the prompt directly
    result = await passthrough.apply_prompt(multipart_messages)
    
    # Assert that all message content is present in the result
    assert "message 1" in result
    assert "message 2" in result
    assert "message 3" in result


if __name__ == "__main__":
    asyncio.run(test_apply_prompt_template_concatenates_all_content())