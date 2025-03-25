import pytest
from mcp.types import GetPromptResult, PromptMessage, TextContent

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.workflows.llm.augmented_llm_passthrough import PassthroughLLM


@pytest.fixture
def passthrough_llm():
    """Create a PassthroughLLM instance for testing."""
    return PassthroughLLM(name="TestPassthrough")


@pytest.fixture
def message_texts():
    """Define message texts to use in tests."""
    return ["message 1", "message 2", "message 3"]


@pytest.fixture
def prompt_messages(message_texts):
    """Create standard PromptMessage objects for testing."""
    return [
        PromptMessage(role="assistant", content=TextContent(type="text", text=message_texts[0])),
        PromptMessage(role="user", content=TextContent(type="text", text=message_texts[1])),
        PromptMessage(role="assistant", content=TextContent(type="text", text=message_texts[2])),
    ]


@pytest.fixture
def multipart_messages(message_texts):
    """Create PromptMessageMultipart objects for testing."""
    return [
        PromptMessageMultipart(role="assistant", content=[TextContent(type="text", text=message_texts[0])]),
        PromptMessageMultipart(role="user", content=[TextContent(type="text", text=message_texts[1])]),
        PromptMessageMultipart(role="assistant", content=[TextContent(type="text", text=message_texts[2])]),
    ]


@pytest.fixture
def prompt_result(prompt_messages):
    """Create a GetPromptResult object for testing."""
    return GetPromptResult(
        id="test-prompt",
        name="test-prompt",
        description="Test prompt",
        messages=prompt_messages,
    )


@pytest.mark.asyncio
async def test_apply_prompt_template_concatenates_all_content(passthrough_llm, prompt_result, message_texts):
    # Apply the template
    result = await passthrough_llm.apply_prompt_template(prompt_result, "test-prompt")

    # Assert that all message content is present in the result
    for message in message_texts:
        assert message in result


@pytest.mark.asyncio
async def test_apply_prompt_concatenates_all_content(passthrough_llm, multipart_messages, message_texts):
    # Apply the prompt directly
    result = await passthrough_llm.apply_prompt(multipart_messages)

    # Assert that all message content is present in the result
    for message in message_texts:
        assert message in result
