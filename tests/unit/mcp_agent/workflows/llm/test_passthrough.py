import pytest
from mcp.types import GetPromptResult, PromptMessage, TextContent

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.workflows.llm.augmented_llm_passthrough import PassthroughLLM


@pytest.fixture
def passthrough_llm():
    """Create a PassthroughLLM instance for testing."""
    return PassthroughLLM()


@pytest.mark.asyncio
async def test_apply_prompt_template_concatenates_all_content(
    passthrough_llm, prompt_result, message_texts
):
    # Apply the template
    result = await passthrough_llm.apply_prompt_template(prompt_result, "test-prompt")

    # Assert that all message content is present in the result
    for message in message_texts:
        assert message in result


@pytest.mark.asyncio
async def test_apply_prompt_concatenates_all_content(
    passthrough_llm, multipart_messages, message_texts
):
    # Apply the prompt directly
    result = await passthrough_llm.apply_prompt(multipart_messages)

    # Assert that all message content is present in the result
    for message in message_texts:
        assert message in result
