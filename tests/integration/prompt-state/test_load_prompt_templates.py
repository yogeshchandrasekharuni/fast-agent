from typing import List
import pytest

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.prompts.prompt_template import PromptTemplateLoader


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_simple_conversation_from_file(fast_agent):
    """Make sure that we can load a simple multiturn conversation from a file."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            loaded: List[PromptMessageMultipart] = (
                PromptTemplateLoader()
                .load_from_file("conv1_simple.md")
                .to_multipart_messages()
            )
            assert 4 == len(loaded)
            # make sure that all messages, including assistant are returned by passthrough
            assert "message 2" in await agent.apply_prompt(loaded)

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_conversation_with_attachments(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            loaded: List[PromptMessageMultipart] = (
                PromptTemplateLoader()
                .load_from_file("conv2_attach.md")
                .to_multipart_messages()
            )
            assert 5 == len(loaded)
            # make sure that all messages, including assistant are returned by passthrough
            assert "message 2" in await agent.apply_prompt(loaded)

    await agent_function()
