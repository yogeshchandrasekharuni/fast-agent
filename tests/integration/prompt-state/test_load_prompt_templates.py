import os
from pathlib import Path
from typing import TYPE_CHECKING, List

import pytest
from mcp.types import ImageContent

from mcp_agent.mcp.prompts.prompt_load import (
    load_prompt_multipart,
)

if TYPE_CHECKING:
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


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
            loaded: List[PromptMessageMultipart] = load_prompt_multipart(Path("conv1_simple.md"))
            assert 4 == len(loaded)
            assert "user" == loaded[0].role
            assert "assistant" == loaded[1].role
            
            # Use the "default" agent directly
            response = await agent.default.generate_x(loaded)
            assert "message 2" in response.first_text()

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
        async with fast.run():
            prompts: list[PromptMessageMultipart] = load_prompt_multipart(Path("conv2_attach.md"))

            assert 5 == len(prompts)
            assert "user" == prompts[0].role
            assert "text/css" == prompts[0].content[1].resource.mimeType  # type: ignore
            assert "f5f5f5" in prompts[0].content[1].resource.text  # type: ignore

            assert "assistant" == prompts[1].role
            assert "sharing" in prompts[1].content[0].text  # type: ignore

            assert 3 == len(prompts[2].content)
            assert isinstance(prompts[2].content[2], ImageContent)
            assert 12780 == len(prompts[2].content[2].data)

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_state_to_simple_text_file(fast_agent):
    """Check to see if we can save a conversation to a text file. This functionality
    is extremely simple, and does not support round-tripping. JSON support using MCP
    types will be added in a future release."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            # Delete the file if it exists before running the test
            if os.path.exists("./simple.txt"):
                os.remove("./simple.txt")
            await agent.send("hello")
            await agent.send("world")
            await agent.send("***SAVE_HISTORY simple.txt")

            prompts: list[PromptMessageMultipart] = load_prompt_multipart(Path("simple.txt"))
            assert 4 == len(prompts)
            assert "user" == prompts[0].role
            assert "assistant" == prompts[1].role

    await agent_function()
