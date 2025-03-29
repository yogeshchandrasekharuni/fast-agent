import pytest

from mcp_agent.core.prompt import Prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_api_with_simple_prompt(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
    )
    async def agent_function():
        async with fast.run() as agent:
            assert "test1" in await agent.agent1.send("test1")
            assert "test2" in await agent["agent1"].send("test2")
            assert "test3" in await agent.send("test3")
            assert "test4" in await agent("test4")
            assert "test5" in await agent.send("test5", "agent1")
            assert "test6" in await agent("test6", "agent1")

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_api_with_prompt_messages(fast_agent):
    """Test that the agent can process a multipart prompts using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
    )
    async def agent_function():
        async with fast.run() as agent:
            assert "test1" in await agent.agent1.send(Prompt.user("test1"))

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_api_with_basic_playback(fast_agent):
    """Test that the agent can process a multipart prompts using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
        model="playback",
        servers=["prompts"],
    )
    async def agent_function():
        async with fast.run() as agent:
            await agent.agent1.apply_prompt("playback")
            assert "assistant1" in await agent.agent1.send("ignored")

    await agent_function()
