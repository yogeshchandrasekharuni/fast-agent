import pytest


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_sampling_output_anthropic(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model="passthrough",  # only need a resource call
        servers=["sampling_resource_anthropic"],
    )
    async def agent_function():
        async with fast.run() as agent:
            story = await agent.with_resource(
                "Here is a story",
                "sampling_resource_anthropic",
                "resource://fast-agent/short-story/kittens",
            )

            assert len(story) > 300
            assert "kitten" in story
            assert "error" not in story.lower()

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_sampling_output_gpt(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model="passthrough",  # only need a resource call
        servers=["sampling_resource_openai"],
    )
    async def agent_function():
        async with fast.run() as agent:
            story = await agent.with_resource(
                "Here is a story",
                "sampling_resource_openai",
                "resource://fast-agent/short-story/kittens",
            )

            assert len(story) > 300
            assert "kitten" in story
            assert "error" not in story.lower()

    await agent_function()
