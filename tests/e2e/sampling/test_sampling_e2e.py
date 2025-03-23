import pytest


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_sampling_output(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model="passthrough",  # only need a resource call
        servers=["sampling_resource"],
    )
    async def agent_function():
        async with fast.run() as agent:
            story = await agent.with_resource(
                "Here is a story",
                "sampling_resource",
                "resource://fast-agent/short-story/kittens",
            )

            assert len(story) > 100
            assert "kitten" in story

    await agent_function()
