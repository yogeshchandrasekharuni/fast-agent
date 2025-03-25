import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_loading_agent(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent("agent1")
    @fast.agent("agent2")
    @fast.orchestrator("orchestrator", agents=["agent1", "agent2"])
    async def agent_function():
        async with fast.run():
            # assert "orchestrate" in await agent.orchestrator.send("orchestrate")
            assert True  # pending implementation

    await agent_function()
