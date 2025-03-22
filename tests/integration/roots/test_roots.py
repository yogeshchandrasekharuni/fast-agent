import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_roots_returned(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="foo", instruction="bar", servers=["roots_test"])
    async def agent_function():
        async with fast.run() as agent:
            assert "tsafdfest" in await agent("***CALL_TOOL roots_test-show_roots {}")

    await agent_function()
