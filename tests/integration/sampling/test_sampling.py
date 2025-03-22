import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sampling_feature(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="foo", instruction="bar", servers=["sampling_test"])
    async def agent_function():
        async with fast.run() as agent:
            result = await agent("***CALL_TOOL SAMPLING {}")
            assert "this is a result of sampling" in result

    await agent_function()
