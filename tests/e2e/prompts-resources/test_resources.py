# integration_tests/mcp_agent/test_agent_with_image.py
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini",  # OpenAI model
    ],
)
async def test_using_resource(fast_agent, model_name):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
        servers=["prompt_server"],
    )
    async def agent_function():
        async with fast.run() as agent:
            assert "fast-agent" in await agent.agent.with_resource(
                "Summarise this PDF please, be sure to include the product name",
                "prompt_server",
                "resource://fast-agent/sample.pdf",
            )

    await agent_function()
