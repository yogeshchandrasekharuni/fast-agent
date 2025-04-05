import pytest

from mcp_agent.core.prompt import Prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chaining_routers(fast_agent):
    """Check that the router routes"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    @fast.agent(name="target1")
    @fast.agent(name="target2")
    @fast.agent(name="target3")
    @fast.router(name="router1", agents=["target1", "target2"])
    @fast.chain(name="chain", sequence=["router1", "target3"])
    async def agent_function():
        async with fast.run() as agent:
            await agent.router1._llm.generate(
                [
                    Prompt.user(
                        """***FIXED_RESPONSE 
                        {"agent": "target2",
                        "confidence": "high",
                        "reasoning": "Request is asking for weather information"}"""
                    )
                ]
            )
            assert "github.com/varaarul" in await agent.chain.send("github.com/varaarul")

    await agent_function()
