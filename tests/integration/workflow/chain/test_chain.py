import pytest

from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.workflows.llm.augmented_llm_passthrough import FIXED_RESPONSE_INDICATOR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_chain(fast_agent):
    """Cumulative Chain"""
    # Not comprehensive
    fast = fast_agent

    # Define the agent
    @fast.agent(name="begin")
    @fast.agent(name="step1")
    @fast.agent(name="finish")
    @fast.chain(name="chain", sequence=["begin", "step1", "finish"])
    async def agent_function():
        async with fast.run() as agent:
            result = await agent.chain.send("foo")
            assert "foo\nfoo" == result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cumulative_chain(fast_agent):
    """Cumulative Chain"""
    fast = fast_agent

    @fast.agent(name="begin")
    @fast.agent(name="step1")
    @fast.agent(name="finish")
    @fast.chain(name="chain", sequence=["begin", "step1", "finish"], cumulative=True)
    async def agent_function():
        async with fast.run() as agent:
            await agent.begin.send(f"{FIXED_RESPONSE_INDICATOR} begin-response")
            await agent.step1.send(f"{FIXED_RESPONSE_INDICATOR} step1-response")
            await agent.finish.send(f"{FIXED_RESPONSE_INDICATOR} finish-response")

            result = await agent.chain.send("initial-prompt")

            assert "<fastagent:response agent='begin'>begin-response</fastagent:response>" in result
            assert "<fastagent:response agent='step1'>step1-response</fastagent:response>" in result
            assert (
                "<fastagent:response agent='finish'>finish-response</fastagent:response>" in result
            )

            assert result.count("<fastagent:response") == 3

            lines = result.strip().split("\n\n")
            assert len(lines) == 3

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disallows_empty_sequence(fast_agent):
    fast = fast_agent

    # Define the agent
    with pytest.raises(AgentConfigError):

        @fast.chain(name="chain", sequence=[], cumulative=True)
        async def agent_function():
            async with fast.run():
                assert True
