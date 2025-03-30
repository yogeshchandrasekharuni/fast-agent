"""
Focused test for the 2-agent chain case to isolate potential issues.
"""

import pytest

from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.workflows.llm.augmented_llm_passthrough import FIXED_RESPONSE_INDICATOR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_agent_chain(fast_agent):
    """
    Test a chain with exactly 2 agents, which might be a special case causing issues.
    """
    fast = fast_agent

    # Define two agents and a chain connecting them
    @fast.agent(name="first", model="passthrough")
    @fast.agent(name="second", model="passthrough")
    @fast.chain(name="two_chain", sequence=["first", "second"])
    async def agent_function():
        async with fast.run() as agent:
            # Test the chain with a simple input
            input_message = "two-agent-test"
            result = await agent.two_chain.send(input_message)
            
            # With PassthroughLLM, we expect the message to be echoed multiple times
            print(f"DEBUG - Two agent chain result: '{result}'")
            
            # The result should contain the input message
            assert input_message in result
            
            # Manual chaining for comparison
            manual_first = await agent.first.send(input_message)
            manual_result = await agent.second.send(manual_first)
            print(f"DEBUG - Manual chaining result: '{manual_result}'")
            
            # The results should be similar
            assert result == manual_result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_agent_chain_fixed_response(fast_agent):
    """
    Test a 2-agent chain with fixed responses to ensure proper message passing.
    """
    fast = fast_agent

    @fast.agent(name="fixed_first")
    @fast.agent(name="fixed_second")
    @fast.chain(name="fixed_chain", sequence=["fixed_first", "fixed_second"])
    async def agent_function():
        async with fast.run() as agent:
            # Configure fixed responses for each agent
            await agent.fixed_first.send(f"{FIXED_RESPONSE_INDICATOR} first-response")
            await agent.fixed_second.send(f"{FIXED_RESPONSE_INDICATOR} second-response")
            
            # Test the chain
            result = await agent.fixed_chain.send("input-message")
            print(f"DEBUG - Fixed response chain result: '{result}'")
            
            # With fixed responses, we expect the final result to be the second agent's response
            assert "second-response" in result
            
            # For comparison, also test manual chaining
            manual_first = await agent.fixed_first.send("manual-input")
            assert "first-response" in manual_first
            
            manual_second = await agent.fixed_second.send(manual_first)
            assert "second-response" in manual_second

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_agent_chain_cumulative(fast_agent):
    """
    Test a 2-agent chain in cumulative mode.
    """
    fast = fast_agent

    @fast.agent(name="cum_first")
    @fast.agent(name="cum_second")
    @fast.chain(name="cumulative_chain", sequence=["cum_first", "cum_second"], cumulative=True)
    async def agent_function():
        async with fast.run() as agent:
            # Configure fixed responses
            await agent.cum_first.send(f"{FIXED_RESPONSE_INDICATOR} first-cum-response")
            await agent.cum_second.send(f"{FIXED_RESPONSE_INDICATOR} second-cum-response")
            
            # Test the chain with cumulative=True
            input_message = "cumulative-input"
            result = await agent.cumulative_chain.send(input_message)
            print(f"DEBUG - Cumulative chain result: '{result}'")
            
            # Verify XML tags and content
            assert f"<fastagent:request>{input_message}</fastagent:request>" in result
            assert "<fastagent:response agent='cum_first'>first-cum-response</fastagent:response>" in result
            assert "<fastagent:response agent='cum_second'>second-cum-response</fastagent:response>" in result

    await agent_function()