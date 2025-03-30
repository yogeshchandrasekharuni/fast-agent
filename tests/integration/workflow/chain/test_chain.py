import pytest

from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.workflows.llm.augmented_llm_passthrough import FIXED_RESPONSE_INDICATOR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_chain(fast_agent):
    """Test a simple chain in non-cumulative mode (default)"""
    fast = fast_agent

    # Define the agent
    @fast.agent(name="begin", model="passthrough")
    @fast.agent(name="step1", model="passthrough")
    @fast.agent(name="finish", model="passthrough")
    @fast.chain(name="chain", sequence=["begin", "step1", "finish"])
    async def agent_function():
        async with fast.run() as agent:
            result = await agent.chain.send("foo")
            # Print for debugging
            print(f"DEBUG - Actual result: '{result}'")
            # Assert exact match to the actual behavior - with the fix, we only pass
            # the previous agent's response to the next agent, not the original message
            assert "foo" == result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cumulative_chain(fast_agent):
    """Test cumulative chain mode with XML tags for request and responses"""
    fast = fast_agent

    @fast.agent(name="begin")
    @fast.agent(name="step1")
    @fast.agent(name="finish")
    @fast.chain(name="chain", sequence=["begin", "step1", "finish"], cumulative=True)
    async def agent_function():
        async with fast.run() as agent:
            # Configure the agents to return fixed responses
            await agent.begin.send(f"{FIXED_RESPONSE_INDICATOR} begin-response")
            await agent.step1.send(f"{FIXED_RESPONSE_INDICATOR} step1-response")
            await agent.finish.send(f"{FIXED_RESPONSE_INDICATOR} finish-response")

            initial_prompt = "initial-prompt"
            result = await agent.chain.send(initial_prompt)

            # Check for original request tag
            assert f"<fastagent:request>{initial_prompt}</fastagent:request>" in result

            # Check for agent response tags
            assert "<fastagent:response agent='begin'>begin-response</fastagent:response>" in result
            assert "<fastagent:response agent='step1'>step1-response</fastagent:response>" in result
            assert (
                "<fastagent:response agent='finish'>finish-response</fastagent:response>" in result
            )

            # Verify correct number of tags
            assert result.count("<fastagent:request>") == 1
            assert result.count("<fastagent:response") == 3

            # Verify formatting with newlines
            lines = result.strip().split("\n\n")
            assert len(lines) == 4  # Request + 3 responses

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chain_functionality(fast_agent):
    """Test that a chain correctly connects agents together"""
    # The goal of this test is to verify that a chain properly
    # connects agents together and passes messages through them

    fast = fast_agent

    # Use passthrough agents for simplicity and predictability
    # Create two separate chains: one normal and one cumulative
    @fast.agent(name="echo1", model="passthrough")
    @fast.agent(name="echo2", model="passthrough")
    @fast.agent(name="echo3", model="passthrough")
    @fast.chain(name="echo_chain", sequence=["echo1", "echo2", "echo3"])
    @fast.chain(name="cumulative_chain", sequence=["echo1", "echo2", "echo3"], cumulative=True)
    async def agent_function():
        async with fast.run() as agent:
            # Test non-cumulative chain
            # With passthrough LLMs in a chain, the message will be echoed
            # through each agent, resulting in the message being repeated
            input_message = "test message"
            result = await agent.echo_chain.send(input_message)

            # With three echo agents in the chain, the non-cumulative result
            # should be the echo from the third agent, which will include all echoes
            assert input_message in result

            # Test cumulative chain separately
            # This chain was created with cumulative=True
            cumulative_input = "cumulative message"
            cumulative_result = await agent.cumulative_chain.send(cumulative_input)

            # Verify both format and content
            assert "<fastagent:request>" in cumulative_result
            assert "<fastagent:response agent='echo1'>" in cumulative_result
            assert "<fastagent:response agent='echo2'>" in cumulative_result
            assert "<fastagent:response agent='echo3'>" in cumulative_result
            assert cumulative_input in cumulative_result

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
