import pytest
from mcp.types import GetPromptResult

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_run(fast_agent):
    """Single user message."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="fan_out_1")
    @fast.agent(name="fan_out_2")
    @fast.agent(name="fan_in")
    @fast.parallel(name="parallel", fan_out=["fan_out_1", "fan_out_2"], fan_in="fan_in")
    async def agent_function():
        async with fast.run() as agent:
            expected: str = """The following request was sent to the agents:

<fastagent:request>
foo
</fastagent:request>

<fastagent:response agent="fan_out_1">
foo
</fastagent:response>

<fastagent:response agent="fan_out_2">
foo
</fastagent:response>"""
            assert expected == await agent.parallel.send("foo")

    await agent_function()
