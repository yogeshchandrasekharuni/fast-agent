from pathlib import Path

import pytest

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.prompts.prompt_load import load_prompt_multipart
from mcp_agent.workflows.llm.augmented_llm_passthrough import FIXED_RESPONSE_INDICATOR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_router_functionality(fast_agent):
    """Check that the router routes"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="target1")
    @fast.agent(name="target2")
    @fast.router(name="router", agents=["target1", "target2"], model="playback")
    async def agent_function():
        async with fast.run() as agent:
            await agent.target1.send(f"{FIXED_RESPONSE_INDICATOR} target1-result")
            await agent.target2.send(f"{FIXED_RESPONSE_INDICATOR} target2")
            router_setup: list[PromptMessageMultipart] = load_prompt_multipart(
                Path("router_script.txt")
            )
            setup: PromptMessageMultipart = await agent.router.generate_x(router_setup)
            assert "LOADED" in setup.first_text()
            result: str = await agent.router.send("some routing")
            assert "target1-result" in result

    await agent_function()
