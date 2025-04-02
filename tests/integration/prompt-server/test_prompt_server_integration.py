from typing import TYPE_CHECKING, Dict, List

import pytest

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

if TYPE_CHECKING:
    from mcp.types import GetPromptResult, Prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_delimiters(fast_agent):
    """Single user message."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="test", servers=["prompts"])
    async def agent_function():
        async with fast.run() as agent:
            x: GetPromptResult = await agent["test"].get_prompt("simple", None)
            y: list[PromptMessageMultipart] = PromptMessageMultipart.to_multipart(x.messages)
            assert "simple, no delimiters" == y[0].first_text()
            assert "user" == y[0].role
            assert len(y) == 1

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_delimiters_with_variables(fast_agent):
    """Single user message, with substitutions."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="test", servers=["prompts"])
    async def agent_function():
        async with fast.run() as agent:
            x: GetPromptResult = await agent["test"].get_prompt(
                "simple_sub", {"product": "fast-agent", "company": "llmindset"}
            )
            y: list[PromptMessageMultipart] = PromptMessageMultipart.to_multipart(x.messages)
            assert "this is fast-agent by llmindset" == y[0].first_text()
            assert "user" == y[0].role
            assert len(y) == 1

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiturn(fast_agent):
    """Multipart Message."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="test", servers=["prompts"])
    async def agent_function():
        async with fast.run() as agent:
            x: GetPromptResult = await agent["test"].get_prompt("multi", None)
            y: list[PromptMessageMultipart] = PromptMessageMultipart.to_multipart(x.messages)
            assert "good morning" == y[0].first_text()
            assert "user" == y[0].role
            assert "how may i help you?" == y[1].first_text()
            assert "assistant" == y[1].role
            assert len(y) == 2

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiturn_with_subsitition(fast_agent):
    """Multipart Message, with substitutions."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="test", servers=["prompts"])
    async def agent_function():
        async with fast.run() as agent:
            x: GetPromptResult = await agent["test"].get_prompt(
                "multi_sub", {"user_name": "evalstate", "assistant_name": "HAL9000"}
            )
            y: list[PromptMessageMultipart] = PromptMessageMultipart.to_multipart(x.messages)
            assert "hello, my name is evalstate" == y[0].first_text()
            assert "user" == y[0].role
            assert "nice to meet you. i am HAL9000" == y[1].first_text()
            assert "assistant" == y[1].role
            assert len(y) == 2

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_interface_returns_prompts_list(fast_agent):
    """Multipart Message, with substitutions."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="test", servers=["prompts"])
    async def agent_function():
        async with fast.run() as agent:
            prompts: Dict[str, List[Prompt]] = await agent.test.list_prompts()
            assert 4 == len(prompts["prompts"])

    await agent_function()
