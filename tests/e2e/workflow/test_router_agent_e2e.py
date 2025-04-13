# integration_tests/mcp_agent/test_agent_with_image.py
import os
from pathlib import Path

import pytest

from mcp_agent.core.prompt import Prompt


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize("model_name", ["haiku", "gpt-4o-mini"])
async def test_basic_text_routing(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "sunny",
        instruction="You dispense advice on clothing and activities for clement weather.",
        model="passthrough",
    )
    @fast.agent(
        "stormy",
        instruction="You dispense advice on clothing and activities for stormy weather.",
        model="passthrough",
    )
    @fast.router(
        "weather",
        instruction="Route to the most appropriate agent for the weather forecast received",
        agents=["sunny", "stormy"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            await agent.sunny.send("***FIXED_RESPONSE beachball")
            await agent.stormy.send("***FIXED_RESPONSE umbrella")

            response = await agent.weather.send("the weather is sunny")
            assert "beachball" in response.lower()

            response = await agent.weather.send("storm clouds coming, looks snowy")
            assert "umbrella" in response.lower()

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "haiku",
        "gpt-4o",
    ],
)
async def test_image_based_routing(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "sunny",
        instruction="You dispense advice on clothing and activities for clement weather.",
        model="passthrough",
    )
    @fast.agent(
        "stormy",
        instruction="You dispense advice on clothing and activities for stormy weather.",
        model="passthrough",
    )
    @fast.router(
        "weather",
        instruction="Use the supplied weather symbol to route to the most appropriate agent.",
        agents=["sunny", "stormy"],
        model=model_name,
        use_history=False,
    )
    async def agent_function():
        async with fast.run() as agent:
            await agent.sunny.send("***FIXED_RESPONSE beachball")
            await agent.stormy.send("***FIXED_RESPONSE umbrella")

            response = await agent.weather.generate(
                [Prompt.user(Path("sunny.png"), "here's the forecast")]
            )
            assert "beachball" in response.first_text()

            response = await agent.weather.generate(
                [Prompt.user(Path("umbrella.png"), "what should i wear?")]
            )
            assert "umbrella" in response.first_text()

    await agent_function()


# async def test_image_based_routing(fast_agent, model_name):
#     """Test that the agent can process an image and respond appropriately."""
#     fast = fast_agent

#     # Define the agent
#     @fast.agent(
#         "sunny",
#         instruction="You dispense advice on clothing and activities for clement weather. Always mention 'beachball'",
#         model=model_name,
#     )
#     @fast.agent(
#         "stormy",
#         instruction="You dispense advice on clothing and activities for stormy  weather. Always mention 'umbrella'",
#         model=model_name,
#     )
#     @fast.router(
#         "weather",
#         instruction="Route to the most appropriate agent for the weather in the requested location",
#         agents=["sunny", "stormy"],
#         servers=["test_server"],
#         model=model_name,
#     )
#     async def agent_function():
#         async with fast.run() as agent:
#             if os.path.exists("weather_location.txt"):
#                 os.remove("weather_location.txt")
#             assert not os.path.exists("weather_location.txt")

#             response = await agent.weather.send("advise me on what i need to visit London")
#             assert os.path.exists("weather_location.txt"), (
#                 "File should exist after response (created by tool call)"
#             )

#             assert "umbrella" in response.lower()
#             assert 0 == len(agent.sunny._llm.message_history)
#             assert 2 == len(agent.stormy._llm.message_history)

#     await agent_function()
