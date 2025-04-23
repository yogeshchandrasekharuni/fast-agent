from typing import TYPE_CHECKING

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_agent_card_and_tools(fast_agent):
    fast = fast_agent

    @fast.agent(name="test", instruction="here are you instructions", servers=["hyphen-test"])
    async def agent_function():
        async with fast.run() as app:
            result = await app.test.send('***CALL_TOOL check_weather {"location": "New York"}')
            assert "sunny" in result

    await agent_function()
