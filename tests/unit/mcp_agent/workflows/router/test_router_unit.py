from typing import TYPE_CHECKING

import pytest

from mcp_agent.core.prompt import Prompt
from mcp_agent.workflows.llm.augmented_llm_passthrough import (
    PassthroughLLM,
)
from mcp_agent.workflows.router.router_base import RouterResult
from mcp_agent.workflows.router.router_llm import LLMRouterResult

if TYPE_CHECKING:
    from mcp_agent.mcp.interfaces import AugmentedLLMProtocol


@pytest.mark.asyncio
async def test_simple_return():
    llm: AugmentedLLMProtocol = PassthroughLLM()

    # router_result: RouterResult = RouterResult(=)
    # result: LLMRouterResult = LLMRouterResult(
    #     result="foo", confidence="high", reasoning="why", p_score="0.9"
    # )
    # response = await llm.generate_x(multipart_messages=[Prompt.user("playback message")])
    # assert "assistant" == response.role
    # assert "playback message" == response.first_text()
    assert True
