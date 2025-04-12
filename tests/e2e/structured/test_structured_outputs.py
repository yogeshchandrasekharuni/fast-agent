# integration_tests/mcp_agent/test_agent_with_image.py

import pytest
from pydantic import BaseModel

from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "generic.qwen2.5:latest",
        "generic.llama3.2:latest",
        "haiku",
        "sonnet",
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini.low",
        "openrouter.google/gemini-2.0-flash-001",
    ],
)
async def test_structured_output_with_no_response_format(fast_agent, model_name):
    """Test that the agent can generate structured response with response_format_specified."""
    fast = fast_agent

    @fast.agent(
        "chat",
        instruction="You are a helpful assistant.",
        model=model_name,
    )
    async def create_structured():
        async with fast.run() as agent:
            thinking, response = await agent.chat.structured(
                [Prompt.user("Let's talk about guitars.")],
                model=FormattedResponse,
            )
            assert thinking is not None

    await create_structured()


response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "formatted_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "string",
                    "description": "Your reflection on the conversation that is not seen by the user.",
                },
                "message": {
                    "type": "string",
                    "description": "Your message to the user.",
                },
            },
            "required": ["thinking", "message"],
            "additionalProperties": False,
        },
    },
}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "generic.qwen2.5:latest",
        "generic.llama3.2:latest",
        "haiku",
        "sonnet",
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini.low",
        "openrouter.google/gemini-2.0-flash-001",
    ],
)
async def test_structured_output_with_response_format_spec(fast_agent, model_name):
    """Test that the agent can generate structured response with response_format_specified."""
    fast = fast_agent

    @fast.agent(
        "chat",
        instruction="You are a helpful assistant.",
        model=model_name,
    )
    async def create_structured():
        async with fast.run() as agent:
            thinking, response = await agent.chat.structured(
                [Prompt.user("Let's talk about guitars.")],
                model=FormattedResponse,
                request_params=RequestParams(response_format=response_format),
            )
            assert thinking is not None

    await create_structured()


class FormattedResponse(BaseModel):
    thinking: str
    message: str
