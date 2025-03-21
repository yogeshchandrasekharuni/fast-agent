# integration_tests/mcp_agent/test_agent_with_image.py
import pytest
from mcp_agent.core.prompt import Prompt
from pathlib import Path


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini",  # OpenAI model
        "haiku35",  # Anthropic model
    ],
)
async def test_agent_with_image_prompt(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            prompt = Prompt.user(
                "what is the user name contained in this image?",
                Path("image.png"),
            )
            response = await agent.send(prompt)

            assert "evalstate" in response
            return response

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini",  # OpenAI model
        "haiku35",  # Anthropic model
    ],
)
async def test_agent_with_mcp_image(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send the prompt and get the response

            response = await agent.send(
                "Use the image fetch tool to get the sample image and tell me what user name contained in this image?"
            )
            assert "evalstate" in response
            # Return the response for assertions
            return response

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini",  # OpenAI model
        "haiku35",  # Anthropic model
    ],
)
async def test_agent_with_mcp_pdf(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send the prompt and get the response

            response = await agent.send(
                "Can you summarise the sample PDF, make sure it includes the company name in the summary"
            )
            assert "llmindset" in response
            # Return the response for assertions
            return response

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini",  # OpenAI model
        "haiku35",  # Anthropic model
    ],
)
async def test_agent_with_pdf_prompt(fast_agent, model_name):
    """Test that the agent can process a PDF document and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response = await agent.send(
                message=Prompt.user(
                    "summarize this document - include the company that made it",
                    Path("sample.pdf"),
                )
            )

            # Send the prompt and get the response
            assert "llmindset".lower() in response.lower()

    # Execute the agent function
    await agent_function()
