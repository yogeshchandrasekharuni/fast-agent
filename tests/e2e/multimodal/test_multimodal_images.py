# integration_tests/mcp_agent/test_agent_with_image.py
import pytest
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import TextContent
from mcp.server.fastmcp import Image


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
async def test_agent_with_image_prompt(model_name):
    """Test that the agent can process an image and respond appropriately."""
    # Create the application
    fast = FastAgent(
        "Image Tests",
        config_path="tests/e2e/multimodal/fastagent.config.yaml",
        ignore_unknown_args=True,
    )

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Create the multipart prompt with the correct image path
            prompt = PromptMessageMultipart(
                role="user",
                content=[
                    TextContent(
                        type="text",
                        text="what is the user name contained in this image?",
                    ),
                    Image(path="tests/e2e/multimodal/image.png").to_image_content(),
                ],
            )

            # Send the prompt and get the response
            response = await agent.agent.send_prompt(prompt)
            assert "evalstate" in response
            # Return the response for assertions
            return response

    # Execute the agent function
    response = await agent_function()

    # Add your assertions here
    assert response is not None


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
async def test_agent_with_mcp_image(model_name):
    """Test that the agent can process an image and respond appropriately."""
    # Create the application
    fast = FastAgent(
        "Image Tests",
        config_path="tests/e2e/multimodal/fastagent.config.yaml",
        ignore_unknown_args=True,
    )

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Create the multipart prompt with the correct image path
            # prompt = PromptMessageMultipart(
            #     role="user",
            #     content=[
            #         TextContent(
            #             type="text",
            #             text="Use the image fetch tool and what is the user name contained in this image?",
            #         ),
            #         Image(path="tests/e2e/multimodal/image.png").to_image_content(),
            #     ],
            # )

            # Send the prompt and get the response
            response = await agent.agent.send_prompt(
                "Use the image fetch tool and tell me what user name contained in this image?"
            )
            assert "evalstate" in response
            # Return the response for assertions
            return response

    # Execute the agent function
    response = await agent_function()

    # Add your assertions here
    assert response is not None
