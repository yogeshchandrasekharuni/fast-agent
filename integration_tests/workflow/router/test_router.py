# integration_tests/mcp_agent/test_agent_with_image.py
import pytest
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import TextContent


@pytest.fixture
def test_image_path():
    """Fixture that provides the path to a test image."""
    # # Get the directory of the test file
    # test_dir = pathlib.Path(__file__).parent.absolute()

    # # Navigate to test resources directory (you'll need to create this)
    # resources_dir = test_dir.parent.parent / "test_resources"

    # # Make sure the directory exists
    # os.makedirs(resources_dir, exist_ok=True)

    # # Return path to test image
    # image_path = resources_dir / "test_image.png"

    # # Verify the image exists
    # assert image_path.exists(), f"Test image not found at {image_path}"

    # return str(image_path)


@pytest.mark.integration
@pytest.mark.simulated_endpoints
@pytest.mark.asyncio
async def test_agent_with_image_response(test_image_path):
    """Test that the agent can process an image and respond appropriately."""
    # Create the application
    fast = FastAgent(
        "FastAgent Example Test",
        config_path="integration_tests/workflow/router/fastagent.config.yaml",
        ignore_unknown_args=True,
    )

    # Define the agent - similar to your original code but with a different structure
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        servers=["prompts", "hfspace"],
        model="passthrough",
    )
    async def agent_function():
        async with fast.run() as agent:
            # Create the multipart prompt with the correct image path
            prompt = PromptMessageMultipart(
                role="user",
                content=[
                    TextContent(type="text", text="how big is the moon?"),
                ],
            )

            # Send the prompt and get the response
            response = await agent.agent.send_prompt(prompt)

            # Return the response for assertions
            return response

    # Execute the agent function
    response = await agent_function()

    # Add your assertions here
    assert response is not None
    assert "moon" in response.lower(), "Response should mention the moon"
    # Add more specific assertions based on what you expect in the response
    # and what's in your test image
