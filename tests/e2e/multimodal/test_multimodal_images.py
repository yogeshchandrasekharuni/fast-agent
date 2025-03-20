# integration_tests/mcp_agent/test_agent_with_image.py
import pytest
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import TextContent, BlobResourceContents, EmbeddedResource
from mcp.server.fastmcp import Image
import base64
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
    # Use the FastAgent instance from the fixture
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
                Path("tests/e2e/multimodal/image.png"),
            )
            response = await agent.send(prompt)

            assert "evalstate" in response
            return response


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
    # Use the FastAgent instance from the fixture
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
    """Test that the agent can process an image and respond appropriately."""
    # Use the FastAgent instance from the fixture
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
                Prompt.user(
                    "summarize this document - include the company that made it",
                    Path("tests/e2e/multimodal/sample.pdf"),
                )
            )

            # Send the prompt and get the response
            assert "llmindset.co.uk".lower() in response.lower()


def create_embedded_resource(filepath: str) -> EmbeddedResource:
    """
    Create an EmbeddedResource from a file.

    Args:
        filepath: Path to the file to embed

    Returns:
        An EmbeddedResource containing the file contents
    """
    path = Path(filepath)
    uri = f"file://{path.absolute()}"

    # try:
    #     # Try to read as text first
    #     text = path.read_text()
    #     resource = TextResourceContents(
    #         uri=uri,
    #         text=text,
    #         mimeType="text/plain"
    #     )
    # except UnicodeDecodeError:
    # If it fails, read as binary
    binary_data = path.read_bytes()
    b64_data = base64.b64encode(binary_data).decode("ascii")

    # Guess mime type from extension
    mime_type = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".csv": "text/csv",
        ".json": "application/json",
    }.get(path.suffix.lower(), "application/octet-stream")

    resource = BlobResourceContents(uri=uri, blob=b64_data, mimeType=mime_type)

    return EmbeddedResource(type="resource", resource=resource)
