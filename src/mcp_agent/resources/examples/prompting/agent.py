import asyncio
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import TextContent, EmbeddedResource
from mcp.server.fastmcp.utilities.types import Image

# Create the application
fast = FastAgent("FastAgent Example")


# Define the agent
@fast.agent(
    "agent",
    instruction="You are a helpful AI Agent",
    servers=["prompts", "fetch"],  # , "imgetage", "hfspace"],
    #    model="gpt-4o",
    #    instruction="You are a helpful AI Agent", servers=["prompts","basic_memory"], model="haiku"
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent()
    foo: PromptMessageMultipart = PromptMessageMultipart(
        role="user",
        content=[
            TextContent(type="text", text="summarize this document"),
            create_embedded_resource("sample.pdf"),
            Image(path="image.png").to_image_content(),
            TextContent(type="text", text="and what is in that image?"),
        ],
    )


#       await agent.agent.send_prompt(foo)
#

import base64
from pathlib import Path
from mcp.types import EmbeddedResource, TextResourceContents, BlobResourceContents


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


if __name__ == "__main__":
    asyncio.run(main())
