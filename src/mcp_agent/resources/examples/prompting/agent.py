import asyncio
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import TextContent
from mcp.server.fastmcp.utilities.types import Image

# Create the application
fast = FastAgent("FastAgent Example")


# Define the agent
@fast.agent(
    "agent",
    instruction="You are a helpful AI Agent",
    servers=["prompts", "image","hfspace"],
    #    instruction="You are a helpful AI Agent", servers=["prompts","basic_memory"], model="haiku"
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent()
        foo: PromptMessageMultipart = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="how big is the moon?"),
                Image(path="image.png").to_image_content(),
                TextContent(type="text", text="and what is in that image?"),
            ],
        )
        await agent.agent.send_prompt(foo)


if __name__ == "__main__":
    asyncio.run(main())
