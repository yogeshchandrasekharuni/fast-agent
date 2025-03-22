from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import SamplingMessage, TextContent

mcp = FastMCP("MCP Root Tester", log_level="DEBUG")


@mcp.tool()
async def sample(ctx: Context) -> str:
    response = await ctx.session.create_message(
        max_tokens=1024,
        messages=[
            SamplingMessage(
                role="user", content=TextContent(type="text", text="hello, world")
            )
        ],
    )
    return str(response)


if __name__ == "__main__":
    mcp.run()
