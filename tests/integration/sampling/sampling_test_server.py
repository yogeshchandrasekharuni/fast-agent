from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import SamplingMessage, TextContent, CallToolResult

mcp = FastMCP("MCP Root Tester", log_level="DEBUG")


@mcp.tool()
async def sample(ctx: Context, to_sample: str | None = "hello, world") -> CallToolResult:
    try:
        # Send message to LLM but we don't use the response
        await ctx.session.create_message(
            max_tokens=1024,
            messages=[
                SamplingMessage(
                    role="user", content=TextContent(type="text", text=to_sample)
                )
            ],
        )
        return CallToolResult(
            content=[TextContent(type="text", text=to_sample)]
        )
    except Exception as e:
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"Error: {str(e)}")]
        )


if __name__ == "__main__":
    mcp.run()
