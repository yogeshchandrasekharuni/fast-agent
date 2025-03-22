from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("MCP Root Tester")


@mcp.tool()
async def show_roots(ctx: Context) -> str:
    ctx.error("LIST ROOTS CALLED")
    try:
        ctx.error("FUCK")
        return await ctx.session.list_roots()
    except Exception as e:
        return f"that didnt#t work. {e}"


if __name__ == "__main__":
    mcp.run()
