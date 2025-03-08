# example_mcp_server.py
import asyncio

from mcp_agent.core.fastagent import FastAgent

# Define your agents as normal
fa = FastAgent("My Application")


@fa.agent("analyst", "hello, world", servers=["fetch"])

# Run the application with MCP server
async def main():
    await fa.run_with_mcp_server(
        transport="sse",  # Use "sse" for web server, "stdio" for command line
        port=8000,
        server_name="MyAgents",
        server_description="MCP Server exposing analyst and researcher agents",
    )


if __name__ == "__main__":
    asyncio.run(main())
