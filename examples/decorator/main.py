"""
Example MCP Agent application showing simplified agent access.
"""

import asyncio
from mcp_agent.core.decorator_app import MCPAgentDecorator

# Create the application
agent_app = MCPAgentDecorator("Interactive Agent Example")


# Define the agent
@agent_app.agent(
    name="basic_agent",
    instruction="A simple agent that helps with basic tasks. Request Human Input whenever needed.",
    servers=["mcp_root"],
)
async def main():
    # Use the app's context manager
    async with agent_app.run() as agent:
        await agent("print the next number in the sequence")
        await agent.prompt(default="STOP")


if __name__ == "__main__":
    asyncio.run(main())
