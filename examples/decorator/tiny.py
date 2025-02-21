"""
Example MCP Agent application showing simplified agent access.
"""

import asyncio
from mcp_agent.core.decorator_app import FastAgent

# Create the application
agent_app = FastAgent("Interactive Agent Example")


# Define the agent
@agent_app.agent()
async def main():
    # use the --model= command line switch to specify model
    async with agent_app.run() as agent:
        await agent()  # Added a specific message instead of empty string


if __name__ == "__main__":
    asyncio.run(main())
