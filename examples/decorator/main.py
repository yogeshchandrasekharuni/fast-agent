"""
Example MCP Agent application showing simplified agent access.
"""

import asyncio
from mcp_agent.core.decorator_app import MCPAgentDecorator

# Create the application
agent_app = MCPAgentDecorator("root-test")


# Define the agent
@agent_app.agent(
    name="basic_agent",
    instruction="A simple agent that helps with basic tasks. Request Human Input whenever needed.",
    servers=["mcp_root"],
)
async def main():
    # Use the app's context manager
    async with agent_app.run() as agent:
        # result = await agent(
        #     "read the persona descriptions from 'persona.md' and rate their suitability for a junior PM job",
        # )
        # print(result)

        # result = await agent(
        #     "can you write a letter to each candidate, informing them of the outcome of their application",
        # )
        # print(result)

        #  await agent("print the next number in the sequence")
        await agent.prompt()


if __name__ == "__main__":
    asyncio.run(main())
