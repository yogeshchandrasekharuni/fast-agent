import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
agent_app = FastAgent("FastAgent Example")
# Uncomment the below to disable human input callback tool
# agent_app.app._human_input_callback = None


# Define the agent
@agent_app.agent(
    instruction="You are a helpful AI Agent",
    servers=[],
)
async def main():
    # use the --model= command line switch to specify model
    async with agent_app.run() as agent:
        await agent()


if __name__ == "__main__":
    asyncio.run(main())
