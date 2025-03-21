import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("FastAgent Example")


# Define the agent
@fast.agent(
    "agent",
    instruction="You are a helpful AI Agent",
    servers=["prompts"],  # , "imgetage", "hfspace"],
    #    model="gpt-4o",
    #    instruction="You are a helpful AI Agent", servers=["prompts","basic_memory"], model="haiku"
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent()


if __name__ == "__main__":
    asyncio.run(main())
