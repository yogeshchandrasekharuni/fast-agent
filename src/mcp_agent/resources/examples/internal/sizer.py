from mcp_agent.core.fastagent import FastAgent

fast = FastAgent("Sizer Prompt Test")


@fast.agent("sizer", "given an object return the size", servers=["prompt"])
async def main():
    async with fast.run() as agent:
        await agent.mcp_prompt("sizing_prompt")
        await agent("What is the size of the moon?")
        await agent("What is the size of the Earth?")
        await agent("What is the size of the Sun?")


if __name__ == "__main__":
    asyncio.run(main())
