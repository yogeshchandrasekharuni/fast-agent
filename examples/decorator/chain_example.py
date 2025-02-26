import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Chain Example")


@fast.agent(
    "url_fetcher",
    instruction="Given a URL, provide a complete and comprehensive summary",
    servers=["fetch"],
)
@fast.agent(
    "social_media",
    instruction="""
    Write a 280 character social media post for any given text. 
    Respond only with the post, never use hashtags.
    """,
)
@fast.chain(
    name="social-poster",
    sequence=["url_fetcher", "social_media"],
)
async def main():
    async with fast.run() as agent:
        # Demonstrate chaining with the new chain decorator
        print("\n=== Using Chain Workflow ===")
        result = await agent.send("social-poster", "http://llmindset.co.uk/resources/mcp-hfspace/")
        print(f"\nChain result: {result}")
        
        # Demonstrate the old way of chaining manually
        print("\n=== Manual Chaining (Original approach) ===")
        manual_result = await agent.social_media(
            await agent.url_fetcher("http://llmindset.co.uk/resources/mcp-hfspace/")
        )
        print(f"\nManual chaining result: {manual_result}")

        # Uncomment to interact with agents
        # await agent()


if __name__ == "__main__":
    asyncio.run(main())