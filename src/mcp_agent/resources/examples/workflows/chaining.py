import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Agent Chaining")


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
async def main():
    async with fast.run() as agent:
        await agent.social_media(
            await agent.url_fetcher("http://llmindset.co.uk/resources/mcp-hfspace/")
        )


#      uncomment below to interact with agents
#      await agent()

# alternative syntax for above is agent["social_media"].send(message)


if __name__ == "__main__":
    asyncio.run(main())
