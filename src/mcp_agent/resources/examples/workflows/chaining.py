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
@fast.chain(
    name="post_writer",
    sequence=["url_fetcher", "social_media"],
)
async def main():
    async with fast.run() as agent:
        # using chain workflow
        await agent.post_writer.prompt()

        # calling directly
        # await agent.url_fetcher("http://llmindset.co.uk/resources/mcp-hfspace/")
        # await agent.social_media(
        #     await agent.url_fetcher("http://llmindset.co.uk/resources/mcp-hfspace/")
        # )

        # agents can also be accessed like dictionaries:
        # awwait agent["post_writer"].prompt()


# alternative syntax for above is result = agent["post_writer"].send(message)
# alternative syntax for above is result = agent["post_writer"].prompt()


if __name__ == "__main__":
    asyncio.run(main())
