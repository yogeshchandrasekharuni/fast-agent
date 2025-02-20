import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent, AgentConfig
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM  # noqa: F401
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM  # noqa: F401
from rich import print

app = MCPApp(name="mcp_basic_agent")


async def example_usage():
    async with app.run():
        finder_config = AgentConfig(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            servers=["fetch", "filesystem"],
        )

        finder_agent = Agent(config=finder_config)

        async with finder_agent:
            llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)

            await llm.generate_str(
                message="Print the first 2 paragraphs of https://www.anthropic.com/research/building-effective-agents",
            )

            # Multi-turn conversations
            await llm.generate_str(
                message="Summarize those paragraphs in a 128 character tweet",
            )


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
