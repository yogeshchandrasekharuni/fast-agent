"""
This demonstrates creating multiple agents and an orchestrator to coordinate them.
"""

import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Agent Builder")


@fast.agent(
    "agent_expert",
    instruction="""
You design agent workflows, using the practices from 'Building Effective Agents'. You provide concise
specific guidance on design and composition. Prefer simple solutions, and don't nest workflows more 
than one deep. Your ultimate goal will be to produce a single '.py' agent that fulfils the Human's needs.""",
    servers=["filesystem", "fetch"],
)
# Define worker agents
@fast.agent(
    "requirements_capture",
    instruction="""
You help the Human define their requirements for building Agent based systems. The agent_expert can 
answer questions about the style and limitations of agents. Request simple inputs from the Human. """,
    human_input=True,
)
# Define the orchestrator to coordinate the other agents
@fast.orchestrator(
    name="orchestrate",
    agents=["agent_expert", "requirements_capture"],
    model="sonnet",
)
async def main():
    async with fast.run() as agent:
        await agent.agent_expert("""
Read this paper: https://www.anthropic.com/research/building-effective-agents" to understand
the principles of Building Effective Agents. Then, look at all the .py files in the current
directory. They provide examples of how to compose simple agent and workflow programs.
        """)

        await agent.orchestrate(
            "Write an Agent program that fulfils the Human's needs."
        )


if __name__ == "__main__":
    asyncio.run(main())
