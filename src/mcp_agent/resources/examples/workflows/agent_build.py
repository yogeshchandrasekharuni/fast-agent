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
than one level deep. Your ultimate goal will be to produce a single '.py' agent in the style
shown to you that fulfils the Human's needs.
Keep the application simple, define agents with appropriate  MCP Servers, Tools  and the Human Input Tool.
The style of the program should be like the examples you have been showm, very little additional code (use
very simple Python where necessary). """,
    servers=["filesystem", "fetch"],
)
# Define worker agents
@fast.agent(
    "requirements_capture",
    instruction="""
You help the Human define their requirements for building Agent based systems. Keep questions short and
simple, collaborate with the agent_expert or other agents in the workflow to refine human interaction. 
Keep requests to the Human simple and minimal. """,
    human_input=True,
)
# Define the orchestrator to coordinate the other agents
@fast.orchestrator(
    name="orchestrator_worker",
    agents=["agent_expert", "requirements_capture"],
    model="sonnet",
)
async def main():
    async with fast.run() as agent:
        await agent.agent_expert("""
- Read this paper: https://www.anthropic.com/research/building-effective-agents" to understand
the principles of Building Effective Agents. 
- Read and examing the sample agent and workflow definitions in the current directory:
  - chaining.py - simple agent chaining example.
  - parallel.py - parallel agents example.
  - evaluator.py - evaluator optimizer example.
  - orchestrator.py - complex orchestration example.
  - router.py - workflow routing example.
- Load the 'fastagent.config.yaml' file to see the available and configured MCP Servers.
When producing the agent/workflow definition, keep to a simple single .py file in the style
of the examples.
        """)

        await agent.orchestrator_worker(
            "Write an Agent program that fulfils the Human's needs."
        )


if __name__ == "__main__":
    asyncio.run(main())
