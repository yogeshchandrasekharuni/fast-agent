"""
Example MCP Agent application showing router workflow with decorator syntax.
Demonstrates router's ability to either:
1. Use tools directly to handle requests
2. Delegate requests to specialized agents
"""

import asyncio
from mcp_agent.core.decorator_app import MCPAgentDecorator

# Create the application
agent_app = MCPAgentDecorator(
    "Router Workflow Example",
)

# Sample requests demonstrating direct tool use vs agent delegation
SAMPLE_REQUESTS = [
    "Download and summarize https://llmindset.co.uk/",  # Router handles directly with fetch
    "Analyze the quality of our Python codebase in the current working directory",  # Delegated to code expert
    "What are the key principles of system design?",  # Delegated to general assistant
]


@agent_app.agent(
    name="code_expert",
    instruction="""You are an expert in code analysis and software engineering.
    When asked about code, architecture, or development practices,
    you provide thorough and practical insights.""",
    servers=["filesystem"],
)
@agent_app.agent(
    name="general_assistant",
    instruction="""You are a knowledgeable assistant that provides clear,
    well-reasoned responses about general topics, concepts, and principles.""",
)
@agent_app.router(
    name="llm_router",
    model="gpt-4o",
    agents=["code_expert", "general_assistant"],
    servers=["fetch"],  # Router can use fetch directly
)
async def main():
    async with agent_app.run() as agent:
        for request in SAMPLE_REQUESTS:
            await agent.send("llm_router", request)


if __name__ == "__main__":
    asyncio.run(main())
