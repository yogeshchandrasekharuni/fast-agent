"""
Example MCP Agent application showing simplified agent access.
"""

import asyncio
from mcp_agent.core.decorator_app import MCPAgentDecorator

# Create the application
agent_app = MCPAgentDecorator("Interactive Agent Example")


@agent_app.agent(
    name="proofreader",
    instruction=""""Review the short story for grammar, spelling, and punctuation errors.
    Identify any awkward phrasing or structural issues that could improve clarity. 
    Provide detailed feedback on corrections.""",
)
@agent_app.agent(
    name="fact_checker",
    instruction="""Verify the factual consistency within the story. Identify any contradictions,
    logical inconsistencies, or inaccuracies in the plot, character actions, or setting. 
    Highlight potential issues with reasoning or coherence.""",
)
@agent_app.agent(
    name="style_enforcer",
    instruction="""Analyze the story for adherence to style guidelines.
    Evaluate the narrative flow, clarity of expression, and tone. Suggest improvements to 
    enhance storytelling, readability, and engagement.""",
)
@agent_app.agent(
    name="grader",
    instruction="""Compile the feedback from the Proofreader, Fact Checker, and Style Enforcer
    into a structured report. Summarize key issues and categorize them by type. 
    Provide actionable recommendations for improving the story, 
    and give an overall grade based on the feedback.""",
)
@agent_app.parallel(
    fan_out=["proofreader", "factchecker", "style_enforcer"],
    fan_in=["grader"],
    name="foo",
)
async def main():
    # Use the app's context manager
    async with agent_app.run() as agent:
        # await agent("print the next number in the sequence")
        # await agent.prompt(default="STOP")
        await agent("foo").send("grade the essay")


if __name__ == "__main__":
    asyncio.run(main())
