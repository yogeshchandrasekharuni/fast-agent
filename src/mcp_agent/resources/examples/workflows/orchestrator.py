"""
This demonstrates creating multiple agents and an orchestrator to coordinate them.
"""

import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Orchestrator-Workers")


# Define worker agents
@fast.agent(
    name="finder",
    instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
    servers=["fetch", "filesystem"],
    model="gpt-4o-mini",
)
@fast.agent(
    name="writer",
    instruction="""You are an agent that can write to the filesystem.
            You are tasked with taking the user's input, addressing it, and 
            writing the result to disk in the appropriate location.""",
    servers=["filesystem"],
)
@fast.agent(
    name="proofreader",
    instruction=""""Review the short story for grammar, spelling, and punctuation errors.
            Identify any awkward phrasing or structural issues that could improve clarity. 
            Provide detailed feedback on corrections.""",
    servers=["fetch"],
    model="gpt-4o",
)
# Define the orchestrator to coordinate the other agents
@fast.orchestrator(
    name="orchestrate",
    instruction="""Load the student's short story from short_story.md, 
    and generate a report with feedback across proofreading, 
    factuality/logical consistency and style adherence. Use the style rules from 
    https://apastyle.apa.org/learn/quick-guide-on-formatting and 
    https://apastyle.apa.org/learn/quick-guide-on-references.
    Write the graded report to graded_report.md in the same directory as short_story.md""",
    agents=["finder", "writer", "proofreader"],
    model="sonnet",
)
async def main():
    async with fast.run() as agent:
        # The orchestrator can be used just like any other agent
        task = (
            """Load the student's short story from short_story.md, 
        and generate a report with feedback across proofreading, 
        factuality/logical consistency and style adherence. Use the style rules from 
        https://apastyle.apa.org/learn/quick-guide-on-formatting and 
        https://apastyle.apa.org/learn/quick-guide-on-references.
        Write the graded report to graded_report.md in the same directory as short_story.md""",
        )

        # Send the task
        await agent.orchestrate(task)


if __name__ == "__main__":
    asyncio.run(main())
