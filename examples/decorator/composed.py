"""
Example MCP Agent application showing simplified agent access including reused agents.
"""

import asyncio
from mcp_agent.core.decorator_app import FastAgent

# Create the application
agent_app = FastAgent(
    "Parallel Workflow Example",
    # config={
    #     "human_input_handler": None  # Disable human input handling
    # },
)
agent_app.app._human_input_callback = None

SHORT_STORY = """
The Battle of Glimmerwood

In the heart of Glimmerwood, a mystical forest knowed for its radiant trees, a small village thrived. 
The villagers, who were live peacefully, shared their home with the forest's magical creatures, 
especially the Glimmerfoxes whose fur shimmer like moonlight.

One fateful evening, the peace was shaterred when the infamous Dark Marauders attack. 
Lead by the cunning Captain Thorn, the bandits aim to steal the precious Glimmerstones which was believed to grant immortality.

Amidst the choas, a young girl named Elara stood her ground, she rallied the villagers and devised a clever plan.
Using the forests natural defenses they lured the marauders into a trap. 
As the bandits aproached the village square, a herd of Glimmerfoxes emerged, blinding them with their dazzling light, 
the villagers seized the opportunity to captured the invaders.

Elara's bravery was celebrated and she was hailed as the "Guardian of Glimmerwood". 
The Glimmerstones were secured in a hidden grove protected by an ancient spell.

However, not all was as it seemed. The Glimmerstones true power was never confirm, 
and whispers of a hidden agenda linger among the villagers.
"""

SECOND_STORY = """
The Lost Library

Deep beneath the city streets, hidden from the world above, lay an ancient library filled with forgotten knowledge. 
The library was guarded by the mysterious Order of Keepers, who dedicated their lifes to preserving its secrets.

One stormy night, a young apprentice named Marcus discovered a peculiar book that glowed with an eerie blue light. 
As he opened its pages, strange symbols began to float off the parchment and dance through the air.

The symbols revealed a map leading to a chamber that noone knew existed. With trembling hands, 
Marcus followed the floating lights deeper into the library's maze-like corridors.

What he found in that hidden chamber would change everything - a collection of texts that suggested 
the Order itself had been responsible for erasing certain knowledge from history.

Now Marcus faced a difficult choice: maintain his loyalty to the Order or expose the truth to the world above.
"""


@agent_app.agent(
    name="proofreader",
    instruction=""""Review stories for grammar, spelling, and punctuation errors.
    Identify any awkward phrasing or structural issues that could improve clarity. 
    Provide detailed feedback on corrections.""",
)
@agent_app.agent(
    name="fact_checker_oai",
    model="gpt-4o",
    history=False,
    instruction="""Verify the factual consistency within the story. Identify any contradictions,
    logical inconsistencies, or inaccuracies in the plot, character actions, or setting. 
    Highlight potential issues with reasoning or coherence.""",
)
@agent_app.agent(
    name="fact_checker_haiku",
    model="haiku",
    history=False,
    instruction="""Verify the factual consistency within the story. Identify any contradictions,
    logical inconsistencies, or inaccuracies in the plot, character actions, or setting. 
    Highlight potential issues with reasoning or coherence.""",
)
@agent_app.agent(
    name="fact_checker_combined",
    model="sonnet",
    history=False,
    instruction="""Compare and consolidate the 2 fact checks, incorporate all points from both.""",
)
@agent_app.parallel(
    name="fact_checker",
    fan_out=["fact_checker_oai", "fact_checker_haiku"],
    fan_in="fact_checker_combined",
)
@agent_app.agent(
    name="style_enforcer",
    model="sonnet",
    instruction="""Analyze the story for adherence to style guidelines.
    Evaluate the narrative flow, clarity of expression, and tone. Suggest improvements to 
    enhance storytelling, readability, and engagement.""",
)
@agent_app.agent(
    name="first_story_grader",
    model="o3-mini.high",
    instruction="""Compile the feedback from the Proofreader, Fact Checker, and Style Enforcer
    into a structured report for the first story. Summarize key issues and categorize them by type. 
    Provide actionable recommendations for improving the story, 
    and give an overall grade based on the feedback.""",
)
@agent_app.agent(
    name="second_story_grader",
    model="o3-mini.high",
    instruction="""Compile the feedback from the Proofreader, Fact Checker, and Style Enforcer
    into a structured report for the second story. Summarize key issues and categorize them by type. 
    Provide actionable recommendations for improving the story, 
    and give an overall grade based on the feedback.""",
)
@agent_app.agent(
    name="final_comparison",
    model="sonnet",
    instruction="""Compare the analysis of both stories and provide a comparative assessment.
    Highlight strengths and weaknesses of each, noting any common issues or unique challenges.
    Recommend which story shows more promise and why.""",
)
@agent_app.parallel(
    fan_out=["proofreader", "fact_checker", "style_enforcer"],
    fan_in="first_story_grader",
    name="story_one_analysis",
)
@agent_app.parallel(
    fan_out=[
        "proofreader",
        "fact_checker",
        "style_enforcer",
    ],  # Reusing the same agents
    fan_in="second_story_grader",
    name="story_two_analysis",
)
@agent_app.parallel(
    fan_out=["story_one_analysis", "story_two_analysis"],
    fan_in="final_comparison",
    name="comparative_analysis",
)
async def main():
    # Use the app's context manager
    async with agent_app.run() as agent:
        # Send first story
        await agent.send(
            "story_one_analysis", f"student short story submission: {SHORT_STORY}"
        )

        # Send second story.
        # TODO -- decide how to handle/expose conversation history options to the User. perhaps a reset()?
        await agent.send(
            "story_two_analysis", f"student short story submission: {SECOND_STORY}"
        )

        # Get comparative analysis
        await agent.send(
            "comparative_analysis", "Please compare the analyses of both stories."
        )

        # Allow for follow-up questions
        await agent.prompt("final_comparison", default="STOP")


if __name__ == "__main__":
    asyncio.run(main())
